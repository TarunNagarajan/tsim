"""Hand-built CTA phase/exit references and adversarial capture mutations."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from execution_events import summarize_events
from cta_barrier_report import observed_controls

class Stream:
    def __init__(self, threads=64):
        self.rows = []
        self.add(type='stream', schema=1, dependency_schema=2, cta_barrier_schema=1)
        self.add(type='launch', launch=1, kernel='k', grid=[1, 1, 1], block=[threads, 1, 1])

    def add(self, **fields):
        self.rows.append(dict(sequence=len(self.rows)+1, **fields))
        return len(self.rows)

    def issue(self, warp, instruction, lanes=0xffffffff, pc=50, path=None):
        is_red = 'red.' in instruction
        guarded = instruction.startswith('@')
        return self.add(type='issue', launch=1, kernel='k', function='k', frame=0, block=[0, 0, 0],
            warp=warp, pc=pc, ptx_line=pc+1, path_mask=lanes if path is None else path, execution_mask=lanes,
            instruction=instruction, defs=[0] if is_red else [], uses=[4] if is_red else [],
            predicate_use=2 if guarded else -1, virtual_registers32=4, virtual_registers64=0,
            declared_local_bytes=0, expected_memory_accesses=0, expected_memory_width=0,
            implicit_uses=[], implicit_defs=[])

    def end(self, issue): self.add(type='issue_end', issue=issue, status='completed')

    def arrive(self, warp, barrier=3, phase=0, count=0, reduction=0, votes=0, blocking=True, lanes=0xffffffff):
        opcode = 'bar.red.'+{1:'and.pred', 2:'or.pred', 3:'popc.u32'}[reduction] if reduction else 'bar.sync' if blocking else 'bar.arrive'
        operands = ('%r0, ' if reduction else '')+str(barrier)+(', '+str(count) if count else '')+(', %p2' if reduction else '')
        issue = self.issue(warp, opcode+' '+operands+';', lanes)
        ref = self.add(type='cta_barrier_arrive', issue=issue, barrier=barrier, phase=phase, count=count,
            reduction=reduction, blocking=blocking, execution_mask=lanes, true_votes=votes)
        self.end(issue)
        return ref

    def release(self, arrivals, participants, waiters, result=0, exits=()):
        first = self.rows[arrivals[0]-1]
        return self.add(type='cta_barrier_release', launch=1, kernel='k', block=[0, 0, 0],
            barrier=first['barrier'], phase=first['phase'], count=first['count'], reduction=first['reduction'],
            arrivals=list(arrivals), exits=list(exits), participants_mask=participants, waiters_mask=waiters, result=result)

    def result(self, arrival, value):
        original = self.rows[self.rows[arrival-1]['issue']-1]
        issue = self.issue(original['warp'], original['instruction'], original['execution_mask'])
        self.add(type='cta_barrier_result', issue=issue, arrival=arrival, execution_mask=original['execution_mask'], result=value)
        self.end(issue)

    def retire(self, warp, lanes=0xffffffff, partial=False):
        issue = self.issue(warp, '@%p1 ret;' if partial else 'ret;', lanes, pc=99, path=0xffffffff if partial else None)
        ref = self.add(type='cta_exit', issue=issue, launch=1, kernel='k', block=[0, 0, 0], warp=warp, execution_mask=lanes)
        self.end(issue)
        return ref

    def close(self):
        self.add(type='launch_end', launch=1, status='completed'); self.add(type='stream_end')
        return self.rows

def reductions():
    s = Stream()
    for phase, reduction, votes, value in [(0, 3, (17, 5), 22), (1, 1, (32, 31), 0), (2, 2, (0, 1), 1)]:
        a = s.arrive(0, phase=phase, reduction=reduction, votes=votes[0])
        b = s.arrive(1, phase=phase, reduction=reduction, votes=votes[1])
        s.release([a, b], 3, 3, value)
        s.result(b, value); s.result(a, value)
    s.retire(0); s.retire(1)
    return s.close()

def producer():
    s = Stream()
    a = s.arrive(0, count=64, blocking=False)
    s.retire(0) # arrival remains valid even though the nonwaiting producer exits
    b = s.arrive(1, count=64)
    s.release([a, b], 3, 2)
    s.retire(1)
    return s.close()

def early_exit():
    s = Stream()
    a = s.arrive(0)
    gone = s.retire(1)
    s.release([a], 1, 1, exits=[gone]) # no instruction scope owns this release
    s.retire(0)
    return s.close()

def partial():
    s = Stream(40)
    s.retire(0, 0xffff0000, partial=True)
    a = s.arrive(0, reduction=3, votes=5, lanes=0xffff)
    b = s.arrive(1, reduction=3, votes=8, lanes=0xff)
    s.release([a,b], 3, 3, 13)
    s.result(a,13); s.result(b,13)
    s.retire(0, 0xffff); s.retire(1, 0xff)
    return s.close()

def independent():
    s = Stream(128)
    a = s.arrive(0, barrier=1, count=64)
    b = s.arrive(2, barrier=2, count=64)
    c = s.arrive(3, barrier=2, count=64)
    s.release([b,c], 12, 12)
    s.retire(2); s.retire(3)
    d = s.arrive(1, barrier=1, count=64)
    s.release([a,d], 3, 3)
    s.retire(0); s.retire(1)
    return s.close()

def verify_cta_event_components():
    checks = []
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder)/'events.jsonl'
        def summary(data):
            path.write_text(''.join(json.dumps(row)+'\n' for row in data))
            return summarize_events(path)
        def accept(name, data, expected):
            report = summary(data)
            assert report['complete_supported_cta_barriers'] and report['complete_supported_memory'], name
            assert all(report['cta_barriers'][key] == value for key, value in expected.items()), name
            checks.append(name)
        def reject(name, original, mutate):
            data = deepcopy(original); mutate(data)
            try: summary(data)
            except ValueError: checks.append(name)
            else: raise AssertionError(name)
        accept('saved popc/and/or results survive phase reuse', reductions(), {'arrivals':6, 'releases':3, 'result_reads':6, 'retired_lanes':64})
        accept('counted nonwaiting producer exits without gaining waiter acquire', producer(), {'released_waiters':1, 'nonblocking_arrivals':1})
        accept('outside-scope exit releases exactly its whole-CTA waiter', early_exit(), {'exit_dependencies':1, 'released_waiters':1})
        phases, gates = observed_controls(early_exit())
        assert len(phases) == len(gates) == 1 and len(gates[0]['arrival_issues']) == len(gates[0]['exit_issues']) == 1
        assert gates[0]['issue'] > phases[0]['release'] > gates[0]['exit_issues'][0]
        checks.append('control gate joins exact arrival and missing-warp exit origins')
        accept('partial warp exits and live predicate counts are conserved', partial(), {'retired_lanes':40, 'result_reads':2})
        accept('independent counted IDs can release in different orders', independent(), {'releases':2, 'released_waiters':4})
        phases, gates = observed_controls(independent())
        assert [p['barrier'] for p in phases] == [2,1] and len(gates) == 4
        assert all(len(g['arrival_issues']) == 2 and not g['exit_issues'] for g in gates)
        checks.append('independent counted control gates preserve phase ownership')
        convergence = Stream(32)
        attempt = convergence.issue(0, 'bar.sync 3;', 0xffff)
        convergence.add(type='cta_barrier_wait', issue=attempt, execution_mask=0xffff)
        convergence.end(attempt)
        arrival = convergence.arrive(0)
        convergence.release([arrival], 1, 1); convergence.retire(0)
        accept('partial selection parks without inventing a named arrival', convergence.close(), {'convergence_waits':1, 'arrivals':1})
        base = reductions()
        def rows(data, kind): return [r for r in data if r['type'] == kind]
        for name, field, value in [('wrong ID', 'barrier', 4), ('stale phase', 'phase', 1), ('boolean phase', 'phase', True),
                                   ('wrong count form', 'count', 64), ('wrong reduction', 'reduction', 2),
                                   ('wrong blocking form', 'blocking', False), ('invented vote lanes', 'true_votes', 33),
                                   ('wrong executing lanes', 'execution_mask', 0xffff)]:
            reject(name, base, lambda d, f=field, v=value: rows(d,'cta_barrier_arrive')[0].update({f:v}))
        for name, field, value in [('borrowed participant', 'participants_mask', 7), ('omitted waiter', 'waiters_mask', 1),
                                   ('stale release phase', 'phase', 1), ('changed release count', 'count', 64),
                                   ('changed release operation', 'reduction', 2), ('wrong reduction answer', 'result', 23),
                                   ('missing arrivals', 'arrivals', []), ('invented exits', 'exits', [1]),
                                   ('crossed block', 'block', [1,0,0])]:
            reject(name, base, lambda d, f=field, v=value: rows(d,'cta_barrier_release')[0].update({f:v}))
        reject('missing actual arrival action', base, lambda d: rows(d,'cta_barrier_arrive')[0].update(type='coverage_gap', reason='removed'))
        reject('changed saved result', base, lambda d: rows(d,'cta_barrier_result')[0].update(result=23))
        reject('result from wrong frame', base, lambda d: d[rows(d,'cta_barrier_result')[0]['issue']-1].update(frame=1))
        reject('result from wrong PC', base, lambda d: d[rows(d,'cta_barrier_result')[0]['issue']-1].update(pc=51))
        reject('changed result origin', base, lambda d: rows(d,'cta_barrier_result')[0].update(arrival=1))
        reject('missing retirement action', base, lambda d: rows(d,'cta_exit')[0].update(type='coverage_gap', reason='removed'))
        reject('device return cannot retire kernel lanes', base, lambda d: d[rows(d,'cta_exit')[0]['issue']-1].update(frame=1, function='callee'))
        reject('exit ownership cannot cross warps', base, lambda d: rows(d,'cta_exit')[0].update(warp=1))
        reject('exit dependency cannot be omitted', early_exit(), lambda d: rows(d,'cta_barrier_release')[0].update(exits=[]))
        reject('exited lanes cannot arrive', partial(), lambda d: d[rows(d,'cta_barrier_arrive')[0]['issue']-1].update(path_mask=0xffffffff))
        reject('unknown CTA schema', base, lambda d: d[0].update(cta_barrier_schema=2))
        reject('boolean CTA schema', base, lambda d: d[0].update(cta_barrier_schema=True))
        too_early = Stream()
        arrival = too_early.arrive(0, count=64)
        too_early.release([arrival], 1, 1)
        reject('counted target cannot borrow missing warp credit', too_early.close(), lambda d: None)
        too_early = Stream()
        arrival = too_early.arrive(0)
        too_early.release([arrival], 1, 1)
        reject('whole CTA cannot release while another warp is live', too_early.close(), lambda d: None)
        duplicate = Stream()
        duplicate.arrive(0, count=64, blocking=False); duplicate.arrive(0, count=64, blocking=False)
        reject('nonblocking producer cannot double-arrive in a phase', duplicate.close(), lambda d: None)
        pending_result = deepcopy(base)
        release = rows(pending_result, 'cta_barrier_release')[0]
        sequence = release['sequence']
        release.update(type='unmodeled_control')
        # Keep a valid stream event so the next issued result tests waiter
        # readiness rather than merely failing an unknown-type check.
        release.clear(); release.update(sequence=sequence, type='launch', launch=2, kernel='unused', grid=[1,1,1], block=[1,1,1])
        reject('a reduction waiter cannot issue before its release', pending_result, lambda d: None)
        implicit = Stream()
        arrival = implicit.arrive(0)
        exited = implicit.add(type='cta_exit', issue=0, launch=1, kernel='k', block=[0,0,0], warp=1, execution_mask=0xffffffff)
        implicit.release([arrival], 1, 1, exits=[exited]); implicit.retire(0)
        report = summary(implicit.close())
        assert report['complete_supported_memory'] and not report['complete_supported_cta_barriers'] and report['cta_barriers']['implicit_exits'] == 1
        checks.append('implicit exits preserve causal identity but refuse complete instruction mapping')
        unused = Stream()
        unused.arrive(0, count=96, blocking=False); unused.retire(0); unused.retire(1)
        report = summary(unused.close())
        assert report['complete_supported_memory'] and not report['complete_supported_cta_barriers']
        checks.append('unused incomplete nonblocking phases remain explicit')
        capped = deepcopy(base[:6]); capped.append({'type':'truncated','limit':6})
        assert not summary(capped)['complete_supported_cta_barriers']
        checks.append('capture cap never implies a completed CTA phase')
        incomplete = deepcopy(early_exit())
        incomplete[-2]['status'] = 'fault'
        assert not summary(incomplete)['complete_supported_cta_barriers']
        checks.append('fault boundary never implies complete CTA synchronization')
    return checks

if __name__ == '__main__':
    checks = verify_cta_event_components()
    print('PASS CTA transition references checks='+str(len(checks)))
