"""Validate actual CTA rendezvous, saved results and retirement causality."""
from collections import Counter
import math
import re

def plain(instruction):
    return re.sub(r'^\s*@\s*!?\s*\S+\s+', '', instruction).split('//')[0].strip()

def barrier_form(instruction):
    parts = plain(instruction).split()[0].rstrip(';').split('.')
    if parts[0] not in ['bar', 'barrier']: return None
    modifiers = [p for p in parts[1:] if p not in ['cta', 'aligned', 'u32', 'pred']]
    if modifiers in [['sync'], ['arrive']]: return modifiers[0], 0
    if len(modifiers) == 2 and modifiers[0] == 'red' and modifiers[1] in ['and', 'or', 'popc']:
        return 'red', {'and': 1, 'or': 2, 'popc': 3}[modifiers[1]]
    return None

def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum: raise ValueError('Invalid CTA ' + name)
    return value

def mask(value, name):
    if integer(value, name) >= 1 << 32: raise ValueError('CTA capture requires 32-bit lane/warp masks')
    return value

class CTABarrierEvents:
    def __init__(self, issues, launches):
        self.issues, self.launches = issues, launches
        self.blocks, self.roles, self.arrivals, self.exits = {}, {}, {}, {}
        self.counts = Counter()
        self.complete_launches = 0
        self.incomplete_launches = 0

    def context(self, launch, block, kernel=None):
        integer(launch, 'launch', 1)
        if launch not in self.launches: raise ValueError('CTA transition outside launch')
        owner = self.launches[launch]
        if kernel is not None and kernel != owner['kernel']: raise ValueError('CTA kernel ownership differs')
        if not isinstance(block, list) or len(block) != 3: raise ValueError('Invalid CTA block identity')
        for coordinate, extent in zip(block, owner['grid']):
            if integer(coordinate, 'coordinate') >= integer(extent, 'grid extent', 1): raise ValueError('CTA block outside grid')
        key = (launch, tuple(block))
        if key not in self.blocks:
            threads = math.prod(integer(v, 'block extent', 1) for v in owner['block'])
            if not 1 <= threads <= 1024: raise ValueError('CTA transition block size unsupported')
            live = [(1 << min(32, threads - start)) - 1 for start in range(0, threads, 32)]
            self.blocks[key] = {'live': live, 'phases': {}, 'blocked': {}, 'results': {}, 'last_exit': {}}
        return key, self.blocks[key]

    def issue(self, event):
        key, state = self.context(event['launch'], event['block'], event['kernel'])
        warp = integer(event['warp'], 'warp')
        if warp >= len(state['live']) or event['path_mask'] & ~state['live'][warp]:
            raise ValueError('CTA instruction from retired or nonexistent lanes')
        if warp in state['blocked']: raise ValueError('CTA waiter resumed before its phase released')
        if warp in state['results'] and barrier_form(event['instruction']) != ('red', self.arrivals[state['results'][warp]]['reduction']):
            raise ValueError('CTA reduction waiter resumed without collecting its saved result')

    def record(self, event):
        try: self._record(event)
        except (KeyError, TypeError, IndexError, AttributeError) as error:
            raise ValueError('Malformed CTA transition') from error

    def _record(self, event):
        kind = event['type']
        if kind == 'cta_barrier_release':
            self.release(event)
            return
        issue = integer(event['issue'], 'issue')
        if kind == 'cta_exit' and issue == 0:
            self.retire(event, None)
            self.counts['implicit_exits'] += 1
            return
        parent = self.issues[issue]
        if issue in self.roles: raise ValueError('Multiple CTA actions in one instruction attempt')
        key, state = self.context(parent['launch'], parent['block'])
        warp = parent['warp']
        lanes = mask(event['execution_mask'], 'execution mask')
        if not lanes or lanes != parent['execution_mask']: raise ValueError('CTA action participation differs from parent')
        if kind == 'cta_exit':
            self.retire(event, parent)
        else:
            form = barrier_form(parent['instruction'])
            if not form: raise ValueError('CTA action parent is not a CTA barrier')
            if kind == 'cta_barrier_wait':
                if form[0] == 'arrive' and lanes == state['live'][warp]:
                    raise ValueError('A converged nonblocking arrival cannot be a convergence wait')
                self.counts['convergence_waits'] += 1
            elif kind == 'cta_barrier_arrive':
                if lanes != state['live'][warp]: raise ValueError('CTA arrival lacks warp convergence')
                self.arrive(event, parent, key, state, form)
            elif kind == 'cta_barrier_result':
                arrival = integer(event['arrival'], 'arrival reference', 1)
                if state['results'].get(warp) != arrival: raise ValueError('CTA result lacks a released waiter')
                saved = self.arrivals[arrival]
                original = self.issues[saved['issue']]
                if any(parent[field] != original[field] for field in ['pc', 'frame', 'function', 'instruction', 'defs', 'uses']):
                    raise ValueError('CTA result belongs to a different instruction/frame')
                if integer(event['result'], 'result') != saved['result']: raise ValueError('CTA saved reduction result changed')
                del state['results'][warp]
                self.counts['result_reads'] += 1
            else: raise ValueError('Unknown CTA transition')
        self.roles[issue] = kind

    def retire(self, event, parent):
        key, state = self.context(event['launch'], event['block'], event['kernel'])
        warp = integer(event['warp'], 'exit warp')
        lanes = mask(event['execution_mask'], 'exit lanes')
        if warp >= len(state['live']) or not lanes or lanes & ~state['live'][warp]:
            raise ValueError('CTA retirement repeats or invents lanes')
        if warp in state['blocked'] or warp in state['results']: raise ValueError('CTA waiter exited before release/result')
        if parent:
            if (parent['launch'], tuple(parent['block']), parent['warp']) != (key[0], key[1], warp):
                raise ValueError('CTA exit ownership differs from parent')
            if parent['function'] != event['kernel'] or parent['frame'] != 0 or plain(parent['instruction']).split()[0] not in ['ret;', 'ret', 'ret.uni;', 'ret.uni', 'exit;', 'exit']:
                raise ValueError('CTA exit must come from kernel retirement, not a device return')
        state['live'][warp] &= ~lanes
        state['last_exit'][warp] = event['sequence']
        self.exits[event['sequence']] = (key, warp)
        self.counts['exit_records'] += 1
        self.counts['retired_lanes'] += lanes.bit_count()

    def arrive(self, event, parent, key, state, form):
        barrier = integer(event['barrier'], 'barrier')
        if barrier >= 16: raise ValueError('CTA barrier outside 0..15')
        phase = integer(event['phase'], 'phase')
        count = integer(event['count'], 'thread count')
        if count % 32: raise ValueError('CTA count must be a multiple of 32')
        if event['reduction'] != form[1] or type(event['reduction']) is not int:
            raise ValueError('CTA reduction differs from parent opcode')
        if type(event['blocking']) is not bool or event['blocking'] != (form[0] != 'arrive'):
            raise ValueError('CTA blocking form differs from parent opcode')
        votes = integer(event['true_votes'], 'votes')
        if votes > event['execution_mask'].bit_count() or (not form[1] and votes): raise ValueError('Invalid CTA predicate vote count')
        operands = [operand.strip().rstrip(';') for operand in plain(parent['instruction']).split(None, 1)[1].split(',')]
        operands = operands[1:] if form[0] == 'red' else operands
        have_count = len(operands) == (3 if form[0] == 'red' else 2)
        if have_count != bool(count) or (form[0] == 'arrive' and not count): raise ValueError('CTA count form differs from parent operands')
        for value, operand in [(barrier, operands[0])] + ([(count, operands[1])] if have_count else []):
            if re.fullmatch(r'(?:0[xX][0-9a-fA-F]+|[0-9]+)', operand):
                if int(operand, 16 if operand.lower().startswith('0x') else 10) != value:
                    raise ValueError('CTA immediate operand differs from actual arrival')
        active = state['phases'].setdefault(barrier, {'phase': 0, 'arrivals': {}})
        if active['phase'] != phase: raise ValueError('CTA phase skipped, repeated or stale')
        warp = parent['warp']
        if warp in active['arrivals']: raise ValueError('Duplicate warp arrival in active CTA phase')
        if active['arrivals'] and (active['count'], active['reduction']) != (count, form[1]):
            raise ValueError('Mixed counts or operations in active CTA phase')
        active.update(count=count, reduction=form[1])
        active['arrivals'][warp] = event['sequence']
        self.arrivals[event['sequence']] = dict(event, owner=key, warp=warp)
        if event['blocking']: state['blocked'][warp] = event['sequence']
        self.counts['arrivals'] += 1
        self.counts['blocking_arrivals' if event['blocking'] else 'nonblocking_arrivals'] += 1

    def release(self, event):
        key, state = self.context(event['launch'], event['block'], event['kernel'])
        barrier = integer(event['barrier'], 'release barrier')
        active = state['phases'].get(barrier)
        if not active or not active['arrivals'] or integer(event['phase'], 'release phase') != active['phase']:
            raise ValueError('CTA release has no matching active phase')
        for name in ['count', 'reduction']:
            if type(event[name]) is not int or event[name] != active[name]: raise ValueError('CTA release configuration changed')
        ordered = sorted(active['arrivals'].items())
        references = [reference for _, reference in ordered]
        participants = sum(1 << warp for warp, _ in ordered)
        waiters = sum(1 << warp for warp, ref in ordered if self.arrivals[ref]['blocking'])
        if mask(event['participants_mask'], 'participants') != participants or mask(event['waiters_mask'], 'waiters') != waiters:
            raise ValueError('CTA release participant/waiter conservation failed')
        for name in ['arrivals', 'exits']:
            if not isinstance(event[name], list) or any(type(v) is not int or v <= 0 for v in event[name]):
                raise ValueError('Invalid CTA causal references')
        if event['arrivals'] != references: raise ValueError('CTA release arrival references differ from phase')
        expected_exits = []
        if active['count']:
            if len(ordered) * 32 != active['count']: raise ValueError('CTA counted phase released before its target')
        else:
            for warp, live in enumerate(state['live']):
                if not participants & (1 << warp):
                    if live: raise ValueError('CTA whole-block phase released before every live warp arrived')
                    expected_exits.append(state['last_exit'][warp])
        if event['exits'] != expected_exits: raise ValueError('CTA release exit references differ from retired participants')
        votes = [self.arrivals[ref] for ref in references]
        result = (int(all(v['true_votes'] == v['execution_mask'].bit_count() for v in votes)) if active['reduction'] == 1 else
                  int(any(v['true_votes'] for v in votes)) if active['reduction'] == 2 else
                  sum(v['true_votes'] for v in votes) if active['reduction'] == 3 else 0)
        if integer(event['result'], 'release result') != result: raise ValueError('CTA reduction vote/result conservation failed')
        for warp, reference in ordered:
            if self.arrivals[reference]['blocking']:
                if state['blocked'].pop(warp, None) != reference: raise ValueError('CTA release lost its waiting owner')
                if active['reduction']: state['results'][warp] = reference
            self.arrivals[reference]['result'] = result
            self.arrivals[reference]['release'] = event['sequence']
        active['arrivals'] = {}
        active['phase'] += 1
        self.counts['releases'] += 1
        self.counts['released_waiters'] += waiters.bit_count()
        self.counts['exit_dependencies'] += len(expected_exits)

    def finish_issue(self, issue, status):
        parent = self.issues[issue]
        if status != 'completed' or not parent['execution_mask']: return
        is_exit = parent['frame'] == 0 and parent['function'] == parent['kernel'] and plain(parent['instruction']).split()[0].rstrip(';') in ['ret', 'ret.uni', 'exit']
        if (barrier_form(parent['instruction']) or is_exit) and issue not in self.roles:
            raise ValueError('CTA instruction lacks actual arrival/wait/result/retirement action')

    def finish_launch(self, launch, status):
        if status != 'completed': return
        complete = sum(owner == launch for owner, _ in self.blocks) == math.prod(self.launches[launch]['grid'])
        for (owner, _), state in self.blocks.items():
            if owner != launch: continue
            if state['blocked'] or state['results']: raise ValueError('Completed launch retains CTA waiters or unread results')
            complete &= not any(state['live'])
            complete &= not any(phase['arrivals'] for phase in state['phases'].values())
        if complete: self.complete_launches += 1
        else: self.incomplete_launches += 1
