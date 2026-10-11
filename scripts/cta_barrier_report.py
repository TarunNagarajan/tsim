"""Export observed CTA phase causes and the next selected waiter instruction."""
import hashlib
import json
from pathlib import Path
from execution_events import summarize_events
from memory_model import validate_compilation_artifacts

def observed_controls(events):
    issues, arrivals, exits, pending = {}, {}, {}, {}
    phases, gates = [], []
    for event in events:
        kind = event['type']
        if kind == 'issue':
            issues[event['sequence']] = event
            actor = (event['launch'], tuple(event['block']), event['warp'])
            if actor in pending:
                phase = pending.pop(actor)
                gates.append({'issue':event['sequence'], 'release':phase['release'],
                              'arrival_issues':phase['arrival_issues'], 'exit_issues':phase['exit_issues']})
        elif kind == 'cta_barrier_arrive': arrivals[event['sequence']] = event
        elif kind == 'cta_exit': exits[event['sequence']] = event
        elif kind == 'cta_barrier_release':
            participants = []
            for reference in event['arrivals']:
                arrival = arrivals[reference]; parent = issues[arrival['issue']]
                participants.append({'arrival':reference, 'issue':parent['sequence'], 'warp':parent['warp'],
                    'function':parent['function'], 'frame':parent['frame'], 'pc':parent['pc'],
                    'ptx_line':parent['ptx_line'], 'execution_mask':arrival['execution_mask'],
                    'blocking':arrival['blocking'], 'true_votes':arrival['true_votes']})
            phase = {key:event[key] for key in ['launch','kernel','block','barrier','phase','count','reduction','result']}
            phase.update(release=event['sequence'], participants=participants,
                arrival_issues=[p['issue'] for p in participants],
                exit_issues=[exits[reference]['issue'] for reference in event['exits']],
                exits=[{'event':reference, 'warp':exits[reference]['warp'], 'issue':exits[reference]['issue']} for reference in event['exits']])
            phases.append(phase)
            for participant in participants:
                if participant['blocking']:
                    actor = (event['launch'], tuple(event['block']), participant['warp'])
                    pending[actor] = phase
    return phases, gates

def analyze_barriers(run_directory):
    from register_dependencies import CompiledInstructions
    run = Path(run_directory).resolve()
    metadata = validate_compilation_artifacts(run)
    events_path = run/'execution-events.jsonl'
    summary = summarize_events(events_path)
    if metadata['exit_code'] or metadata['timed_out'] or not summary['complete_supported_cta_barriers']:
        raise ValueError('CTA report requires successful complete typed phases, retirement and instruction mapping')
    data = [json.loads(line) for line in events_path.read_text().splitlines()]
    compiled = CompiledInstructions((run/'kernel.ptx').read_text())
    for event in data:
        if event['type'] == 'issue' and not compiled.matches(event['ptx_line'], event['instruction']):
            raise ValueError('CTA instruction differs from compiled PTX spelling or start line')
    phases, gates = observed_controls(data)
    if len(gates) != summary['cta_barriers'].get('released_waiters',0):
        raise ValueError('CTA release lacks a subsequent selected waiter instruction')
    return {'schema':1, 'kind':'observed CTA synchronization causes; no timing or memory-visibility prediction',
            'event_sha256':hashlib.sha256(events_path.read_bytes()).hexdigest(),
            'ptx_sha256':metadata['compilation_sha256']['kernel.ptx'],
            'counts':summary['cta_barriers'], 'phases':phases, 'control_gates':gates,
            'hardware_cycles':False, 'complete_scoreboard_dependencies':False,
            'limits':['Selected instruction order follows the functional run, not a regenerated GPU control flow',
                      'Causal arrivals/exits do not establish outstanding-memory completion or calibrated barrier latency',
                      'Virtual CTA reduction data dependencies are not yet integrated into the register graph']}
