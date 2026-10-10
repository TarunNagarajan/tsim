"""Validate interpreter event identities and summarize observed logical traffic."""
from collections import Counter
import json
from pathlib import Path
import re

def summarize_events(path):
    issues, launches, ends, launch_ends = {}, {}, {}, {}
    traffic = Counter()
    rows = 0
    truncated = False
    gaps, faults = 0, 0
    stream = None
    stream_end = False
    previous = 0
    observed = Counter()
    accesses = {}
    with Path(path).open() as source:
        for line in source:
            event = json.loads(line)
            rows += 1
            kind = event['type']
            if stream_end:
                raise ValueError('Event after stream completion')
            if kind == 'truncated':
                if truncated or event['limit'] != previous:
                    raise ValueError('Invalid execution event cap marker')
                truncated = True
                continue
            if truncated or event['sequence'] != previous + 1:
                raise ValueError('Noncontiguous execution event sequence')
            previous = event['sequence']
            if kind == 'stream':
                if stream is not None or previous != 1 or event['schema'] != 1:
                    raise ValueError('Unsupported or duplicate execution event header')
                stream = event
            elif kind == 'stream_end':
                if stream is None:
                    raise ValueError('Stream completion without header')
                stream_end = True
            elif kind == 'launch':
                if event['launch'] in launches:
                    raise ValueError('Duplicate launch identity')
                launches[event['launch']] = event
            elif kind == 'launch_end':
                key = event['launch']
                if key not in launches or key in launch_ends:
                    raise ValueError('Invalid launch completion')
                launch_ends[key] = event['status']
                faults += event['status'] != 'completed'
            elif kind == 'issue':
                if event['launch'] not in launches or event['launch'] in launch_ends:
                    raise ValueError('Issue outside live launch')
                if event['execution_mask'] & ~event['path_mask']:
                    raise ValueError('Execution lanes outside path')
                expected = event['expected_memory_accesses']
                lanes = event['execution_mask'].bit_count()
                if expected < 0 or (lanes and expected % lanes) or (not lanes and expected):
                    raise ValueError('Invalid expected memory access count')
                issues[previous] = event
                accesses[previous] = set()
            elif kind in ['memory', 'issue_end', 'coverage_gap']:
                key = event['issue']
                if key not in issues or key in ends:
                    raise ValueError('Event outside live parent instruction')
                parent = issues[key]
                if kind == 'issue_end':
                    if event['status'] not in ['completed', 'fault']:
                        raise ValueError('Unknown instruction completion status')
                    ends[key] = event['status']
                    expected = parent['expected_memory_accesses']
                    if observed[key] > expected or (event['status'] == 'completed' and observed[key] != expected):
                        raise ValueError('Memory access conservation failed')
                    faults += event['status'] == 'fault'
                elif kind == 'coverage_gap':
                    gaps += 1
                else:
                    if not isinstance(event['lane'], int) or not 0 <= event['lane'] < 64:
                        raise ValueError('Invalid memory lane')
                    if not parent['execution_mask'] & (1 << event['lane']):
                        raise ValueError('Memory event from inactive lane')
                    elements = parent['expected_memory_accesses'] // parent['execution_mask'].bit_count()
                    if not isinstance(event['element'], int) or not 0 <= event['element'] < elements:
                        raise ValueError('Invalid memory vector element')
                    if (event['status'] != 'completed' or event['bytes'] != parent['expected_memory_width'] or
                            event['bytes'] not in [1, 2, 4, 8] or
                            not re.fullmatch(r'0x[0-9a-f]+', event['address']) or
                            int(event['address'], 16) >= 1 << 64):
                        raise ValueError('Invalid actual memory event')
                    identity = (event['lane'], event['element'])
                    if identity in accesses[key]:
                        raise ValueError('Duplicate lane/element memory access')
                    accesses[key].add(identity)
                    observed[key] += 1
                    if event['space'] not in ['global', 'shared', 'local', 'constant', 'param']:
                        raise ValueError('Unknown resolved memory space')
                    if event['operation'] not in ['load', 'store', 'atomic']:
                        raise ValueError('Unknown memory operation')
                    prefix = event['space'] + '_' + event['operation']
                    traffic[prefix + '_accesses'] += 1
                    traffic[prefix + '_bytes'] += event['bytes']
            else:
                raise ValueError('Unknown execution event type: ' + kind)
    complete = bool(stream and stream_end and launches and len(launch_ends) == len(launches) and
                    len(ends) == len(issues) and not truncated and not faults and not gaps)
    return {'schema': 1, 'kind': 'observed logical PTX traffic; not hardware transactions or cycles',
            'records': rows, 'launches': len(launches), 'issues': len(issues),
            'memory': dict(sorted(traffic.items())), 'truncated': truncated,
            'coverage_gaps': gaps, 'fault_boundaries': faults,
            'unfinished_issues': len(issues) - len(ends),
            'unfinished_launches': len(launches) - len(launch_ends),
            'stream_finished': stream_end,
            'complete_supported_memory': complete,
            'complete_scoreboard_dependencies': False}
