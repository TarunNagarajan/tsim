"""Independent interval-sweep checks of emitted execution schedules.

No scheduler implementation is used: audit timestamps against the dependency
graph, queue definitions and physical admission budgets.
"""
from collections import defaultdict
from math import isclose

def verify_schedule(report, graph):
    nodes = {n['sequence']: n for n in graph['nodes']}
    records = {r['sequence']: r for launch in report['launches'] for r in launch['issues']}
    assert len(records) == sum(len(l['issues']) for l in report['launches']) and set(records) == set(nodes), 'Issue conservation'
    config, profile = report['config'], report['profile']
    count = config['sm_count']; pipes = config['pipelines']; seen = set(); streams = defaultdict(list)
    totals = defaultdict(list)
    for sequence, row in records.items():
        begin, end = row['issue_cycle'], row['complete_cycle']
        assert type(begin) is int and type(end) is int and 0 <= begin < end, 'Invalid issue interval'
        assert 0 <= row['sm'] < count and 0 <= row['partition'] < profile['partitions'], 'Placement bounds'
        slot = (row['sm'], row['partition'], begin)
        assert slot not in seen, 'Partition issue bandwidth'; seen.add(slot)
        pipe = pipes[row['pipeline']]
        assert end - begin == pipe['latency'], 'Configured latency'
        key = (row['sm'], row['pipeline'], row['partition'] if pipe['scope'] == 'partition' else -1)
        totals[key].append((begin, end))
        node = nodes[sequence]
        assert row['block'] == node['block'] and row['warp'] == node['warp'], 'Captured ownership'
        streams[(node['launch'], tuple(row['block']), row['warp'])].append(row)
        for edge in node['edges']:
            predecessor = records[edge['source']]
            required = predecessor['issue_cycle'] if edge['kind'] == 'WAR' else predecessor['complete_cycle']
            assert begin >= required, 'Hazard readiness: ' + edge['kind']
    for stream in streams.values():
        ordered = sorted(stream, key=lambda row: row['sequence'])
        assert all(b['issue_cycle'] > a['issue_cycle'] for a, b in zip(ordered, ordered[1:])), 'Warp issue order'
        assert all(a['pipeline'] != 'control' or b['issue_cycle'] >= a['complete_cycle'] for a, b in zip(ordered, ordered[1:])), 'Control path readiness'
    peaks = defaultdict(int)
    for key, intervals in totals.items():
        # Launches drain independently; serial origins permit one global sweep.
        pipe = pipes[key[1]]; begins = sorted(a for a, _ in intervals)
        assert all(b - a >= pipe['interval'] for a, b in zip(begins, begins[1:])), 'Initiation interval'
        active = 0
        for _, delta in sorted([(a, 1) for a, _ in intervals] + [(b, -1) for _, b in intervals]):
            active += delta
            assert 0 <= active <= pipe['capacity'], 'Pipeline capacity'
            peaks[key] = max(peaks[key], active)
        assert active == 0, 'Pipeline drain'
    previous_end = 0
    for launch in report['launches']:
        assert launch['start_cycle'] == previous_end, 'Sequential launch origin'
        assert launch['end_cycle'] == max(r['complete_cycle'] for r in launch['issues']), 'Launch retirement'
        previous_end = launch['end_cycle']
        footprint = launch['footprint']; areas = [0] * count; sweeps = [[] for _ in range(count)]
        mapping = {tuple(b['block']): b for b in launch['blocks']}
        assert len(mapping) == len(launch['blocks']), 'Unique block identity'
        for block, interval in mapping.items():
            own = [r for r in launch['issues'] if tuple(r['block']) == block]
            assert own and interval['release_cycle'] == max(r['complete_cycle'] for r in own), 'Block drains all tails'
            assert len(interval['warp_partitions']) == footprint['warps'], 'Warp placement completeness'
            for row in own:
                assert interval['sm'] == row['sm'] and interval['warp_partitions'][row['warp']] == row['partition'], 'Stable placement'
                assert interval['admit_cycle'] <= row['issue_cycle'] < row['complete_cycle'] <= interval['release_cycle'], 'Resident instruction'
            sm = interval['sm']; start, end = interval['admit_cycle'], interval['release_cycle']
            areas[sm] += footprint['warps'] * (end - start)
            charges = [interval['warp_partitions'].count(p) * footprint['registers_per_warp'] for p in range(profile['partitions'])]
            resource = [1, footprint['warps'], footprint['shared_bytes'], footprint['barriers']] + charges
            sweeps[sm].extend([(start, 1, resource), (end, -1, resource)])
        capacities = [profile['blocks'], profile['warps'], profile['shared_bytes'], profile['barriers'] or float('inf')] + [profile['registers'] // profile['partitions']] * profile['partitions']
        for sm, sweep in enumerate(sweeps):
            held = [0] * len(capacities); peak = [0] * 3
            for _, sign, charge in sorted(sweep):
                held = [a + sign * b for a, b in zip(held, charge)]
                assert all(0 <= a <= b for a, b in zip(held, capacities)), 'Admission capacity'
                peak = [max(a, b) for a, b in zip(peak, held[:3])]
            assert not any(held), 'Resident drain'
            assert peak == [launch['peak_resident_per_sm'][sm][r] for r in ['blocks', 'warps', 'shared_bytes']], 'Resident peaks'
        assert areas == launch['resident_warp_cycle_integral'], 'Resident area'
        fraction = sum(areas) / ((launch['end_cycle'] - launch['start_cycle']) * count * profile['warps'])
        assert isclose(fraction, launch['modeled_resident_warp_fraction']), 'Resident fraction'
        for final in launch['final_residency']:
            assert all(not any(v) if isinstance(v, list) else not v for v in final.values()), 'Final resources'
        local = defaultdict(list)
        for row in launch['issues']:
            pipe = pipes[row['pipeline']]; key = (row['sm'], row['pipeline'], row['partition'] if pipe['scope'] == 'partition' else -1)
            local[key].extend([(row['issue_cycle'], 1), (row['complete_cycle'], -1)])
        local_peaks = {}
        for key, events in local.items():
            active = maximum = 0
            for _, delta in sorted(events): active += delta; maximum = max(maximum, active)
            local_peaks[key] = maximum
        assert local_peaks == {(p['sm'], p['pipeline'], p['partition']): p['inflight_peak'] for p in launch['pipeline_peaks'] if p['inflight_peak']}, 'Pipeline peaks'
    assert previous_end == report['model_cycles'], 'Total duration'
    assert report['hardware_cycles'] is report['calibrated'] is report['complete_scoreboard_dependencies'] is False, 'Evidence boundaries'
    return True
