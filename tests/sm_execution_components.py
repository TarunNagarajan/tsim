"""Hand-solved issue, latency, queue and resource-lifetime references."""
from dataclasses import replace
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from sm_execution import Issue, Pipeline, TimingConfig, default_config, simulate_launch
from sm_residency import block_footprint, profile_for, SMResidency

def verify_execution_components():
    checks = []
    def check(name, condition):
        if not condition: raise AssertionError(name)
        checks.append(name)
    def reject(name, action):
        try: action()
        except ValueError: checks.append(name)
        else: raise AssertionError(name)
    profile = profile_for('nvidia/a100')
    footprint = block_footprint(profile, [32, 1, 1], 16)
    def config(latency=4, interval=1, capacity=4, scope='partition', sms=1):
        return TimingConfig(sms, {'alu': Pipeline(latency, interval, capacity, scope)}, 'Hand-solved synthetic reference')
    def issue(sequence, dependencies=(), block=(0, 0, 0), warp=0, pipeline='alu', control=False):
        return Issue(sequence, block, warp, pipeline, dependencies, control)
    def solve(issues, conf=None, fp=footprint, origin=0): return simulate_launch(profile, fp, issues, conf or config(), origin)
    def times(result): return {record['sequence']: (record['issue_cycle'], record['complete_cycle']) for record in result['issues']}
    result = solve([issue(1)])
    check('single instruction latency includes retirement tail', result['model_cycles'] == 4 and times(result) == {1: (0, 4)})
    check('single block holds residency until final retirement', result['blocks'][0]['admit_cycle'] == 0 and result['blocks'][0]['release_cycle'] == 4)
    check('single warp integral is independently four warp cycles', result['resident_warp_cycle_integral'] == [4] and result['modeled_resident_warp_fraction'] == 1 / 64)
    independent = solve([issue(1), issue(2), issue(3)])
    check('independent instructions pipeline before earlier completions', times(independent) == {1: (0, 4), 2: (1, 5), 3: (2, 6)})
    check('independent issue stream drains after final completion', independent['model_cycles'] == 6)
    raw = solve([issue(1), issue(2, ((1, 'RAW'),)), issue(3, ((2, 'RAW'),))])
    check('RAW chain waits for each result', times(raw) == {1: (0, 4), 2: (4, 8), 3: (8, 12)})
    waw = solve([issue(1), issue(2, ((1, 'WAW'),))])
    check('WAW pending destination blocks second writer', times(waw)[2] == (4, 8))
    ports = TimingConfig(1, {'slow': Pipeline(10, 1, 4), 'fast': Pipeline(1, 1, 4)}, 'Operand reads at issue')
    war = solve([issue(1, pipeline='slow'), issue(2, ((1, 'WAR'),), pipeline='fast')], ports)
    check('WAR waits operand read instead of full result latency', times(war) == {1: (0, 10), 2: (1, 2)})
    check('out-of-order retirement still holds block until slow tail', war['blocks'][0]['release_cycle'] == 10)
    limited = solve([issue(1), issue(2), issue(3)], config(capacity=1))
    check('one-entry in-flight queue applies backpressure', times(limited) == {1: (0, 4), 2: (4, 8), 3: (8, 12)})
    check('bounded queue peak is exact', limited['pipeline_peaks'][0]['inflight_peak'] == 1)
    spacing = solve([issue(1), issue(2), issue(3)], config(interval=3, capacity=16))
    check('initiation interval is independent of latency', times(spacing) == {1: (0, 4), 2: (3, 7), 3: (6, 10)})
    two_blocks = [issue(1), issue(2, block=(1, 0, 0))]
    separate = solve(two_blocks)
    check('different partition pipelines overlap', times(separate) == {1: (0, 4), 2: (0, 4)})
    shared = solve(two_blocks, config(scope='sm'))
    check('SM-scoped pipeline arbitrates across partitions', times(shared) == {1: (0, 4), 2: (1, 5)})
    two_sms = solve(two_blocks, config(scope='sm', sms=2))
    check('SM-scoped pipeline capacity is not shared across SMs', times(two_sms) == {1: (0, 4), 2: (0, 4)} and {b['sm'] for b in two_sms['blocks']} == {0, 1})
    four_warps = block_footprint(profile, [128, 1, 1], 16)
    streams = [issue(warp * 3 + ordinal + 1, warp=warp) for warp in range(4) for ordinal in range(3)]
    fair = solve(streams, config(scope='sm', capacity=16), four_warps)
    check('shared port rotating arbitration gives every partition progress',
          times(fair) == {warp * 3 + ordinal + 1: (warp + ordinal * 4, warp + ordinal * 4 + 4)
                         for warp in range(4) for ordinal in range(3)})
    check('shared pipeline peak counts overlapping latency tails', fair['pipeline_peaks'][0]['inflight_peak'] == 4)
    serial_control = solve([issue(1, control=True), issue(2)], config(latency=5))
    check('recorded next control path awaits branch resolution', times(serial_control) == {1: (0, 5), 2: (5, 10)})
    shifted = solve([issue(1)], origin=13)
    check('launch origin shifts timestamps without changing duration', times(shifted) == {1: (13, 17)} and shifted['model_cycles'] == 4)
    zero = block_footprint(profile, [32, 1, 1], 0)
    resident = SMResidency(profile); resident.admit('a', zero); resident.admit('b', zero)
    check('zero-register warps keep explicit rotating placement', resident.warp_partitions('a') == (0,) and resident.warp_partitions('b') == (1,))
    resident.release('a'); reject('released block has no stale placement', lambda: resident.warp_partitions('a'))
    full = block_footprint(profile, [1024, 1, 1], 16)
    workload = [issue(block * 32 + warp + 1, block=(block, 0, 0), warp=warp) for block in range(3) for warp in range(32)]
    waves = solve(workload, fp=full)
    check('two full blocks fill sixty-four resident warp slots', waves['peak_resident_per_sm'][0]['warps'] == 64 and waves['peak_resident_per_sm'][0]['blocks'] == 2)
    check('waiting third block enters only after first tail drains', waves['blocks'][2]['admit_cycle'] == 11)
    check('three full-block retirement waves end at cycle twenty-seven', waves['end_cycle'] == 27)
    check('residency integral follows independent block lifetime sum', waves['resident_warp_cycle_integral'] == [1472] and waves['modeled_resident_warp_fraction'] == 23 / 27)
    check('all block and partition resources return to zero', waves['final_residency'] == [dict(blocks=0, warps=0, threads=0, shared_bytes=0, barriers=0, partition_registers=[0, 0, 0, 0])])
    check('full blocks preserve every allocator warp assignment', all(b['warp_partitions'] == [0, 1, 2, 3] * 8 for b in waves['blocks']))
    for name, conf in [('zero latency', config(latency=0)), ('boolean latency', config(latency=True)),
                       ('zero interval', config(interval=0)), ('fractional interval', config(interval=1.5)),
                       ('negative capacity', config(capacity=-1)), ('boolean capacity', config(capacity=True)),
                       ('unknown scope', config(scope='chip')), ('zero SMs', config(sms=0)),
                       ('boolean SMs', config(sms=True)), ('unsupported calibration claim', replace(config(), provenance_kind='measured')),
                       ('empty provenance', replace(config(), provenance=''))]:
        reject(name + ' rejected', lambda conf=conf: solve([issue(1)], conf))
    for name, issues in [('duplicate sequence', [issue(1), issue(1)]),
                         ('boolean sequence', [issue(True)]), ('invalid coordinate', [issue(1, block=(-1, 0, 0))]),
                         ('missing predecessor', [issue(2, ((1, 'RAW'),))]), ('self predecessor', [issue(1, ((1, 'RAW'),))]),
                         ('unknown hazard', [issue(1), issue(2, ((1, 'control'),))]),
                         ('unknown pipeline', [issue(1, pipeline='missing')]), ('warp outside footprint', [issue(1, warp=1)]),
                         ('cross-warp register hazard', [issue(1), issue(2, ((1, 'RAW'),), block=(1, 0, 0))])]:
        reject(name + ' rejected', lambda issues=issues: solve(issues))
    reject('empty launch rejected', lambda: solve([]))
    reject('missing warp stream rejected', lambda: solve([issue(1)], fp=four_warps))
    reject('overfull resource footprint rejected before execution', lambda: solve([issue(1)], fp=block_footprint(profile, [1024, 1, 1], 72)))
    reject('negative cycle origin rejected', lambda: solve([issue(1)], origin=-1))
    check('default parameters carry only explicit uncalibrated assumptions', default_config().record()['provenance_kind'] == 'assumption')
    return checks

if __name__ == '__main__': print('PASS SM execution components checks=' + str(len(verify_execution_components())))
