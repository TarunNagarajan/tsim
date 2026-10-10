"""Hand-solved SM schedules driven by unresolved memory completions."""
from dataclasses import replace
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from sm_execution import Issue, Pipeline, TimingConfig, SectorRequest, simulate_launch
from sm_residency import profile_for, block_footprint
from sector_memory_service import MemoryConfig

def verify_sm_memory():
    checks = []
    def check(name, condition):
        if not condition: raise AssertionError(name)
        checks.append(name)
    def reject(name, action):
        try: action()
        except ValueError: checks.append(name)
        else: raise AssertionError(name)
    profile = profile_for('nvidia/a100'); fp = block_footprint(profile, [32, 1, 1], 16)
    timing = TimingConfig(1, {'memory': Pipeline(999, 1, 8), 'alu': Pipeline(4, 1, 4), 'control': Pipeline(1, 1, 4)}, 'Independent asynchronous-memory reference')
    def memory(n, sector=0, operation='load', deps=(), block=(0, 0, 0), warp=0):
        return Issue(n, block, warp, 'memory', deps, memory=SectorRequest(operation, (sector,)))
    def ordinary(n, deps=(), control=False): return Issue(n, (0, 0, 0), 0, 'control' if control else 'alu', deps, control)
    def solve(stream, config=None, origin=0, timing=timing, fp=fp):
        return simulate_launch(profile, fp, stream, timing, origin, config or MemoryConfig(sm_count=timing.sm_count))
    def times(result): return {r['sequence']: (r['issue_cycle'], r['complete_cycle']) for r in result['issues']}
    result = solve([memory(1), ordinary(2, ((1, 'RAW'),)), ordinary(3, control=True)])
    check('unresolved load RAW waits actual return instead of configured token latency', times(result) == {1: (0, 21), 2: (21, 25), 3: (22, 23)})
    check('early return cannot release a block with an arithmetic tail', result['blocks'][0]['release_cycle'] == result['instruction_end_cycle'] == result['end_cycle'] == 25)
    check('cold load records actual service request and backing transaction', result['memory_service']['requests'][0]['complete_cycle'] == 21 and result['memory_service']['stats']['dram_reads'] == 1)
    result = solve([memory(1), ordinary(2, ((1, 'WAR'),)), ordinary(3, control=True)])
    check('WAR waits operand consumption and permits overwrite before load response', times(result) == {1: (0, 21), 2: (1, 5), 3: (2, 3)})
    check('early return and overwrite still hold block through memory response', result['blocks'][0]['release_cycle'] == result['end_cycle'] == 21)
    result = solve([memory(1), ordinary(2, ((1, 'WAW'),))])
    check('WAW on an unresolved load waits its actual completion', times(result) == {1: (0, 21), 2: (21, 25)})
    result = solve([memory(1), memory(2, 4), ordinary(3, control=True)])
    check('independent loads overlap through asynchronous memory service', times(result) == {1: (0, 21), 2: (1, 22), 3: (2, 3)})
    result = solve([memory(1), memory(2, 4), ordinary(3, control=True)], MemoryConfig(lsu_slots_per_sm=1))
    check('finite LSU pressure holds an otherwise-ready load until capacity returns', times(result) == {1: (0, 21), 2: (21, 42), 3: (22, 23)})
    check('memory admission pressure is explicitly recorded', result['counts']['memory_admission_blocked_probes'] > 0 and result['memory_service']['peaks']['lsu_slots'] == 1)
    result = solve([memory(1, operation='store'), ordinary(2, control=True)])
    check('store acknowledgement follows L2 commit and can outlive return instruction', times(result) == {1: (0, 18), 2: (1, 2)} and result['blocks'][0]['release_cycle'] == 18)
    check('terminal writeback tail extends launch boundary after block retirement', result['instruction_end_cycle'] == 18 and result['end_cycle'] == 29 and result['memory_service']['writebacks'][0]['complete_cycle'] == 29)
    check('resident area excludes a backing drain after all blocks release', result['resident_warp_cycle_integral'] == [18] and result['modeled_resident_warp_fraction'] == 18 / (29 * 64))
    two = block_footprint(profile, [64, 1, 1], 16)
    result = solve([memory(1), memory(2, warp=1)], fp=two)
    check('shared LSU port arbitrates different scheduler partitions', times(result) == {1: (0, 21), 2: (1, 21)})
    check('same-SM pending sector requests merge one backing read', result['memory_service']['stats']['dram_reads'] == 1 and result['memory_service']['stats']['merged_miss_sectors'] == 1)
    result = solve([memory(1), memory(2, block=(1, 0, 0))], timing=replace(timing, sm_count=2))
    check('two SM admission ports issue in parallel and merge at shared L2', times(result) == {1: (0, 21), 2: (0, 21)} and result['memory_service']['stats']['dram_reads'] == 1)
    check('cross-SM resident areas independently equal two twenty-one-cycle lifetimes', result['resident_warp_cycle_integral'] == [21, 21])
    result = solve([memory(1), memory(2, deps=((1, 'RAW'),))])
    check('dependent repeated load sees completed L1 fill and takes only lookup latency', times(result) == {1: (0, 21), 2: (21, 24)} and result['memory_service']['stats']['l1_hit_sectors'] == 1)
    result = solve([memory(1), ordinary(2, ((1, 'RAW'),))], origin=13)
    check('nonzero launch origin shifts every asynchronous timestamp', times(result) == {1: (13, 34), 2: (34, 38)} and result['model_cycles'] == 25)
    def continuation(first, config, timing=timing):
        return simulate_launch(profile, fp, [memory(2)], timing, first['end_cycle'], config,
                               first['next_pipeline_readiness'], first['memory_service']['next_port_readiness'])
    slow_front = replace(timing, pipelines=dict(timing.pipelines, memory=Pipeline(999, 100, 8)))
    first = solve([memory(1)], timing=slow_front); second = continuation(first, MemoryConfig(), slow_front)
    check('pipeline initiation deadline persists across sequential launches', times(second) == {2: (100, 121)} and first['end_cycle'] == 21)
    config = MemoryConfig(lsu_interval=100); first = solve([memory(1)], config); second = continuation(first, config)
    check('LSU port readiness persists while caches start cold on the next launch', times(second) == {2: (100, 121)})
    config = MemoryConfig(l2_interval=100); first = solve([memory(1)], config); second = continuation(first, config)
    check('shared L2 initiation deadline persists across launch boundaries', second['memory_service']['lookup_records'][0]['start_cycle'] == 103 and second['end_cycle'] == 121)
    config = MemoryConfig(dram_interval=100); first = solve([memory(1)], config); second = continuation(first, config)
    check('backing initiation deadline persists without delaying independent front-end work', times(second) == {2: (21, 121)} and second['memory_service']['dram_reads'][0]['start_cycle'] == 108)
    reject('sector requests require an explicit memory controller', lambda: simulate_launch(profile, fp, [memory(1)], timing))
    reject('timed memory cannot silently fall back to fixed tokens without addresses', lambda: solve([Issue(1, (0, 0, 0), 0, 'memory')]))
    reject('mismatched SM counts refused', lambda: solve([memory(1)], MemoryConfig(sm_count=2)))
    reject('memory request cannot be attached to a control instruction', lambda: solve([replace(memory(1), control=True)]))
    reject('unknown sector operation refused before issue', lambda: solve([memory(1, operation='atomic')]))
    reject('boolean sector identity refused before issue', lambda: solve([memory(1, sector=True)]))
    reject('duplicate sector identities refused before issue', lambda: solve([replace(memory(1), memory=SectorRequest('load', (0, 0)))]))
    reject('invalid restored pipeline deadline refused', lambda: simulate_launch(profile, fp, [ordinary(1)], timing, pipeline_readiness=[{'sm': 0, 'pipeline': 'alu', 'partition': 0, 'next_issue_cycle': True}]))
    return checks
