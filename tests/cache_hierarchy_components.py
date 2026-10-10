"""Analytically solvable two-level traffic and full-opcode policy references."""
from copy import deepcopy
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from cache_hierarchy import CacheHierarchy, DEFAULT_PROFILE, decode_policy, validate_profile

def verify_cache_components():
    checks = []
    def check(name, condition):
        if not condition: raise AssertionError(name)
        checks.append(name)
    def reject(name, action):
        try: action()
        except ValueError: checks.append(name)
        else: raise AssertionError(name)
    def policy(opcode, operation='load'):
        return decode_policy(opcode + ' %r0, [%rd0]', operation)
    ca, cg = policy('ld.global.ca.u32'), policy('ld.global.cg.u32')
    wb = policy('st.global.u32', 'store')
    atom = policy('atom.global.add.u32', 'atomic')
    for name, instruction, operation, expected in [
        ('default load caches all levels', 'ld.global.u32', 'load', 'ca'),
        ('default store writes coherent levels', 'st.global.u32', 'store', 'wb'),
        ('explicit global bypass', 'ld.global.cg.u32', 'load', 'cg'),
        ('streaming retained', 'ld.global.cs.u32', 'load', 'cs'),
        ('last-use retained', 'ld.local.lu.u32', 'load', 'lu'),
        ('system fetch retained', 'ld.global.cv.u32', 'load', 'cv'),
        ('system write-through retained', 'st.global.wt.u32', 'store', 'wt'),
        ('predicated policy decoded', '@!%p3 ld.global.cg.u32', 'load', 'cg'),
        ('reduction routes atomic work', 'red.global.add.u32', 'atomic', 'atomic-l2')]:
        check(name, policy(instruction, operation).cache_operator == expected)
    hinted = policy('ld.global.L1::no_allocate.L2::128B.u32')
    check('parser-discarded L1 and L2 hints retained', hinted.hints == ('L1::no_allocate', 'L2::128B'))
    check('noncoherent qualifier retained', policy('ld.global.nc.u32').noncoherent)
    strong = policy('ld.acquire.gpu.global.u32')
    check('ordering and scope retained', strong.ordering == 'acquire' and strong.scope == 'gpu')
    check('reconstructed store without opcode whitespace decoded', decode_policy('st.global.cg.u32[%rd0], %r0', 'store').cache_operator == 'cg')
    reject('wrong captured operation rejected', lambda: policy('st.global.u32'))
    reject('duplicate cache policies rejected', lambda: policy('ld.global.ca.cg.u32'))
    reject('illegal load write-back rejected', lambda: policy('ld.global.wb.u32'))
    reject('unknown modifier rejected', lambda: policy('ld.global.future_hint.u32'))
    reject('missing instruction rejected', lambda: decode_policy('ld.global.ca.u32', 'load'))

    config = deepcopy(DEFAULT_PROFILE)
    config['sm_count'] = 2
    config['l1'].update(sets=1, ways=1)
    config['l2'].update(sets=1, ways=2)
    model = CacheHierarchy(config)
    first = model.request({0, 1, 2, 3}, 'load', ca)
    check('cold full warp fills four sectors', first == {'l1_hits': 0, 'l1_misses': 4, 'l1_bypass': 0, 'l2_hits': 0, 'l2_misses': 4})
    second = model.request({0, 1, 2, 3}, 'load', ca)
    check('same SM reuse stays in L1', second['l1_hits'] == 4 and second['l2_hits'] == second['l2_misses'] == 0)
    third = model.request({0, 1, 2, 3}, 'load', ca, sm=1)
    check('other SM misses private L1 and hits shared L2', third['l1_misses'] == 4 and third['l2_hits'] == 4)
    fourth = model.request({0, 1, 2, 3}, 'load', cg)
    check('cg bypasses warm L1 and hits L2', fourth['l1_bypass'] == 4 and fourth['l2_hits'] == 4)
    summary = model.finish()
    check('hierarchy request and backing conservation', summary['traffic']['logical_request_sectors'] == 16 and
          summary['traffic']['backing_read_sectors'] == 4 and summary['traffic']['backing_write_sectors'] == 0)
    reject('post-drain issue rejected', lambda: model.request({0}, 'load', ca))
    reject('duplicate drain rejected', model.finish)
    model = CacheHierarchy(config)
    model.request({0}, 'load', ca)
    absent = model.request({1}, 'load', ca)
    check('tag hits still miss absent L1 and L2 sectors', absent['l1_misses'] == absent['l2_misses'] == 1)
    model.request({4}, 'load', ca)
    retained = model.request({0}, 'load', ca)
    check('L1 conflict can hit retained L2 line', retained['l1_misses'] == 1 and retained['l2_hits'] == 1)
    model.request({8}, 'load', ca)
    lost = model.request({4}, 'load', ca)
    check('L2 capacity eviction reaches backing memory', lost['l2_misses'] == 1)

    model = CacheHierarchy(config)
    stored = model.request({0, 1}, 'store', wb)
    check('global store bypasses L1 and allocates dirty L2', stored['l1_bypass'] == 2 and stored['l2_misses'] == 2)
    model.request({4}, 'load', cg); model.request({8}, 'load', cg)
    check('L2 dirty eviction conserves backing writes', model.traffic['backing_write_sectors'] == 2)
    model.request({12}, 'store', wb)
    summary = model.finish()
    check('terminal dirty drain counted separately', summary['traffic']['backing_terminal_write_sectors'] == 1)
    check('terminal drain clears L2 dirtiness', all(not e.dirty for b in model.l2.sets for e in b.values()))

    model = CacheHierarchy(config)
    model.request({0}, 'load', ca)
    rmw = model.request({0}, 'atomic', atom)
    check('atomic RMW bypasses warm global L1', rmw['l1_bypass'] == 1 and rmw['l2_hits'] == 1)
    check('atomic RMW dirties one sector', model.finish()['traffic']['backing_terminal_write_sectors'] == 1)
    model = CacheHierarchy(config)
    model.request({0}, 'store', wb, namespace='local:warp0')
    check('private local stores are L1 write-back', model.l1[0].stats['sector_fills'] == 1 and model.l2.stats['write_access_sectors'] == 0)
    model.request({4}, 'load', ca, namespace='local:warp0')
    check('dirty local L1 eviction routes to L2', model.traffic['l1_eviction_writeback_sectors'] == 1)
    private = model.request({0}, 'load', ca, namespace='local:warp1')
    check('different private owner cannot alias L2', private['l2_misses'] == 1)
    model.request({0}, 'store', wb, namespace='local:warp1')
    summary = model.finish()
    check('local terminal drain propagates through both levels', summary['traffic']['l1_terminal_writeback_sectors'] == 1 and
          summary['traffic']['backing_terminal_write_sectors'] + summary['traffic']['backing_write_sectors'] == 2)
    model = CacheHierarchy(config)
    for name, p in [('eviction/prefetch hints', hinted), ('noncoherent cache', policy('ld.global.nc.u32')),
                    ('strong ordering', strong), ('streaming', policy('ld.global.cs.u32')),
                    ('system address cache', policy('ld.global.cv.u32'))]:
        reject(name + ' refused before modeling', lambda p=p: model.request({0}, 'load', p))
    check('unsupported policy leaves caches untouched', not model.traffic and not model.l2.stats and all(not c.stats for c in model.l1))
    reject('invalid namespace rejected', lambda: model.request({0}, 'load', ca, namespace='shared'))
    reject('invalid SM rejected', lambda: model.request({0}, 'load', ca, sm=2))
    reject('negative sector rejected before state change', lambda: model.request({-1}, 'load', ca))
    reject('private local atomic rejected', lambda: model.request({0}, 'atomic', atom, namespace='local:warp0'))
    for name, key, value in [('unknown profile field', 'fake', 1), ('unimplemented ordering', 'ordering', 'GPU'),
                              ('boolean SM count', 'sm_count', True), ('zero SM count', 'sm_count', 0),
                              ('false calibration provenance', 'provenance', 'measured on A100')]:
        bad = deepcopy(config); bad[key] = value
        reject(name + ' rejected', lambda bad=bad: validate_profile(bad))
    bad = deepcopy(config); bad['l2']['sector_bytes'] = 64
    reject('unsupported hierarchy geometry rejected', lambda: validate_profile(bad))
    return checks

if __name__ == '__main__':
    print('PASS cache hierarchy components checks=' + str(len(verify_cache_components())))
