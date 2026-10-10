"""Independent analytical address patterns and cache-state expectations."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from memory_model import (SectorCache, sectors_for_spans, shared_scalar_request,
                          constant_scalar_request, local_spans, LOCAL_BASE, SHARED_BASE)

def verify_components():
    checks = []
    def check(name, condition):
        if not condition: raise AssertionError(name)
        checks.append(name)
    def reject(name, operation):
        try: operation()
        except ValueError: checks.append(name)
        else: raise AssertionError(name)
    def accesses(addresses, width=4):
        return [{'address': hex(a), 'bytes': width, 'lane': i, 'element': 0} for i, a in enumerate(addresses)]
    def shared(stride=1, broadcast=False):
        return accesses([SHARED_BASE + (0 if broadcast else i * stride * 4) for i in range(32)])

    patterns = [('aligned full warp', [4 * i for i in range(32)], 4),
                ('offset full warp', [4 + 4 * i for i in range(32)], 5),
                ('stride two', [8 * i for i in range(32)], 8),
                ('stride eight', [32 * i for i in range(32)], 32),
                ('broadcast word', [0] * 32, 1),
                ('partial eight lanes', [4 * i for i in range(8)], 1),
                ('permuted warp', [4 * ((i * 13) % 32) for i in range(32)], 4)]
    for name, addresses, expected in patterns:
        check(name, len(sectors_for_spans([(a, 4) for a in addresses])) == expected)
    check('byte span crosses sector', sectors_for_spans([(31, 2)]) == {0, 1})
    check('vector span merges components', len(sectors_for_spans([(16 * i + 4 * e, 4) for i in range(32) for e in range(4)])) == 16)
    check('64-bit boundary preserved', sectors_for_spans([((1 << 64) - 8, 8)]) == {((1 << 64) - 8) // 32})
    reject('address wrap rejected', lambda: sectors_for_spans([((1 << 64) - 4, 8)]))
    reject('negative address rejected', lambda: sectors_for_spans([(-1, 4)]))
    reject('zero sector rejected', lambda: sectors_for_spans([(0, 4)], 0))
    reject('zero access rejected', lambda: sectors_for_spans([(0, 0)]))
    check('shared contiguous banks', shared_scalar_request(shared(), 'load')['bank_conflict_degree'] == 1)
    check('shared two-way conflicts', shared_scalar_request(shared(2), 'load')['bank_conflict_degree'] == 2)
    check('shared thirty-two-way conflicts', shared_scalar_request(shared(32), 'load')['bank_conflict_degree'] == 32)
    topology = shared_scalar_request(shared(broadcast=True), 'load')
    check('shared load broadcast', topology['bank_conflict_degree'] == 1 and topology['same_word_extra_lanes'] == 31)
    topology = shared_scalar_request(shared(broadcast=True), 'store')
    check('shared write alias distinct from broadcast semantics', topology['write_alias_words'] == 1)
    topology = shared_scalar_request(shared(broadcast=True), 'atomic')
    check('atomic work counts without cycle prediction', topology['atomic_updates'] == 32 and
          topology['max_updates_to_one_word'] == 32 and 'bank_conflict_degree' not in topology)
    reject('wide shared topology rejected', lambda: shared_scalar_request(accesses([SHARED_BASE], 8), 'load'))
    reject('unknown shared operation rejected', lambda: shared_scalar_request(shared(), 'unknown'))
    reject('distributed shared owners rejected', lambda: shared_scalar_request(accesses([SHARED_BASE, SHARED_BASE + (1 << 24)]), 'load'))
    local = accesses([LOCAL_BASE] * 32)
    check('private locals are four coalesced sectors, not broadcast', len(sectors_for_spans(local_spans(local))) == 4)
    check('64-bit locals touch two word rows', len(sectors_for_spans(local_spans(accesses([LOCAL_BASE] * 32, 8)))) == 8)
    check('private local divergent offsets stay separate', len(sectors_for_spans(local_spans(accesses([LOCAL_BASE + i * 4 for i in range(32)])))) == 32)
    reject('unknown local mapping rejected', lambda: local_spans(accesses([0])))
    reject('non-NVIDIA warp layout rejected', lambda: local_spans(local, 64))
    check('constant broadcast has one address request', constant_scalar_request(accesses([0] * 32)) == 1)
    check('constant divergent words have thirty-two requests', constant_scalar_request(accesses([4 * i for i in range(32)])) == 32)
    reject('wide constant topology rejected', lambda: constant_scalar_request(accesses([0], 8)))

    cache = SectorCache(1, 2)
    check('cache cold sector miss', cache.request({0})['misses'] == {0})
    check('cache same-tag absent-sector miss', cache.request({1})['misses'] == {1})
    check('cache warm sector hit', cache.request({0, 1})['hits'] == {0, 1})
    check('cache sector/tag conservation', cache.stats['tag_misses'] == 1 and cache.stats['tag_hits'] == 2 and
          cache.stats['sector_misses'] == 2 and cache.stats['sector_hits'] == 2 and cache.stats['sector_fills'] == 2)
    cache = SectorCache(1, 2)
    cache.request({0}); cache.request({4}); cache.request({0}); cache.request({8})
    check('LRU retains recently used line', ('global', 0) in cache.sets[0] and ('global', 1) not in cache.sets[0])
    check('LRU evicts one line', cache.stats['evictions'] == 1)
    cache = SectorCache(1, 1)
    cache.request({0, 1}, write=True)
    result = cache.request({4})
    check('dirty sector eviction identity', result['evicted_dirty'] == [('global', 0), ('global', 1)] and cache.stats['dirty_sectors_evicted'] == 2)
    cache = SectorCache(2, 1)
    cache.request({0}); cache.request({4})
    check('distinct sets avoid conflict', cache.stats['evictions'] == 0)
    cache.request({8})
    check('same set causes conflict eviction', cache.stats['evictions'] == 1)
    cache = SectorCache(1, 1)
    check('no allocate preserves cache', cache.request({0}, allocate=False)['misses'] == {0} and not cache.sets[0])
    cache.request({0}); cache.request({1}, allocate=False)
    check('no allocate leaves absent sector invalid', cache.sets[0][('global', 0)].valid == {0})
    cache = SectorCache(1, 2)
    cache.request({0}, namespace='local:warp0'); cache.request({0}, namespace='local:warp1')
    check('private owners cannot cache-alias', cache.stats['sector_misses'] == 2 and cache.stats['sector_hits'] == 0)
    cache = SectorCache(1, 1)
    cache.request({0, 1, 2, 3})
    check('one tag lookup for vector sectors', cache.stats['tag_lookups'] == 1 and cache.stats['sector_fills'] == 4)
    check('empty request adds no traffic', cache.request(set())['hits'] == set() and cache.stats['requests'] == 1)
    reject('zero sets rejected', lambda: SectorCache(0, 1))
    reject('fractional ways rejected', lambda: SectorCache(1, 1.5))
    reject('boolean geometry rejected', lambda: SectorCache(True, 1))
    reject('partial sector line geometry rejected', lambda: SectorCache(1, 1, line_bytes=100))
    reject('negative cache sector rejected', lambda: cache.request({-1}))
    return checks

if __name__ == '__main__':
    checks = verify_components()
    print('PASS memory model components checks=' + str(len(checks)))
