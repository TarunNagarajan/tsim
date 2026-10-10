"""Hand-solved staged read-service references, without an SM timing claim."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from sector_read_service import ReadConfig, SectorReadService

def verify_sector_reads():
    checks = []
    def check(name, condition):
        if not condition: raise AssertionError(name)
        checks.append(name)
    def reject(name, action):
        try: action()
        except ValueError: checks.append(name)
        else: raise AssertionError(name)
    def cold(config=None, sectors=(0,), bypass=False):
        service = SectorReadService(config); assert service.submit(1, 0, sectors, bypass_l1=bypass)
        return service
    model = cold(); model.advance(8)
    check('cold request reaches backing service after three plus five cycles', model.dram_records == [{'sector': 0, 'start_cycle': 8, 'complete_cycle': 19}])
    check('L1 and L2 remain invalid while backing data is pending', not model.l1[0].sets[0] and not model.l2.sets[0])
    check('pending miss consumes one MSHR and one LSU slot', model.snapshot()['mshrs'] == 1 and model.snapshot()['lsu_slots'] == [1])
    check('backing completion makes L2 valid before L1 return', not model.advance(19) and model.l2.sets[0][('global', 0)].valid == {0} and not model.l1[0].sets[0])
    check('cold data returns at independently summed cycle twenty-one', model.advance(21)[0]['complete_cycle'] == 21)
    check('L1 becomes valid at return and all pending resources drain', model.l1[0].sets[0][('global', 0)].valid == {0} and model.snapshot()['mshrs'] == 0 and model.snapshot()['lsu_slots'] == [0])
    assert model.submit(2, 0, {0}); model.drain()
    check('warm L1 request completes after only its three-cycle lookup', model.finished[2]['complete_cycle'] == 24 and model.stats['dram_reads'] == 1 and model.stats['l1_hit_sectors'] == 1)
    model = cold(bypass=True); model.drain()
    check('L1 bypass cold path omits L1 lookup latency', model.finished[1]['complete_cycle'] == 18 and not model.l1[0].sets[0])
    assert model.submit(2, 0, {0}, bypass_l1=True); model.drain()
    check('warm L2 bypass takes five plus two cycles without another backing read', model.finished[2]['complete_cycle'] == 25 and model.stats['dram_reads'] == 1 and model.stats['l2_hit_sectors'] == 1)
    model = cold(); model.advance(1); assert model.submit(2, 0, {0}); model.drain()
    check('two pending same-sector misses merge into one backing transaction', model.stats['dram_reads'] == 1 and model.stats['merged_miss_sectors'] == 1)
    check('merged requests both return actual pending data at twenty-one', [model.finished[n]['complete_cycle'] for n in [1, 2]] == [21, 21])
    model = cold(ReadConfig(sm_count=2)); assert model.submit(2, 1, {0}); model.drain()
    check('private L1 misses across SMs share one pending L2 miss', model.stats['dram_reads'] == 1 and [model.finished[n]['complete_cycle'] for n in [1, 2]] == [21, 21])
    check('shared completion fills each requesting private L1', all(cache.sets[0][('global', 0)].valid == {0} for cache in model.l1))
    model = cold(); model.advance(17); assert model.submit(2, 0, {0}); model.advance(18); assert model.submit(3, 0, {0}); model.drain()
    check('cache classification occurs at lookup completion rather than submission', model.finished[2]['complete_cycle'] == 27 and model.finished[3]['complete_cycle'] == 21)
    check('L2-filled but L1-pending request uses L2 without another backing read', model.stats['dram_reads'] == 1 and model.stats['l2_hit_sectors'] == 1)
    model = cold(ReadConfig(mshrs=1, l2_slots=1, l2_queue=1), (0, 4, 8)); model.drain()
    check('finite MSHR and lookup pressure stages three backing reads at eight nineteen thirty', [r['start_cycle'] for r in model.dram_records] == [8, 19, 30])
    check('three-sector request returns only after last response at forty-three', model.finished[1]['complete_cycle'] == 43 and model.stats['returned_sectors'] == 3)
    check('queue and MSHR peaks obey independently configured unit capacities', model.peak['mshrs'] == model.peak['l2_slots'] == model.peak['l2_input'] == 1)
    check('finite staged request drains every internal reservation', model.snapshot() == {'cycle': 43, 'lsu_slots': [0], 'l2_input': 0, 'upstream_sectors': 0, 'l2_slots': 0, 'blocked_lookups': 0, 'mshrs': 0, 'outstanding_requests': 0})
    model = cold(ReadConfig(dram_interval=20), (0, 1)); model.drain()
    check('backing initiation interval serializes starts while MSHRs retain queued misses', [r['start_cycle'] for r in model.dram_records] == [8, 28] and model.finished[1]['complete_cycle'] == 41)
    check('different sectors in one line remain distinct missing data', model.stats['dram_reads'] == 2 and model.l2.sets[0][('global', 0)].valid == {0, 1})
    model = cold(ReadConfig(dram_latency=10, mshrs=1), (0,), bypass=True)
    model.advance(1); assert model.submit(2, 0, {4}, bypass_l1=True)
    model.advance(10); assert model.submit(3, 0, {8}, bypass_l1=True); model.drain()
    check('older blocked miss receives freed credit before a same-cycle new lookup',
          [(r['sector'], r['start_cycle']) for r in model.dram_records] == [(0, 5), (4, 15), (8, 25)])
    check('credit arbitration preserves old request progress and final return', model.finished[2]['complete_cycle'] == 27 and model.finished[3]['complete_cycle'] == 37)
    model = cold(ReadConfig(cache_sets=1, cache_ways=1), (0, 4)); model.drain(); assert model.submit(2, 0, {0}); model.drain()
    check('completed conflicting fills evict data and force a later backing miss', model.stats['dram_reads'] == 3 and model.finished[2]['complete_cycle'] == 43)
    model = cold(ReadConfig(lsu_slots_per_sm=1)); before = model.snapshot()
    check('full LSU refuses whole request without partial reservation', not model.submit(2, 0, {1}) and model.snapshot() == before and model.stats['accepted_requests'] == 1)
    model.drain(); assert model.submit(2, 0, {1}); model.drain()
    check('rejected request identity can be retried after capacity returns', model.stats['accepted_requests'] == 2 and model.stats['returned_sectors'] == 2)
    model = cold(); before = model.snapshot()
    check('one-cycle LSU port refuses a same-cycle second request', not model.submit(2, 0, {1}) and model.snapshot() == before)
    model.advance(1); check('port accepts on its declared next admission cycle', model.submit(2, 0, {1}))
    model.drain()
    for label, mutation in [('zero latency', {'dram_latency': 0}), ('boolean capacity', {'mshrs': True}),
                            ('fractional interval', {'l2_interval': 1.5}), ('missing assumptions', {'provenance': ''})]:
        reject(label + ' refused', lambda mutation=mutation: SectorReadService(replace(ReadConfig(), **mutation)))
    for label, action in [('duplicate request identity', lambda: model.submit(1, 0, {0})),
                          ('boolean identity', lambda: model.submit(True, 0, {0})),
                          ('invalid SM', lambda: model.submit(3, 1, {0})),
                          ('negative sector', lambda: model.submit(3, 0, {-1})),
                          ('boolean sector', lambda: model.submit(3, 0, {True})),
                          ('boolean hidden by duplicate equality', lambda: model.submit(3, 0, [1, True])),
                          ('empty request', lambda: model.submit(3, 0, set())),
                          ('excessive request width', lambda: model.submit(3, 0, range(129))),
                          ('backwards clock', lambda: model.advance(0)),
                          ('invalid bypass flag', lambda: model.submit(3, 0, {0}, bypass_l1=1))]:
        before = model.snapshot(); reject(label + ' refused', action)
        check(label + ' leaves reservations intact', model.snapshot() == before)
    # Unique sectors have an independent one-backing-read-per-sector oracle.
    # Mixed request widths exercise finite fanout and asynchronous slot reuse.
    config = ReadConfig(sm_count=2, lsu_slots_per_sm=3, lsu_interval=2, l2_slots=3, l2_queue=2,
                        mshrs=2, l2_interval=3, dram_interval=5, dram_latency=17, return_latency=4)
    model = SectorReadService(config); expected_sectors = 0
    for identity in range(1, 41):
        sectors = {identity * 16 + n for n in range(1 + identity % 4)}; expected_sectors += len(sectors)
        sm = identity % 2
        while not model.submit(identity, sm, sectors):
            model.advance(max(model.cycle + 1, model.next_admission(sm)))
    model.drain()
    check('forty mixed-width requests conserve unique backing reads and returned sectors',
          len(model.finished) == 40 and model.stats['dram_reads'] == model.stats['returned_sectors'] == expected_sectors)
    check('mixed-width service obeys all queue capacities and backing initiation intervals',
          model.peak['mshrs'] <= config.mshrs and model.peak['l2_input'] <= config.l2_queue and
          model.peak['l2_slots'] <= config.l2_slots and model.peak['lsu_slots'] <= config.sm_count * config.lsu_slots_per_sm and
          all(b['start_cycle'] - a['start_cycle'] >= config.dram_interval for a, b in zip(model.dram_records, model.dram_records[1:])) and
          all(r['complete_cycle'] - r['submit_cycle'] >= config.l1_latency + config.l2_latency + config.dram_latency + config.return_latency for r in model.finished.values()))
    return checks

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path); args = parser.parse_args()
    checks = verify_sector_reads()
    if args.out:
        root = Path(__file__).resolve().parents[1]; args.out.mkdir(parents=True, exist_ok=True)
        digest = lambda p: hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        report = {'kind': 'independent staged sector-read component references; no SM integration or hardware timing claim',
                  'passed': True, 'total_checks': len(checks), 'checks': checks, 'hardware_cycles': False,
                  'model_sha256': {n: digest(root / 'scripts' / n) for n in ['sector_read_service.py', 'memory_model.py']},
                  'reference_sha256': digest(Path(__file__))}
        (args.out / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS staged sector read components checks=' + str(len(checks)), flush=True)
