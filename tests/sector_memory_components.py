"""Independently solved L2 store, eviction and backing-write schedules."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from sector_memory_service import MemoryConfig, SectorMemoryService, load_memory_config

def verify_sector_memory():
    checks = []
    def check(name, condition):
        if not condition: raise AssertionError(name)
        checks.append(name)
    def reject(name, action):
        try: action()
        except ValueError: checks.append(name)
        else: raise AssertionError(name)
    service = SectorMemoryService(); assert service.submit(1, 0, {0}, operation='store'); service.drain()
    check('cold global store bypasses L1 and acknowledges at eighteen', service.finished[1]['complete_cycle'] == 18 and not service.l1[0].sets[0])
    check('cold store conservatively fetches ownership before L2 commit at sixteen', service.dram_records == [{'sector': 0, 'start_cycle': 5, 'complete_cycle': 16}] and service.commit_records == [{'identity': 1, 'sector': 0, 'cycle': 16}])
    check('store data is dirty in L2 and not written to backing at acknowledgement', service.l2.sets[0][('global', 0)].dirty == {0} and not service.writeback_records)
    assert service.submit(2, 0, {0}, operation='store'); service.drain()
    check('warm store acknowledges after only lookup plus return', service.finished[2]['complete_cycle'] == 25 and service.stats['dram_reads'] == 1)
    service.finish()
    check('terminal dirty drain writes one latest sector and holds backing tail to thirty-six', service.cycle == 36 and service.stats['backing_write_sectors'] == service.stats['backing_write_completed_sectors'] == 1)
    check('terminal write is explicitly attributed and reserves storage until completion', service.writeback_records == [{'identity': 1, 'sector': 0, 'reserve_cycle': 25, 'start_cycle': 25, 'complete_cycle': 36, 'origin': 'terminal'}])
    reject('finished service refuses new requests', lambda: service.submit(3, 0, {1}))
    reject('finished service refuses duplicate flush', service.finish)
    service = SectorMemoryService(); assert service.submit(1, 0, {0}); service.advance(1)
    assert service.submit(2, 0, {0}, operation='store'); service.drain()
    check('load and store missing the same sector merge one ownership read', service.stats['dram_reads'] == 1 and service.stats['merged_miss_sectors'] == 1)
    check('merged store commits at seventeen and both requests acknowledge at nineteen', service.commit_records == [{'identity': 2, 'sector': 0, 'cycle': 17}] and [service.finished[n]['complete_cycle'] for n in [1, 2]] == [19, 19])
    check('store bypass does not invent L1 coherence or invalidate a modeled read fill', service.l1[0].sets[0][('global', 0)].valid == {0} and service.l2.sets[0][('global', 0)].dirty == {0})
    service.finish()
    service = SectorMemoryService(MemoryConfig(cache_sets=1, cache_ways=1))
    assert service.submit(1, 0, {0}, operation='store'); service.drain()
    assert service.submit(2, 0, {4}, operation='store'); service.drain()
    # Second store commits at34/acks36; its dirty victim writes34..45.
    check('dirty eviction remains live after the store acknowledges', service.finished[2]['complete_cycle'] == 36 and service.cycle == 45)
    check('dirty eviction uses the shared backing port with exact retained identity', service.writeback_records[0] == {'identity': 1, 'sector': 0, 'reserve_cycle': 34, 'start_cycle': 34, 'complete_cycle': 45, 'origin': 'eviction'})
    # Submit during that tail using a fresh independently timed replay.
    concurrent = SectorMemoryService(MemoryConfig(cache_sets=1, cache_ways=1))
    assert concurrent.submit(1, 0, {0}, operation='store'); concurrent.advance(18)
    assert concurrent.submit(2, 0, {4}, operation='store'); concurrent.advance(36)
    assert concurrent.submit(3, 0, {0}, bypass_l1=True); concurrent.drain()
    check('read of an evicted dirty sector cannot precede its pending backing write', concurrent.dram_records[-1] == {'sector': 0, 'start_cycle': 45, 'complete_cycle': 56})
    check('read returns at fifty-eight while another dirty eviction holds drain to sixty-seven', concurrent.finished[3]['complete_cycle'] == 58 and concurrent.cycle == 67)
    concurrent.finish()
    check('clean final cache requires no extra terminal write', concurrent.cycle == 67 and concurrent.stats['terminal_write_sectors'] == 0)
    config = MemoryConfig(cache_sets=1, cache_ways=1, mshrs=8, l2_slots=8, writeback_slots=4)
    service = SectorMemoryService(config); assert service.submit(1, 0, range(0, 28, 4), operation='store'); service.drain()
    check('full writeback storage parks completed fills without discarding dirty victims', service.peak['writeback_slots'] == 4 and service.peak['ready_fills'] == 2 and service.stats['fill_writeback_blocked_probes'] == 2)
    check('blocked dirty replacements resume only at available writeback credits', [r['reserve_cycle'] for r in service.writeback_records] == [17, 18, 19, 20, 28, 29])
    check('store acknowledgement precedes backing drain while all ownership misses remain conserved', service.finished[1]['complete_cycle'] == 31 and service.cycle == 40 and service.stats['dram_reads'] == service.stats['store_commit_sectors'] == 7)
    service.finish()
    check('eviction and terminal packets together write every distinct dirty sector once', service.cycle == 51 and service.stats['backing_write_sectors'] == service.stats['backing_write_completed_sectors'] == 7)
    check('final controller has no requests, misses, queues, fills or backing reservations', all(not any(v) if isinstance(v, list) else not v for k, v in service.snapshot().items() if k != 'cycle'))
    # Independently sweep retained writeback lifetimes, including queued tails.
    active = peak = 0
    for _, delta in sorted([(r['reserve_cycle'], 1) for r in service.writeback_records] + [(r['complete_cycle'], -1) for r in service.writeback_records]):
        active += delta; peak = max(peak, active)
        if not 0 <= active <= config.writeback_slots: raise AssertionError('Writeback interval capacity')
    check('independent writeback interval sweep matches queue peak and conservation', active == 0 and peak == service.peak['writeback_slots'])
    starts = sorted(r['start_cycle'] for r in service.dram_records + service.writeback_records)
    check('reads and writes obey one shared backing initiation interval', all(b - a >= config.dram_interval for a, b in zip(starts, starts[1:])))
    for label, mutation in [('too-small victim storage', {'writeback_slots': 3}), ('boolean writeback capacity', {'writeback_slots': True}), ('zero backing interval', {'dram_interval': 0})]:
        reject(label + ' refused', lambda mutation=mutation: SectorMemoryService(replace(MemoryConfig(), **mutation)))
    service = SectorMemoryService()
    reject('atomic service refused until explicitly modeled', lambda: service.submit(1, 0, {0}, operation='atomic'))
    reject('unknown operation refused before resource mutation', lambda: service.submit(1, 0, {0}, operation='prefetch'))
    check('unsupported operation leaves an empty controller', service.snapshot()['outstanding_requests'] == service.stats['accepted_requests'] == 0)
    return checks

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path); args = parser.parse_args()
    checks = verify_sector_memory()
    if args.out:
        root = Path(__file__).resolve().parents[1]; args.out.mkdir(parents=True, exist_ok=True)
        digest = lambda p: hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        report = {'kind': 'independent staged global load/store and writeback references; uncalibrated', 'passed': True,
                  'total_checks': len(checks), 'checks': checks, 'hardware_cycles': False,
                  'model_sha256': {n: digest(root / 'scripts' / n) for n in ['sector_memory_service.py', 'sector_read_service.py', 'memory_model.py']},
                  'reference_sha256': digest(Path(__file__))}
        (args.out / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS staged global memory components checks=' + str(len(checks)), flush=True)
