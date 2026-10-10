"""Independent resource-token and interval audit of staged service reports."""
from collections import defaultdict

def verify_service(service, rows, start, end):
    config = service['config']; count = config['sm_count']
    requests = {r['identity']: r for r in service['requests']}
    modeled = {r['sequence']: r for r in rows if r.get('memory')}
    assert len(requests) == len(service['requests']) and set(requests) == set(modeled), 'Memory request conservation'
    for identity, request in requests.items():
        row = modeled[identity]; memory = row['memory']
        assert request['sm'] == row['sm'] and request['submit_cycle'] == row['issue_cycle'] and request['complete_cycle'] == row['complete_cycle'], 'Response readiness'
        assert request['sectors'] == list(memory['sectors']) and request['operation'] == memory['operation'], 'Conserved request sectors'
    limits = {'lsu_slots': config['lsu_slots_per_sm'], 'l2_slots': config['l2_slots'], 'l2_input': config['l2_queue'],
              'mshrs': config['mshrs'], 'writeback_slots': config['writeback_slots'],
              'upstream': count * config['lsu_slots_per_sm'] * config['max_sectors_per_request']}
    live, begins, intervals, peaks = defaultdict(set), {}, defaultdict(list), defaultdict(int)
    previous = start; global_lsu_peak = 0
    for event in service['resource_records']:
        cycle, name, scope, delta = event['cycle'], event['resource'], event['scope'], event['delta']
        assert type(cycle) is int and previous <= cycle <= end and name in limits, 'Resource event order'
        assert type(delta) is int and delta in [-1, 1], 'Resource delta'
        assert type(scope) is int and (0 <= scope < count if name == 'lsu_slots' else scope == -1), 'Resource ownership'
        token = tuple(event['token']); key = (name, scope)
        if delta == 1:
            assert token not in live[key], 'Duplicate reservation'; live[key].add(token); begins[(key, token)] = cycle
        else:
            assert token in live[key], 'Release without reservation'; live[key].remove(token)
            intervals[(name, scope, token)].append((begins.pop((key, token)), cycle))
        assert len(live[key]) <= limits[name], 'Resource capacity: ' + name
        peaks[name] = max(peaks[name], len(live[key]))
        global_lsu_peak = max(global_lsu_peak, sum(len(tokens) for (resource, _), tokens in live.items() if resource == 'lsu_slots'))
        previous = cycle
    assert not begins and all(not tokens for tokens in live.values()), 'Complete reservation drain'
    for name in ['mshrs', 'l2_slots', 'l2_input', 'writeback_slots']:
        assert peaks[name] == service['peaks'].get(name, 0), 'Independent resource peak: ' + name
    assert global_lsu_peak == service['peaks'].get('lsu_slots', 0), 'Independent LSU peak'
    for identity, request in requests.items():
        assert intervals[('lsu_slots', request['sm'], (identity,))] == [(request['submit_cycle'], request['complete_cycle'])], 'LSU lifetime'
    lookups = service['lookup_records']; starts = sorted(r['start_cycle'] for r in lookups)
    assert all(b - a >= config['l2_interval'] for a, b in zip(starts, starts[1:])), 'L2 lookup bandwidth'
    for lookup in lookups:
        identity, sector = lookup['identity'], lookup['sector']; row = modeled[identity]
        earliest = row['issue_cycle'] + (0 if row['memory']['bypass_l1'] or row['memory']['operation'] == 'store' else config['l1_latency'])
        assert earliest <= lookup['start_cycle'] and lookup['lookup_complete_cycle'] == lookup['start_cycle'] + config['l2_latency'], 'L2 lookup latency'
        assert lookup['release_cycle'] >= lookup['lookup_complete_cycle'], 'Lookup retains blocked slot'
        assert intervals[('l2_slots', -1, (identity, sector))] == [(lookup['start_cycle'], lookup['release_cycle'])], 'Lookup slot lifetime'
    assert sum(len(v) for (name, _, _), v in intervals.items() if name == 'l2_slots') == len(lookups), 'Lookup ledger conservation'
    backing = service['dram_reads'] + service['writebacks']; starts = sorted(r['start_cycle'] for r in backing)
    assert all(b - a >= config['dram_interval'] for a, b in zip(starts, starts[1:])), 'Shared backing bandwidth'
    assert all(r['complete_cycle'] == r['start_cycle'] + config['dram_latency'] for r in backing), 'Backing service latency'
    for row in service['writebacks']:
        assert intervals[('writeback_slots', -1, (row['identity'],))] == [(row['reserve_cycle'], row['complete_cycle'])], 'Writeback retains queued tail'
    misses = defaultdict(list)
    for row in service['dram_reads']: misses[row['sector']].append(row)
    for sector, records in misses.items():
        holds = intervals[('mshrs', -1, (sector,))]
        assert len(records) == len(holds), 'Backing miss conservation'
        for row, (reserve, release) in zip(records, holds):
            assert reserve <= row['start_cycle'] < row['complete_cycle'] <= release, 'MSHR retains arrived-but-blocked data'
            assert all(row['start_cycle'] >= wb['complete_cycle'] for wb in service['writebacks'] if wb['sector'] == sector and wb['reserve_cycle'] <= reserve), 'Pending dirty version precedes backing read'
    stats = service['stats']; sectors = sum(len(r['sectors']) for r in requests.values())
    assert stats.get('accepted_requests', 0) == stats.get('completed_requests', 0) == len(requests), 'Request counter conservation'
    assert stats.get('requested_sectors', 0) == stats.get('returned_sectors', 0) == sectors, 'Sector counter conservation'
    assert stats.get('dram_reads', 0) == len(service['dram_reads']) and stats.get('l2_lookups', 0) == len(lookups), 'Service traffic conservation'
    assert stats.get('backing_write_sectors', 0) == stats.get('backing_write_completed_sectors', 0) == len(service['writebacks']), 'Writeback counter conservation'
    assert stats.get('store_commit_sectors', 0) == len(service['store_commits']) == sum(len(r['sectors']) for r in requests.values() if r['operation'] == 'store'), 'Store commitment conservation'
    for row in service['store_commits']:
        request = requests[row['identity']]
        assert request['operation'] == 'store' and row['sector'] in request['sectors'] and request['submit_cycle'] <= row['cycle'] <= request['complete_cycle'], 'Store commitment ownership'
    assert service['closed'] and service['final_snapshot']['cycle'] == end, 'Closed launch service'
    assert all(not any(v) if isinstance(v, list) else not v for k, v in service['final_snapshot'].items() if k != 'cycle'), 'Final service resources'
    return True
