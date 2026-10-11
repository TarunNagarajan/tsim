"""Staged, read-only sector service with completion-time fills and backpressure.

This base supplies asynchronous completions to the global memory/SM integration.
All geometry, ports and latencies are assumptions, not NVIDIA measurements.
Writes/coherence/ordering and DRAM bank scheduling require additional models.
"""
from collections import Counter, deque
from dataclasses import dataclass
import heapq

from memory_model import SectorCache, positive_integer

@dataclass(frozen=True)
class ReadConfig:
    sm_count: int = 1
    lsu_slots_per_sm: int = 8
    lsu_interval: int = 1
    l1_latency: int = 3
    l2_latency: int = 5
    return_latency: int = 2
    dram_latency: int = 11
    dram_interval: int = 1
    l2_interval: int = 1
    l2_queue: int = 8
    l2_slots: int = 4
    mshrs: int = 4
    max_sectors_per_request: int = 128
    cache_sets: int = 4
    cache_ways: int = 2
    provenance: str = 'Synthetic read-service assumptions; no hardware calibration'

    def validate(self):
        for name in self.__dataclass_fields__:
            if name != 'provenance': positive_integer(getattr(self, name), name)
        if not isinstance(self.provenance, str) or not self.provenance.strip(): raise ValueError('Missing read-service assumptions')

class SectorReadService:
    """One LSU admission port per SM and shared L2/DRAM service ports.

    LSU slots bound outstanding warp requests. A finite L2 input queue feeds
    finite lookup slots. A lookup waiting for an MSHR retains its slot. Misses
    to the same sector merge; cache data becomes valid only at actual return.
    Sector fanout waiting upstream remains bounded by LSU slots and the
    declared maximum request width. No request is partially admitted.
    """
    def __init__(self, config=None):
        self.config = config or ReadConfig(); self.config.validate()
        c = self.config
        self.l1 = [SectorCache(c.cache_sets, c.cache_ways) for _ in range(c.sm_count)]
        self.l2 = SectorCache(c.cache_sets, c.cache_ways)
        self.cycle = 0; self._serial = 0; self._events = []
        self.requests = {}; self.finished = {}; self._completed = []
        self._used_lsu = [0] * c.sm_count; self._next_lsu = [0] * c.sm_count
        self._feeds = deque(); self._input = deque(); self._blocked = deque()
        self._lookup_slots = 0; self._misses = {}; self._next_l2 = self._next_dram = 0
        self.stats = Counter(); self.peak = Counter(); self.dram_records = []
        self.resource_records = []; self.lookup_records = []; self._active_lookups = {}

    def _resource(self, name, token, delta, scope=-1):
        self.resource_records.append({'cycle': self.cycle, 'resource': name, 'scope': scope, 'token': list(token), 'delta': delta})

    def _feed(self, pairs):
        for pair in pairs:
            self._feeds.append(pair); self._resource('upstream', pair, 1)

    def _enqueue_input(self):
        while self._feeds and len(self._input) < self.config.l2_queue:
            pair = self._feeds.popleft(); self._resource('upstream', pair, -1)
            self._input.append(pair); self._resource('l2_input', pair, 1)

    def _release_lookup(self, pair):
        self._lookup_slots -= 1; self._resource('l2_slots', pair, -1)
        self._active_lookups.pop(pair)['release_cycle'] = self.cycle

    def _release_miss(self, sector):
        waiters = self._misses.pop(sector); self._resource('mshrs', (sector,), -1)
        return waiters

    def _event(self, when, stage, payload):
        self._serial += 1
        # Fills and returns precede same-cycle lookups.
        priority = {'writeback': 0, 'dram': 1, 'data': 2, 'l1': 3, 'l2': 4}[stage]
        heapq.heappush(self._events, (when, priority, self._serial, stage, payload))

    def can_submit(self, sm):
        """Read-only preflight for an indivisible mixed-space request."""
        if type(sm) is not int or not 0 <= sm < self.config.sm_count: raise ValueError('Invalid read-service SM')
        return self._used_lsu[sm] < self.config.lsu_slots_per_sm and self._next_lsu[sm] <= self.cycle

    def submit(self, identity, sm, sectors, *, bypass_l1=False):
        """Attempt admission at the current cycle, after advance(cycle).

        False means retry after a port/LSU-slot wakeup. Validation precedes
        admission; failed admission cannot mutate caches or reserve sectors.
        """
        positive_integer(identity, 'request identity')
        if identity in self.requests or identity in self.finished: raise ValueError('Duplicate read request identity')
        if type(sm) is not int or not 0 <= sm < self.config.sm_count: raise ValueError('Invalid read-service SM')
        if type(bypass_l1) is not bool: raise ValueError('Invalid L1 bypass flag')
        sectors = list(sectors)
        if any(type(s) is not int or s < 0 for s in sectors): raise ValueError('Invalid read sector')
        sectors = frozenset(sectors)
        if not sectors or len(sectors) > self.config.max_sectors_per_request: raise ValueError('Unsupported read request width')
        if self._used_lsu[sm] == self.config.lsu_slots_per_sm:
            self.stats['lsu_capacity_blocked_probes'] += 1; return False
        if self._next_lsu[sm] > self.cycle:
            self.stats['lsu_port_blocked_probes'] += 1; return False
        self.requests[identity] = {'identity': identity, 'sm': sm, 'sectors': sectors, 'remaining': set(sectors),
                                   'bypass_l1': bypass_l1, 'submit_cycle': self.cycle}
        self._used_lsu[sm] += 1; self._next_lsu[sm] = self.cycle + self.config.lsu_interval
        self._resource('lsu_slots', (identity,), 1, sm)
        self.stats['accepted_requests'] += 1; self.stats['requested_sectors'] += len(sectors)
        self.peak['lsu_slots'] = max(self.peak['lsu_slots'], sum(self._used_lsu))
        if bypass_l1:
            self.stats['l1_bypass_sectors'] += len(sectors)
            self._feed((identity, s) for s in sorted(sectors)); self._pump()
        else: self._event(self.cycle + self.config.l1_latency, 'l1', identity)
        return True

    def _response(self, pair):
        self._event(self.cycle + self.config.return_latency, 'data', pair)

    def _start_dram_read(self, sector):
        start = max(self.cycle, self._next_dram); self._next_dram = start + self.config.dram_interval
        end = start + self.config.dram_latency
        self.dram_records.append({'sector': sector, 'start_cycle': start, 'complete_cycle': end})
        self.stats['dram_reads'] += 1; self._event(end, 'dram', sector)

    def _complete_miss(self, sector):
        self.l2.request({sector})
        for pair in self._release_miss(sector): self._response(pair)
        self._retry_blocked()

    def _l2_hit(self, pair):
        self.stats['l2_hit_sectors'] += 1; self._release_lookup(pair); self._response(pair)

    def _lookup_result(self, pair):
        identity, sector = pair
        if sector in self._misses:
            self._misses[sector].append(pair); self.stats['merged_miss_sectors'] += 1
        elif len(self._misses) < self.config.mshrs:
            self._misses[sector] = [pair]
            self._resource('mshrs', (sector,), 1)
            self.peak['mshrs'] = max(self.peak['mshrs'], len(self._misses))
            self._start_dram_read(sector)
        else: return False
        self._release_lookup(pair)
        return True

    def _retry_blocked(self):
        for _ in range(len(self._blocked)):
            pair = self._blocked.popleft()
            if not self._lookup_result(pair): self._blocked.append(pair)

    def _pump(self):
        # Completed lookups retaining a slot retry before new lookups issue.
        self._retry_blocked()
        self._enqueue_input()
        self.peak['l2_input'] = max(self.peak['l2_input'], len(self._input))
        if self._input and self._lookup_slots < self.config.l2_slots and self._next_l2 <= self.cycle:
            pair = self._input.popleft(); self._lookup_slots += 1
            self._resource('l2_input', pair, -1); self._resource('l2_slots', pair, 1)
            record = {'identity': pair[0], 'sector': pair[1], 'start_cycle': self.cycle,
                      'lookup_complete_cycle': self.cycle + self.config.l2_latency}
            self.lookup_records.append(record); self._active_lookups[pair] = record
            self.peak['l2_slots'] = max(self.peak['l2_slots'], self._lookup_slots)
            self._next_l2 = self.cycle + self.config.l2_interval
            self.stats['l2_lookups'] += 1; self._event(self.cycle + self.config.l2_latency, 'l2', pair)
            self._enqueue_input()
            self.peak['l2_input'] = max(self.peak['l2_input'], len(self._input))

    def _process(self):
        while self._events and self._events[0][0] <= self.cycle:
            _, _, _, stage, payload = heapq.heappop(self._events)
            self._dispatch(stage, payload)
        self._pump()

    def _dispatch(self, stage, payload):
        if stage == 'dram': self._complete_miss(payload)
        elif stage == 'l1':
            request = self.requests[payload]; sm = request['sm']
            lookup = self.l1[sm].request(request['sectors'], allocate=False)
            self.stats['l1_hit_sectors'] += len(lookup['hits']); self.stats['l1_miss_sectors'] += len(lookup['misses'])
            for sector in sorted(lookup['hits']): self._event(self.cycle, 'data', (payload, sector))
            self._feed((payload, sector) for sector in sorted(lookup['misses']))
        elif stage == 'l2':
            identity, sector = payload
            lookup = self.l2.request({sector}, allocate=False)
            if lookup['hits']: self._l2_hit(payload)
            else:
                self.stats['l2_miss_sectors'] += 1
                if not self._lookup_result(payload): self._blocked.append(payload)
        else:
            identity, sector = payload; request = self.requests[identity]
            if not request['bypass_l1']: self.l1[request['sm']].request({sector})
            request['remaining'].remove(sector); self.stats['returned_sectors'] += 1
            if not request['remaining']:
                self._used_lsu[request['sm']] -= 1; self._resource('lsu_slots', (identity,), -1, request['sm'])
                self.requests.pop(identity)
                record = {'identity': identity, 'sm': request['sm'], 'submit_cycle': request['submit_cycle'],
                          'complete_cycle': self.cycle, 'sectors': sorted(request['sectors']), 'operation': request.get('operation', 'load')}
                self.finished[identity] = record; self._completed.append(record)
                self.stats['completed_requests'] += 1

    def next_wakeup(self):
        candidates = [self._events[0][0]] if self._events else []
        if self._input and self._lookup_slots < self.config.l2_slots:
            candidates.append(max(self.cycle + 1, self._next_l2))
        return min(candidates) if candidates else None

    def next_admission(self, sm):
        if type(sm) is not int or not 0 <= sm < self.config.sm_count: raise ValueError('Invalid read-service SM')
        if self._used_lsu[sm] < self.config.lsu_slots_per_sm: return max(self.cycle, self._next_lsu[sm])
        return self.next_wakeup()

    def advance(self, target):
        if type(target) is not int or target < self.cycle: raise ValueError('Read-service time cannot move backwards')
        self._process()
        while (wake := self.next_wakeup()) is not None and wake <= target:
            self.cycle = wake; self._process()
        self.cycle = target; self._process()
        complete = self._completed; self._completed = []
        return complete

    def _pending(self): return bool(self.requests)

    def _assert_drained(self):
        if self._events or self._feeds or self._input or self._blocked or self._misses or self._lookup_slots or self._active_lookups or any(self._used_lsu):
            raise AssertionError('Read-service conservation failure')

    def drain(self):
        complete = self.advance(self.cycle)
        while self._pending():
            wake = self.next_wakeup()
            if wake is None: raise AssertionError('Read-service deadlock')
            complete.extend(self.advance(wake))
        self._assert_drained()
        return complete

    def snapshot(self):
        return {'cycle': self.cycle, 'lsu_slots': list(self._used_lsu), 'l2_input': len(self._input),
                'upstream_sectors': len(self._feeds), 'l2_slots': self._lookup_slots,
                'blocked_lookups': len(self._blocked), 'mshrs': len(self._misses), 'outstanding_requests': len(self.requests)}
