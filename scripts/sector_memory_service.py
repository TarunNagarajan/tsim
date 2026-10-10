"""Staged global loads/stores with finite writeback storage and actual fills.

PTX hints are performance policy. This service does not carry data values or
establish weak-memory outcomes. Shared ports, RFO, fill/LRU and cold drains are
declared assumptions, with no proprietary or measured hardware-cycle claim.
"""
from collections import deque
from dataclasses import asdict, dataclass
import json
from pathlib import Path

from sector_read_service import ReadConfig, SectorReadService

@dataclass(frozen=True)
class MemoryConfig(ReadConfig):
    writeback_slots: int = 8
    provenance: str = 'Synthetic load/store service: assumed ports, geometry, latency, ownership reads and dirty drains; no hardware calibration'

    def validate(self):
        super().validate()
        if self.writeback_slots < 4: raise ValueError('Writeback storage must hold a whole four-sector victim line')

    def record(self):
        self.validate(); return {'schema': 1, 'provenance_kind': 'assumption', **asdict(self)}

def load_memory_config(path):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or set(data) != set(MemoryConfig.__dataclass_fields__) | {'schema', 'provenance_kind'}:
        raise ValueError('Malformed sector-service configuration')
    schema = data.pop('schema')
    if type(schema) is not int or schema != 1: raise ValueError('Unsupported sector-service schema')
    if data.pop('provenance_kind') != 'assumption': raise ValueError('Hardware memory timing requires measurement validation')
    config = MemoryConfig(**data); config.validate(); return config

class SectorMemoryService(SectorReadService):
    """Global ca/cg loads and wb/cg stores, with an L2 write-back model.

    Global stores bypass L1. Missing store sectors use read-for-ownership, an
    explicit conservative fill assumption independent of byte coverage.
    Store acknowledgement follows L2 commit; dirty backing writes have their
    own completion tail. L1 coherence/data values are not simulated.
    """
    def __init__(self, config=None):
        super().__init__(config or MemoryConfig())
        if not isinstance(self.config, MemoryConfig): raise ValueError('Missing writeback configuration')
        self._ready_fills = deque(); self._writebacks = {}; self._wb_serial = 0
        self._latest_wb = {}; self._terminal = deque(); self.closed = False
        self.writeback_records = []; self.commit_records = []

    def submit(self, identity, sm, sectors, *, operation='load', bypass_l1=False):
        if self.closed: raise ValueError('Memory service is closed')
        if operation not in ['load', 'store']: raise ValueError('Unsupported timed memory operation')
        if type(bypass_l1) is not bool: raise ValueError('Invalid L1 bypass flag')
        accepted = super().submit(identity, sm, sectors, bypass_l1=operation == 'store' or bypass_l1)
        if accepted: self.requests[identity]['operation'] = operation
        return accepted

    def _start_dram_read(self, sector):
        # A queued dirty version must reach backing storage before a read of
        # that sector can consume backing data. The backing port is FIFO.
        start = max(self.cycle, self._next_dram, self._latest_wb.get(sector, 0))
        self._next_dram = start + self.config.dram_interval; end = start + self.config.dram_latency
        self.dram_records.append({'sector': sector, 'start_cycle': start, 'complete_cycle': end})
        self.stats['dram_reads'] += 1; self._event(end, 'dram', sector)

    def _send_writeback(self, sector, origin):
        if len(self._writebacks) >= self.config.writeback_slots: raise AssertionError('Writeback capacity exceeded')
        self._wb_serial += 1; identity = self._wb_serial
        start = max(self.cycle, self._next_dram); self._next_dram = start + self.config.dram_interval
        end = start + self.config.dram_latency
        self._writebacks[identity] = sector; self._latest_wb[sector] = end
        self._resource('writeback_slots', (identity,), 1)
        self.peak['writeback_slots'] = max(self.peak['writeback_slots'], len(self._writebacks))
        self.writeback_records.append({'identity': identity, 'sector': sector, 'reserve_cycle': self.cycle,
                                       'start_cycle': start, 'complete_cycle': end, 'origin': origin})
        self.stats['backing_write_sectors'] += 1; self._event(end, 'writeback', identity)

    def _victim(self, sector):
        line = sector // self.l2.sectors_per_line; bucket = self.l2.sets[line % self.l2.set_count]
        if ('global', line) in bucket or len(bucket) < self.l2.ways: return []
        (namespace, old_line), entry = next(iter(bucket.items()))
        if namespace != 'global': raise AssertionError('Unexpected global cache ownership')
        return [old_line * self.l2.sectors_per_line + n for n in sorted(entry.dirty)]

    def _store_commit(self, pair):
        identity, sector = pair
        if self.requests[identity]['operation'] == 'store':
            self.l2.request({sector}, write=True)
            self.stats['store_commit_sectors'] += 1
            self.commit_records.append({'identity': identity, 'sector': sector, 'cycle': self.cycle})

    def _l2_hit(self, pair):
        self._store_commit(pair); super()._l2_hit(pair)

    def _try_fill(self, sector):
        dirty = self._victim(sector)
        if len(self._writebacks) + len(dirty) > self.config.writeback_slots: return False
        result = self.l2.request({sector})
        if [s for namespace, s in result['evicted_dirty']] != dirty: raise AssertionError('Dirty victim changed during fill')
        for old in dirty: self._send_writeback(old, 'eviction')
        for pair in self._release_miss(sector):
            self._store_commit(pair); self._response(pair)
        self._retry_blocked()
        return True

    def _complete_miss(self, sector):
        if not self._try_fill(sector):
            self._ready_fills.append(sector); self.stats['fill_writeback_blocked_probes'] += 1
            self.peak['ready_fills'] = max(self.peak['ready_fills'], len(self._ready_fills))

    def _retry_fills(self):
        for _ in range(len(self._ready_fills)):
            sector = self._ready_fills.popleft()
            if not self._try_fill(sector): self._ready_fills.append(sector)

    def _pump(self):
        self._retry_fills(); super()._pump()
        while self._terminal and len(self._writebacks) < self.config.writeback_slots:
            self._send_writeback(self._terminal.popleft(), 'terminal')

    def _dispatch(self, stage, payload):
        if stage == 'writeback':
            self._writebacks.pop(payload); self.stats['backing_write_completed_sectors'] += 1
            self._resource('writeback_slots', (payload,), -1)
            self._retry_fills()
        else: super()._dispatch(stage, payload)

    def _pending(self): return super()._pending() or bool(self._writebacks or self._terminal)

    def _assert_drained(self):
        super()._assert_drained()
        if self._writebacks or self._terminal or self._ready_fills: raise AssertionError('Memory service conservation failure')

    def finish(self):
        if self.closed: raise ValueError('Memory service already finished')
        completed = self.drain(); self.closed = True
        self._terminal.extend(s for namespace, s in self.l2.drain_dirty())
        self.stats['terminal_write_sectors'] += len(self._terminal)
        completed.extend(self.drain()); return completed

    def snapshot(self):
        return {**super().snapshot(), 'writeback_slots': len(self._writebacks),
                'ready_fills': len(self._ready_fills), 'terminal_sectors': len(self._terminal)}

    def port_readiness(self):
        return {'lsu': list(self._next_lsu), 'l2': self._next_l2, 'backing': self._next_dram}

    def restore_port_readiness(self, readiness):
        if self.closed or self.stats['accepted_requests']: raise ValueError('Port restoration requires a fresh service')
        if not isinstance(readiness, dict) or set(readiness) != {'lsu', 'l2', 'backing'}:
            raise ValueError('Malformed memory port readiness')
        lsu = readiness['lsu']
        if not isinstance(lsu, list) or len(lsu) != self.config.sm_count: raise ValueError('Memory port SM count changed')
        if any(type(n) is not int or n < 0 for n in lsu + [readiness['l2'], readiness['backing']]):
            raise ValueError('Invalid memory port deadline')
        self._next_lsu = list(lsu); self._next_l2 = readiness['l2']; self._next_dram = readiness['backing']
