"""Address transaction models and a configurable sector-valid cache component.

PTX requests and synthetic cache behavior are distinct from measured GPU cycles.
"""
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import re

BEST_PRACTICES = 'https://docs.nvidia.com/cuda/archive/12.6.3/cuda-c-best-practices-guide/index.html'
PROGRAMMING = 'https://docs.nvidia.com/cuda/archive/12.6.3/cuda-c-programming-guide/index.html'
NSIGHT = 'https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#metrics-guide'
LOCAL_BASE = 0x6fff00000000
SHARED_BASE = 0x6ffe00000000
ENGINE_PIN = '771bb8f96abc3721700219d88fa74e10d43dcfd5'

def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(name + ' must be a positive integer')
    return value

def sectors_for_spans(spans, sector_bytes=32):
    """Union requested byte spans; preserve partial and crossing sectors."""
    positive_integer(sector_bytes, 'sector_bytes')
    result = set()
    for address, size in spans:
        positive_integer(size, 'access width')
        if isinstance(address, bool) or not isinstance(address, int) or address < 0 or address + size > 1 << 64:
            raise ValueError('Access outside unsigned 64-bit address space')
        result.update(range(address // sector_bytes, (address + size - 1) // sector_bytes + 1))
    return result

def local_spans(accesses, warp_size=32):
    """A word-striped private layout model; namespace ownership is separate.

    Same relative word across lanes is coalesced, never one broadcast word.
    No physical base/partition hashing or stack placement is inferred.
    """
    if warp_size != 32: raise ValueError('Local layout currently supports NVIDIA warp size 32')
    spans = []
    for event in accesses:
        relative = int(event['address'], 16) - LOCAL_BASE
        if relative < 0 or relative >= 1 << 30 or not 0 <= event['lane'] < warp_size:
            raise ValueError('Invalid local owner/address')
        remaining, position = event['bytes'], relative
        while remaining:
            word, offset = divmod(position, 4)
            size = min(remaining, 4 - offset)
            spans.append((word * warp_size * 4 + event['lane'] * 4 + offset, size))
            remaining -= size; position += size
    return spans

def shared_scalar_request(accesses, operation):
    """Exact scalar word/bank topology; no wide request splitting prediction."""
    if operation not in ['load', 'store', 'atomic']: raise ValueError('Unknown shared operation')
    words = defaultdict(set)
    owners = defaultdict(set)
    ranks = set()
    for event in accesses:
        address = int(event['address'], 16)
        if event['bytes'] != 4 or address % 4 or event['element'] != 0:
            raise ValueError('Shared topology currently supports scalar aligned 32-bit accesses')
        if not SHARED_BASE <= address < SHARED_BASE + (1 << 30):
            raise ValueError('Shared topology requires the pinned engine address window')
        ranks.add((address - SHARED_BASE) >> 24)
        word = address // 4
        words[word % 32].add(word)
        owners[word].add(event['lane'])
    if len(ranks) > 1:
        raise ValueError('Distributed shared requests spanning owners require a separate model')
    result = {'banks_touched': len(words), 'distinct_words': len(owners),
              'same_word_extra_lanes': sum(len(v) - 1 for v in owners.values()),
              'atomic_updates': len(accesses) if operation == 'atomic' else 0}
    if operation == 'atomic':
        result['max_updates_to_one_word'] = max(map(len, owners.values()), default=0)
    else:
        result['bank_conflict_degree'] = max(map(len, words.values()), default=0)
        result['write_alias_words'] = sum(len(v) > 1 for v in owners.values()) if operation == 'store' else 0
    return result

def constant_scalar_request(accesses):
    if any(r['bytes'] != 4 or r['element'] for r in accesses):
        raise ValueError('Constant topology currently supports scalar 32-bit reads')
    return len({int(r['address'], 16) for r in accesses})

def compiled_memory_statements(ptx):
    """Locate complete memory statements, including inline/multiline operands.

    Semicolons terminate ld/st/atom/red operands; comments must not create fake
    statements. Preserve comment newlines to retain the compiler's line IDs.
    """
    clean = re.sub(r'/\*.*?\*/|//[^\n]*', lambda match: '\n' * match[0].count('\n'), ptx, flags=re.S)
    pattern = r'(?:@\s*!?\s*%[A-Za-z0-9_$]+\s+)?(?:ld|st|atom|red)\.[A-Za-z0-9_.:$]+(?=\s|\[|\{)[^;]*;'
    result = defaultdict(set)
    for match in re.finditer(pattern, clean):
        if match.start() and (clean[match.start() - 1].isalnum() or clean[match.start() - 1] in '_.$%'):
            continue
        line = clean.count('\n', 0, match.start()) + 1
        result[line].add(re.sub(r'\s+', '', match[0]).rstrip(';'))
    return result

@dataclass
class CacheLine:
    valid: set = field(default_factory=set)
    dirty: set = field(default_factory=set)

class SectorCache:
    """Parameterized set-associative LRU cache with independent valid sectors.

    Modulo set indexing, LRU, allocation/fill policies are model assumptions.
    A tag hit never makes absent sector data a cache hit.
    """
    def __init__(self, sets, ways, line_bytes=128, sector_bytes=32):
        for name, value in [('sets', sets), ('ways', ways), ('line_bytes', line_bytes), ('sector_bytes', sector_bytes)]:
            positive_integer(value, name)
        if line_bytes % sector_bytes: raise ValueError('Cache line must contain whole sectors')
        self.set_count, self.ways = sets, ways
        self.sectors_per_line = line_bytes // sector_bytes
        self.line_bytes, self.sector_bytes = line_bytes, sector_bytes
        self.sets = [OrderedDict() for _ in range(sets)]
        self.stats = Counter()

    def request(self, sectors, namespace='global', write=False, allocate=True):
        groups = defaultdict(set)
        for sector in set(sectors):
            if isinstance(sector, bool) or not isinstance(sector, int) or sector < 0:
                raise ValueError('Invalid cache sector identity')
            line, within = divmod(sector, self.sectors_per_line)
            groups[line].add(within)
        hits, misses, evicted_dirty = set(), set(), []
        if groups: self.stats['requests'] += 1
        # All sectors for a tag are queried together, not as artificial extra
        # tag lookups caused by the order the lanes happened to be serialized.
        for line, requested in sorted(groups.items()):
            bucket = self.sets[line % self.set_count]
            key = (namespace, line)
            entry = bucket.get(key)
            self.stats['tag_lookups'] += 1
            self.stats['tag_hits' if entry is not None else 'tag_misses'] += 1
            valid = entry.valid if entry else set()
            present, absent = requested & valid, requested - valid
            hits.update(line * self.sectors_per_line + i for i in present)
            misses.update(line * self.sectors_per_line + i for i in absent)
            self.stats['sector_hits'] += len(present)
            self.stats['sector_misses'] += len(absent)
            if entry is None and allocate:
                if len(bucket) == self.ways:
                    old_key, old = bucket.popitem(last=False)
                    self.stats['evictions'] += 1
                    self.stats['dirty_sectors_evicted'] += len(old.dirty)
                    evicted_dirty.extend((old_key[0], old_key[1] * self.sectors_per_line + i) for i in sorted(old.dirty))
                entry = CacheLine(); bucket[key] = entry
            if entry is not None:
                if allocate:
                    entry.valid.update(requested)
                    self.stats['sector_fills'] += len(absent)
                if write: entry.dirty.update(present | (absent if allocate else set()))
                bucket.move_to_end(key)
            self.stats['write_access_sectors' if write else 'read_access_sectors'] += len(requested)
        return {'hits': hits, 'misses': misses, 'evicted_dirty': evicted_dirty}

    def drain_dirty(self):
        """Explicit model-boundary writeback; not inferred hardware behavior."""
        dirty = []
        for bucket in self.sets:
            for (namespace, line), entry in bucket.items():
                dirty.extend((namespace, line * self.sectors_per_line + i) for i in sorted(entry.dirty))
                entry.dirty.clear()
        self.stats['dirty_sectors_drained'] += len(dirty)
        return dirty

def validate_compilation_artifacts(run_directory):
    """Validate pinned run identity and unchanged compiled inputs/host binary."""
    run = Path(run_directory).resolve()
    metadata = json.loads((run / 'run.json').read_text())
    if metadata['engine_revision'] != ENGINE_PIN or metadata['gpu'] not in ['nvidia/a100', 'nvidia/h100']:
        raise ValueError('Transaction model currently requires the pinned engine and A100/H100 profiles')
    if set(metadata['compilation_sha256']) != {'program', 'kernel.ptx'}:
        raise ValueError('Missing compilation provenance')
    for name, expected in metadata['compilation_sha256'].items():
        if name not in ['program', 'kernel.ptx']: raise ValueError('Unexpected compilation artifact')
        if hashlib.sha256((run / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Compilation artifact changed: ' + name)
    return metadata

def analyze_transactions(run_directory):
    """Analyze complete observed execution; refuse partial coverage.

    Cache components are tested separately; this report does not yet model a
    GPU cache hierarchy or convert topological counts into hardware cycles.
    """
    from execution_events import summarize_events
    run = Path(run_directory).resolve()
    metadata = validate_compilation_artifacts(run)
    events_path = run / 'execution-events.jsonl'
    summary = summarize_events(events_path)
    if metadata['exit_code'] or metadata['timed_out'] or not summary['complete_supported_memory']:
        raise ValueError('Memory model requires successful, complete, conserved event capture without gaps')
    statements = compiled_memory_statements((run / 'kernel.ptx').read_text())
    normalize = lambda text: re.sub(r'\s+', '', text).rstrip(';')
    issues, pending = {}, defaultdict(list)
    totals = Counter()
    requests = []
    with events_path.open() as source:
        for line in source:
            event = json.loads(line)
            if event['type'] == 'issue':
                line_number = event['ptx_line']
                if event['expected_memory_accesses'] and (isinstance(line_number, bool) or not isinstance(line_number, int) or
                        normalize(event['instruction']) not in statements.get(line_number, set())):
                    raise ValueError('Captured instruction differs from compiled PTX')
                issues[event['sequence']] = event
            elif event['type'] == 'memory': pending[event['issue']].append(event)
            elif event['type'] == 'issue_end':
                issue = issues.pop(event['issue'])
                groups = defaultdict(list)
                for access in pending.pop(event['issue'], []): groups[(access['space'], access['operation'])].append(access)
                for (space, operation), accesses in groups.items():
                    record = {'issue': event['issue'], 'launch': issue['launch'], 'block': issue['block'],
                              'warp': issue['warp'], 'space': space, 'operation': operation,
                              'instruction': issue['instruction'],
                              'lane_accesses': len(accesses), 'requested_bytes': sum(r['bytes'] for r in accesses)}
                    if space in ['global', 'local']:
                        spans = local_spans(accesses) if space == 'local' else [(int(r['address'], 16), r['bytes']) for r in accesses]
                        sectors = sectors_for_spans(spans)
                        record['sectors'] = len(sectors)
                        record['sector_ids'] = [hex(s) for s in sorted(sectors)]
                        record['sector_bytes'] = len(sectors) * 32
                        if space == 'local': record['ownership'] = 'private to launch/block/warp/lane; word-striped model'
                    elif space == 'shared':
                        if all(a['bytes']==4 and a['element']==0 for a in accesses):
                            record.update(shared_scalar_request(accesses, operation))
                        else:
                            from shared_memory_service import SharedRequest, shared_topology
                            record.update(shared_topology(SharedRequest.captured(operation,issue['block'],accesses)))
                    elif space == 'constant':
                        if operation != 'load': raise ValueError('Constant memory is read-only')
                        record['distinct_constant_addresses'] = constant_scalar_request(accesses)
                    elif space == 'param':
                        record['excluded_from_physical_transactions'] = True
                    for key in ['lane_accesses', 'requested_bytes', 'sectors', 'sector_bytes', 'atomic_updates', 'write_alias_words', 'distinct_constant_addresses']:
                        if key in record: totals[space + '_' + operation + '_' + key] += record[key]
                    requests.append(record)
    return {'schema': 1, 'kind': 'logical PTX address transactions; not SASS requests or measured GPU timing',
            'event_sha256': hashlib.sha256(events_path.read_bytes()).hexdigest(),
            'complete_supported_memory': True, 'hardware_cycles': False, 'cache_hierarchy_simulated': False,
            'totals': dict(sorted(totals.items())), 'requests': requests,
            'provenance': {'global_sector_bytes': {'value': 32, 'kind': 'public constraint', 'source': BEST_PRACTICES + '#coalesced-access-to-global-memory'},
                           'shared_banks': {'value': 32, 'word_bytes': 4, 'kind': 'public constraint', 'source': BEST_PRACTICES + '#shared-memory-and-memory-banks'},
                           'local_layout': {'kind': 'word-striped layout assumption constrained by public coalescing rule', 'source': PROGRAMMING + '#device-memory-accesses',
                                            'physical_base_and_partition_hash': 'unverified'},
                           'constant_requests': {'kind': 'public distinct-address rule; scalar 32-bit subset', 'source': PROGRAMMING + '#device-memory-accesses'},
                           'cache_component': {'kind': 'configurable LRU/set-index/fill assumptions; not wired to GPU hierarchy', 'sector_and_line_source': NSIGHT}},
            'limits': ['PTX can lower to multiple SASS operations', 'wide shared accesses use logical byte-span word work, without predicting SASS splitting',
                       'atomic updates are counted without latency predictions', 'parameter ABI traffic excluded',
                       'local physical placement is unverified', 'SM/cache hierarchy and timing remain open']}
