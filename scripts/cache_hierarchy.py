"""Configurable traffic replay from actual PTX addresses, without GPU cycles.

The functional parser retains full instruction spelling even when it discards
performance modifiers. Decode that preserved spelling, never the stripped AST.
"""
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import re

from memory_model import SectorCache, analyze_transactions, positive_integer

PTX = 'https://docs.nvidia.com/cuda/archive/12.6.3/parallel-thread-execution/index.html#cache-operators'

@dataclass(frozen=True)
class MemoryPolicy:
    opcode: str
    operation: str
    cache_operator: str
    ordering: str
    scope: str
    noncoherent: bool
    hints: tuple

def decode_policy(instruction, operation):
    """Preserve every cache hint and require an unambiguous base opcode.

    A profile may then reject a parsed hint it cannot model. Predicate prefixes
    are excluded from the opcode, and L1/L2 :: qualifiers are never dropped.
    """
    match = re.match(r'^\s*(?:@\s*!?\s*%[A-Za-z0-9_$]+\s+)?([A-Za-z][A-Za-z0-9_.:$]*)(?=\s|\[|\{)', instruction)
    if not match: raise ValueError('Cannot decode preserved PTX instruction')
    opcode = match[1]
    parts = opcode.split('.')
    base = parts.pop(0)
    expected = {'ld': 'load', 'st': 'store', 'atom': 'atomic', 'red': 'atomic'}
    if base not in expected or expected[base] != operation:
        raise ValueError('Instruction disagrees with captured memory operation')
    operators, orders, scopes, hints = [], [], [], []
    noncoherent = False
    for part in parts:
        if part in ['ca', 'cg', 'cs', 'lu', 'cv', 'wb', 'wt']: operators.append(part)
        elif part in ['weak', 'volatile', 'relaxed', 'acquire', 'release', 'acq_rel']: orders.append(part)
        elif part in ['cta', 'cluster', 'gpu', 'sys']: scopes.append(part)
        elif part == 'nc': noncoherent = True
        elif part.startswith(('L1::', 'L2::')): hints.append(part)
        elif part in ['global', 'shared', 'shared::cta', 'shared::cluster', 'local', 'param', 'const', 'v2', 'v4', 'v8', 'noftz']:
            pass
        elif re.fullmatch(r'[busf](?:8|16|32|64|128)|f16x2|bf16|bf16x2', part): pass
        elif base in ['atom', 'red'] and part in ['add', 'and', 'or', 'xor', 'exch', 'cas', 'min', 'max', 'inc', 'dec']: pass
        else: raise ValueError('Unmodeled PTX memory modifier: .' + part)
    if len(operators) > 1 or len(orders) > 1 or len(scopes) > 1:
        raise ValueError('Ambiguous PTX memory policy')
    default = {'load': 'ca', 'store': 'wb', 'atomic': 'atomic-l2'}[operation]
    legal = {'load': {'ca', 'cg', 'cs', 'lu', 'cv'}, 'store': {'wb', 'cg', 'cs', 'wt'}, 'atomic': set()}
    if operators and operators[0] not in legal[operation]: raise ValueError('Illegal PTX cache operator for operation')
    return MemoryPolicy(opcode, operation, operators[0] if operators else default,
                        orders[0] if orders else ('relaxed' if operation == 'atomic' else 'weak'),
                        scopes[0] if scopes else ('gpu' if operation == 'atomic' else ''), noncoherent, tuple(hints))

DEFAULT_PROFILE = {
    'schema': 1, 'name': 'synthetic-two-level', 'sm_count': 1,
    'l1': {'sets': 64, 'ways': 4, 'line_bytes': 128, 'sector_bytes': 32},
    'l2': {'sets': 256, 'ways': 8, 'line_bytes': 128, 'sector_bytes': 32},
    'placement': 'linear-block-modulo-sm', 'replacement': 'LRU',
    'fill': 'requested-sectors', 'write_miss': 'read-for-ownership',
    'launch_boundary': 'cold-start-drain-dirty', 'ordering': 'captured-completion-order',
    'provenance': 'model assumptions; no NVIDIA geometry, scheduling or latency calibration'
}

def validate_profile(profile):
    if not isinstance(profile, dict) or set(profile) != set(DEFAULT_PROFILE):
        raise ValueError('Cache profile must declare every supported field without unknown fields')
    if profile['schema'] != 1 or isinstance(profile['schema'], bool): raise ValueError('Unsupported cache profile schema')
    if not isinstance(profile['name'], str) or not profile['name']: raise ValueError('Missing cache profile name')
    sm_count = positive_integer(profile['sm_count'], 'sm_count')
    if sm_count > 1024: raise ValueError('Unreasonable synthetic SM count')
    for level in ['l1', 'l2']:
        if not isinstance(profile[level], dict) or set(profile[level]) != {'sets', 'ways', 'line_bytes', 'sector_bytes'}:
            raise ValueError('Incomplete or unknown cache geometry')
        geometry = profile[level]
        for key, value in geometry.items(): positive_integer(value, key)
        if geometry['sector_bytes'] != 32 or geometry['line_bytes'] != 128:
            raise ValueError('Hierarchy currently requires 32-byte sectors and 128-byte lines')
        if geometry['sets'] * geometry['ways'] * (sm_count if level == 'l1' else 1) > 1048576:
            raise ValueError('Unreasonable aggregate cache geometry')
    for key in ['placement', 'replacement', 'fill', 'write_miss', 'launch_boundary', 'ordering', 'provenance']:
        if profile[key] != DEFAULT_PROFILE[key]: raise ValueError('Unimplemented cache profile rule: ' + key)
    return json.loads(json.dumps(profile))

class CacheHierarchy:
    """Sector-state traffic model; no data values, coherence protocol or queues."""
    def __init__(self, profile):
        self.profile = validate_profile(profile)
        self.l1 = [SectorCache(**self.profile['l1']) for _ in range(self.profile['sm_count'])]
        self.l2 = SectorCache(**self.profile['l2'])
        self.traffic = Counter()
        self.finished = False

    def _l2(self, sectors, namespace, write=False):
        result = self.l2.request(sectors, namespace=namespace, write=write)
        self.traffic['backing_read_sectors'] += len(result['misses'])
        self.traffic['backing_write_sectors'] += len(result['evicted_dirty'])
        return result

    def _writebacks(self, dirty, source):
        groups = defaultdict(set)
        for namespace, sector in dirty: groups[namespace].add(sector)
        for namespace, sectors in groups.items():
            self.traffic[source + '_writeback_sectors'] += len(sectors)
            self._l2(sectors, namespace, write=True)

    def request(self, sectors, operation, policy, sm=0, namespace='global'):
        if self.finished: raise ValueError('Cache hierarchy already drained')
        if isinstance(sm, bool) or not isinstance(sm, int) or not 0 <= sm < len(self.l1): raise ValueError('Invalid synthetic SM identity')
        if operation != policy.operation: raise ValueError('Policy operation mismatch')
        if namespace != 'global' and not (isinstance(namespace, str) and namespace.startswith('local:')):
            raise ValueError('Unsupported cache ownership namespace')
        if policy.hints or policy.noncoherent or policy.ordering not in ['weak', 'relaxed']:
            raise ValueError('Hierarchy does not yet model hints, noncoherent or strong/volatile accesses: ' + policy.opcode)
        supported = {'load': {'ca', 'cg'}, 'store': {'wb', 'cg'}, 'atomic': {'atomic-l2'}}
        if operation not in supported or policy.cache_operator not in supported[operation]:
            raise ValueError('Unsupported hierarchy cache operator: ' + policy.cache_operator)
        sectors = set(sectors)
        # Validate before counters/state change, including on a bypass path.
        if any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in sectors): raise ValueError('Invalid sector identity')
        if operation == 'atomic' and namespace != 'global': raise ValueError('Private-local atomics are not modeled')
        self.traffic['logical_request_sectors'] += len(sectors)
        use_l1 = policy.cache_operator == 'ca' or (namespace.startswith('local:') and operation == 'store' and policy.cache_operator == 'wb')
        l1_hits, l1_misses, l2_hits, l2_misses = set(), set(), set(), set()
        if use_l1:
            first = self.l1[sm].request(sectors, namespace=namespace, write=operation == 'store')
            l1_hits, l1_misses = first['hits'], first['misses']
            self._writebacks(first['evicted_dirty'], 'l1_eviction')
            second = self._l2(l1_misses, namespace)
        else:
            self.traffic['l1_bypass_sectors'] += len(sectors)
            second = self._l2(sectors, namespace, write=operation != 'load')
        l2_hits, l2_misses = second['hits'], second['misses']
        return {'l1_hits': len(l1_hits), 'l1_misses': len(l1_misses), 'l1_bypass': 0 if use_l1 else len(sectors),
                'l2_hits': len(l2_hits), 'l2_misses': len(l2_misses)}

    def finish(self):
        if self.finished: raise ValueError('Cache hierarchy already drained')
        for cache in self.l1: self._writebacks(cache.drain_dirty(), 'l1_terminal')
        self.traffic['backing_terminal_write_sectors'] += len(self.l2.drain_dirty())
        self.finished = True
        return {'l1': [dict(sorted(c.stats.items())) for c in self.l1],
                'l2': dict(sorted(self.l2.stats.items())), 'traffic': dict(sorted(self.traffic.items()))}

def analyze_cache(run_directory, profile=None):
    profile = validate_profile(profile if profile is not None else DEFAULT_PROFILE)
    transactions = analyze_transactions(run_directory)
    launches = {r['launch']: r for r in map(json.loads, (Path(run_directory) / 'execution-events.jsonl').read_text().splitlines()) if r['type'] == 'launch'}
    # Preflight every modeled request before mutating cache state. No partial
    # report can masquerade as a complete hierarchy when a hint is unsupported.
    requests, models = [], {}
    validation_model = CacheHierarchy(profile)
    for request in transactions['requests']:
        if request['space'] not in ['global', 'local']: continue
        launch = launches[request['launch']]
        x, y, z = request['block']; gx, gy, gz = launch['grid']
        if not (0 <= x < gx and 0 <= y < gy and 0 <= z < gz): raise ValueError('Block outside captured grid')
        sm = (x + gx * (y + gy * z)) % profile['sm_count']
        namespace = 'global' if request['space'] == 'global' else 'local:' + ':'.join(map(str, [request['launch'], x, y, z, request['warp']]))
        policy = decode_policy(request['instruction'], request['operation'])
        # Validation uses a scratch model; the actual launch model starts cold.
        validation_model.request(set(), request['operation'], policy, sm, namespace)
        requests.append((request, policy, sm, namespace))
    records = []
    for request, policy, sm, namespace in requests:
        model = models.setdefault(request['launch'], None)
        if model is None: model = models[request['launch']] = CacheHierarchy(profile)
        result = model.request({int(s, 16) for s in request['sector_ids']}, request['operation'], policy, sm, namespace)
        records.append({'issue': request['issue'], 'launch': request['launch'], 'space': request['space'],
                        'sm': sm, 'ownership': namespace, 'policy': asdict(policy), **result})
    summaries = {str(key): model.finish() for key, model in models.items()}
    totals = Counter()
    for summary in summaries.values(): totals.update(summary['traffic'])
    return {'schema': 1, 'kind': 'synthetic PTX sector-cache traffic replay; not measured NVIDIA behavior',
            'event_sha256': transactions['event_sha256'], 'profile_sha256': hashlib.sha256(json.dumps(profile, sort_keys=True).encode()).hexdigest(),
            'profile': profile, 'hardware_cycles': False, 'complete_supported_memory': True,
            'request_records': records, 'launches': summaries, 'totals': dict(sorted(totals.items())),
            'policy_source': PTX, 'policy_provenance': 'decoded preserved full opcode; performance hints do not change functional consistency',
            'limits': ['captured completion order, no queues or memory-level parallelism', 'synthetic block placement without residency',
                       'requested-sector fills and read-for-ownership write misses are assumptions',
                       'stores/atomics bypass global L1; no data values or L1 coherence modeled',
                       'cold caches per launch; terminal dirty drains are explicit assumptions',
                       'shared/constant/param excluded from cache hierarchy', 'system address provenance, streaming/eviction hints and strong ordering unmodeled',
                       'no calibrated geometry, physical partition mapping or GPU timing']}
