"""Configured SM issue/retirement simulation; parameters are explicit assumptions.

This integrates the SDK-checked admission allocator with observed warp streams,
register hazards and finite execution queues. It does not regenerate SIMT paths,
execute SASS, simulate memory consistency or predict measured NVIDIA cycles.
"""
from collections import Counter, deque
from dataclasses import asdict, dataclass
import hashlib
import heapq
import json
from pathlib import Path
import re

from register_dependencies import analyze_dependencies
from sm_residency import BlockFootprint, SMResidency, analyze_residency, profile_for, validate_footprint, resource_ceiling
from memory_model import positive_integer, analyze_transactions
from cache_hierarchy import decode_policy, PTX as CACHE_POLICY_SOURCE
from sector_memory_service import MemoryConfig, SectorMemoryService

@dataclass(frozen=True)
class Pipeline:
    latency: int
    interval: int
    capacity: int
    scope: str = 'partition'

    def validate(self):
        for name in ['latency', 'interval', 'capacity']: positive_integer(getattr(self, name), name)
        if self.scope not in ['partition', 'sm']: raise ValueError('Unknown pipeline scope')

@dataclass(frozen=True)
class TimingConfig:
    sm_count: int
    pipelines: dict
    provenance: str
    provenance_kind: str = 'assumption'

    def validate(self):
        positive_integer(self.sm_count, 'modeled SM count')
        if self.provenance_kind != 'assumption': raise ValueError('Hardware timing provenance requires measurement validation')
        if not isinstance(self.provenance, str) or not self.provenance.strip(): raise ValueError('Missing timing assumption provenance')
        if not isinstance(self.pipelines, dict) or not self.pipelines: raise ValueError('Missing pipeline definitions')
        for name, pipeline in self.pipelines.items():
            if not isinstance(name, str) or not name or not isinstance(pipeline, Pipeline): raise ValueError('Malformed pipeline definition')
            pipeline.validate()

    def record(self):
        self.validate()
        return {'schema': 1, 'sm_count': self.sm_count, 'provenance': self.provenance,
                'provenance_kind': self.provenance_kind,
                'pipelines': {name: asdict(pipe) for name, pipe in sorted(self.pipelines.items())}}

def default_config():
    return TimingConfig(1, {
        'alu': Pipeline(4, 1, 4), 'fp32': Pipeline(4, 1, 4), 'convert': Pipeline(4, 1, 4),
        'parameter': Pipeline(1, 1, 2), 'memory': Pipeline(24, 1, 8),
        'control': Pipeline(1, 1, 2), 'collective': Pipeline(8, 1, 4),
    }, 'Uncalibrated example: fixed warp-instruction latencies and finite queues; not NVIDIA hardware measurements')

def load_config(path):
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or set(data) != {'schema', 'sm_count', 'pipelines', 'provenance', 'provenance_kind'}:
        raise ValueError('Malformed SM execution configuration')
    if type(data['schema']) is not int or data['schema'] != 1: raise ValueError('Unsupported execution configuration schema')
    if not isinstance(data['pipelines'], dict): raise ValueError('Malformed pipeline table')
    try: pipes = {name: Pipeline(**values) for name, values in data['pipelines'].items()}
    except TypeError as error: raise ValueError('Malformed pipeline fields') from error
    config = TimingConfig(data['sm_count'], pipes, data['provenance'], data['provenance_kind']); config.validate()
    return config

@dataclass(frozen=True)
class SectorRequest:
    operation: str
    sectors: tuple
    bypass_l1: bool = False

    def validate(self, config):
        if self.operation not in ['load', 'store'] or type(self.bypass_l1) is not bool: raise ValueError('Unsupported modeled sector request')
        if not isinstance(self.sectors, tuple) or not self.sectors: raise ValueError('Missing immutable sector request')
        if any(type(s) is not int or s < 0 for s in self.sectors): raise ValueError('Invalid modeled sector identity')
        if len(set(self.sectors)) != len(self.sectors) or len(self.sectors) > config.max_sectors_per_request:
            raise ValueError('Invalid or excessive modeled sector request width')

@dataclass(frozen=True)
class Issue:
    sequence: int
    block: tuple
    warp: int
    pipeline: str
    dependencies: tuple = ()  # (earlier issue sequence, RAW/WAW/WAR)
    control: bool = False
    memory: SectorRequest | None = None

def simulate_launch(profile, footprint, issues, config, origin=0, memory_config=None, pipeline_readiness=None, memory_readiness=None):
    """Discrete-event simulation with reads at issue and writes at retirement.

    RAW and WAW await preceding retirement. WAR awaits preceding operand read
    at issue, permitting a later independent overwrite to retire earlier.
    Every partition issues at most one warp instruction per model cycle.
    Queue capacity counts all in-flight instructions, including latency tails.
    """
    config.validate(); validate_footprint(profile, footprint)
    service = None
    if memory_config is not None:
        if not isinstance(memory_config, MemoryConfig): raise ValueError('Missing timed memory configuration')
        memory_config.validate()
        if memory_config.sm_count != config.sm_count: raise ValueError('Memory service and scheduler SM counts disagree')
        service = SectorMemoryService(memory_config)
        if memory_readiness is not None: service.restore_port_readiness(memory_readiness)
    elif memory_readiness is not None: raise ValueError('Memory port deadlines require a memory service')
    if type(origin) is not int or origin < 0: raise ValueError('Invalid launch cycle origin')
    if not resource_ceiling(profile, footprint)['blocks_per_sm']: raise ValueError('Block cannot reside on an empty modeled SM')
    if not issues: raise ValueError('Empty launch issue stream')
    by_id, streams = {}, {}
    for issue in issues:
        if not isinstance(issue, Issue): raise ValueError('Malformed modeled issue')
        positive_integer(issue.sequence, 'issue sequence')
        if issue.sequence in by_id: raise ValueError('Duplicate modeled issue identity')
        if not isinstance(issue.block, tuple) or len(issue.block) != 3 or any(type(n) is not int or n < 0 for n in issue.block):
            raise ValueError('Invalid modeled block coordinate')
        if type(issue.warp) is not int or not 0 <= issue.warp < footprint.warps: raise ValueError('Warp outside block footprint')
        if issue.pipeline not in config.pipelines or type(issue.control) is not bool: raise ValueError('Unknown pipeline or control flag')
        if issue.memory is not None:
            if service is None or issue.pipeline != 'memory' or issue.control or not isinstance(issue.memory, SectorRequest):
                raise ValueError('Sector request requires a timed memory issue')
            issue.memory.validate(memory_config)
        elif service is not None and issue.pipeline == 'memory': raise ValueError('Timed memory issue lacks conserved address sectors')
        by_id[issue.sequence] = issue
        streams.setdefault((issue.block, issue.warp), []).append(issue)
    blocks = sorted({issue.block for issue in issues}, key=lambda block: (block[2], block[1], block[0]))
    for block in blocks:
        if any((block, warp) not in streams for warp in range(footprint.warps)): raise ValueError('Missing warp issue stream')
    for stream in streams.values(): stream.sort(key=lambda issue: issue.sequence)
    for issue in issues:
        if not isinstance(issue.dependencies, tuple): raise ValueError('Malformed dependency tuple')
        for dependency in issue.dependencies:
            if not isinstance(dependency, tuple) or len(dependency) != 2: raise ValueError('Malformed hazard predecessor')
            source, kind = dependency
            if type(source) is not int or source not in by_id or source >= issue.sequence or kind not in ['RAW', 'WAW', 'WAR']:
                raise ValueError('Unknown, forward or unsupported hazard predecessor')
            previous = by_id[source]
            if (previous.block, previous.warp) != (issue.block, issue.warp): raise ValueError('Register predecessor crosses warp ownership')
    residents = [SMResidency(profile) for _ in range(config.sm_count)]
    queues = [[deque() for _ in range(profile.partitions)] for _ in residents]
    pipeline_states = [{} for _ in residents]
    if pipeline_readiness is not None:
        if not isinstance(pipeline_readiness, list): raise ValueError('Malformed pipeline readiness')
        for row in pipeline_readiness:
            if not isinstance(row, dict) or set(row) != {'sm', 'pipeline', 'partition', 'next_issue_cycle'}:
                raise ValueError('Malformed pipeline readiness record')
            sm, name, partition, deadline = row['sm'], row['pipeline'], row['partition'], row['next_issue_cycle']
            if type(sm) is not int or not 0 <= sm < config.sm_count or name not in config.pipelines or type(deadline) is not int or deadline < 0:
                raise ValueError('Invalid pipeline port readiness')
            expected = range(profile.partitions) if config.pipelines[name].scope == 'partition' else [-1]
            if type(partition) is not int or partition not in expected or (name, partition) in pipeline_states[sm]:
                raise ValueError('Invalid or duplicate pipeline readiness ownership')
            pipeline_states[sm][(name, partition)] = {'next': max(origin, deadline), 'inflight': 0, 'peak': 0}
    peak = [{'blocks': 0, 'warps': 0, 'shared_bytes': 0} for _ in residents]
    areas = [0 for _ in residents]
    positions = {key: 0 for key in streams}; ready_at = {key: origin for key in streams}
    remaining = {block: sum(len(streams[(block, warp)]) for warp in range(footprint.warps)) for block in blocks}
    scheduled, intervals, records, completions = {}, {}, [], []
    pending = deque(blocks); next_sm = 0; partition_start = [0] * config.sm_count
    cycle = origin; completed = 0; counts = Counter()
    while completed < len(issues):
        if service is not None:
            for response in service.advance(cycle):
                sequence = response['identity']; record = scheduled[sequence]
                record['complete_cycle'] = response['complete_cycle']
                pipe = config.pipelines[record['pipeline']]
                key = (record['pipeline'], record['partition'] if pipe.scope == 'partition' else -1)
                heapq.heappush(completions, (response['complete_cycle'], sequence, record['sm'], key, tuple(record['block'])))
        # Retire before this cycle's admission and issue phases.
        while completions and completions[0][0] <= cycle:
            finish, sequence, sm, pipe_key, block = heapq.heappop(completions)
            pipeline_states[sm][pipe_key]['inflight'] -= 1
            remaining[block] -= 1; completed += 1
            if not remaining[block]:
                residents[sm].release(block); intervals[block]['release_cycle'] = finish
        while pending:
            block = pending[0]; accepted = False
            for shift in range(config.sm_count):
                sm = (next_sm + shift) % config.sm_count
                if not residents[sm].admit(block, footprint): continue
                placement = residents[sm].warp_partitions(block)
                intervals[block] = {'block': list(block), 'sm': sm, 'admit_cycle': cycle,
                                    'warp_partitions': list(placement)}
                for warp, partition in enumerate(placement): queues[sm][partition].append((block, warp))
                snapshot = residents[sm].snapshot()
                for resource in peak[sm]: peak[sm][resource] = max(peak[sm][resource], snapshot[resource])
                next_sm = (sm + 1) % config.sm_count; pending.popleft(); accepted = True; break
            if not accepted: break
        wakeups = [completions[0][0]] if completions else []
        for sm in range(config.sm_count):
            last_winner = None
            for offset in range(profile.partitions):
                partition = (partition_start[sm] + offset) % profile.partitions
                queue = queues[sm][partition]
                for _ in range(len(queue)):
                    warp_key = queue.popleft(); issue = streams[warp_key][positions[warp_key]]
                    pipe = config.pipelines[issue.pipeline]
                    pipe_key = (issue.pipeline, partition if pipe.scope == 'partition' else -1)
                    state = pipeline_states[sm].setdefault(pipe_key, {'next': origin, 'inflight': 0, 'peak': 0})
                    available = ready_at[warp_key]; unresolved = False
                    for source, kind in issue.dependencies:
                        predecessor = scheduled.get(source)
                        if predecessor is None: raise ValueError('Predecessor missing from ordered warp issue stream')
                        required = predecessor['issue_cycle'] if kind == 'WAR' else predecessor['complete_cycle']
                        if required is None: unresolved = True
                        else: available = max(available, required)
                    if unresolved or available > cycle:
                        if available > cycle: wakeups.append(available)
                        counts['dependency_or_control_blocked_probes'] += 1
                    elif state['next'] > cycle:
                        wakeups.append(state['next']); counts['pipeline_interval_blocked_probes'] += 1
                    elif state['inflight'] == pipe.capacity:
                        counts['pipeline_capacity_blocked_probes'] += 1
                    elif issue.memory is not None and not service.submit(issue.sequence, sm, issue.memory.sectors,
                                                                         operation=issue.memory.operation, bypass_l1=issue.memory.bypass_l1):
                        wake = service.next_admission(sm)
                        if wake is not None and wake > cycle: wakeups.append(wake)
                        counts['memory_admission_blocked_probes'] += 1
                    else:
                        record = {'sequence': issue.sequence, 'block': list(issue.block), 'warp': issue.warp,
                                  'sm': sm, 'partition': partition, 'pipeline': issue.pipeline,
                                  'issue_cycle': cycle, 'complete_cycle': None if issue.memory else cycle + pipe.latency}
                        if issue.memory: record['memory'] = asdict(issue.memory)
                        records.append(record); scheduled[issue.sequence] = record
                        state['next'] = cycle + pipe.interval; state['inflight'] += 1
                        state['peak'] = max(state['peak'], state['inflight'])
                        if issue.memory is None: heapq.heappush(completions, (record['complete_cycle'], issue.sequence, sm, pipe_key, issue.block))
                        positions[warp_key] += 1
                        ready_at[warp_key] = cycle + (pipe.latency if issue.control else 1)
                        if positions[warp_key] < len(streams[warp_key]): queue.append(warp_key)
                        wakeups.append(cycle + 1); last_winner = partition; counts['issued'] += 1
                        break
                    queue.append(warp_key)
            if last_winner is not None: partition_start[sm] = (last_winner + 1) % profile.partitions
        if completed == len(issues): break
        if service is not None:
            wake = service.next_wakeup()
            if wake is not None: wakeups.append(wake)
        future = [time for time in wakeups if time > cycle]
        if not future: raise ValueError('Modeled execution cannot make forward progress')
        following = min(future)
        for sm, resident in enumerate(residents): areas[sm] += resident.used_warps * (following - cycle)
        cycle = following
    if pending or completions or any(resident.live for resident in residents): raise AssertionError('Execution resource conservation failed')
    instruction_end = cycle; memory_report = None
    if service is not None:
        if service.requests: raise AssertionError('Instruction completion preceded memory response')
        service.finish(); cycle = max(cycle, service.cycle)
        memory_report = {'config': memory_config.record(), 'stats': dict(service.stats), 'peaks': dict(service.peak),
                         'requests': list(service.finished.values()), 'dram_reads': service.dram_records,
                         'writebacks': service.writeback_records, 'store_commits': service.commit_records,
                         'resource_records': service.resource_records, 'lookup_records': service.lookup_records,
                         'initial_port_readiness': memory_readiness, 'next_port_readiness': service.port_readiness(),
                         'final_snapshot': service.snapshot(), 'closed': service.closed}
    duration = cycle - origin
    return {'start_cycle': origin, 'end_cycle': cycle, 'model_cycles': duration, 'issues': records,
            'instruction_end_cycle': instruction_end, 'memory_service': memory_report,
            'initial_pipeline_readiness': pipeline_readiness,
            'next_pipeline_readiness': [{'sm': sm, 'pipeline': name, 'partition': partition, 'next_issue_cycle': state['next']}
                                        for sm, states in enumerate(pipeline_states) for (name, partition), state in sorted(states.items())],
            'blocks': [intervals[block] for block in blocks], 'counts': dict(counts),
            'peak_resident_per_sm': peak, 'resident_warp_cycle_integral': areas,
            'modeled_resident_warp_fraction': sum(areas) / (duration * profile.warps * config.sm_count),
            'pipeline_peaks': [{'sm': sm, 'pipeline': name, 'partition': partition, 'inflight_peak': state['peak']}
                               for sm, states in enumerate(pipeline_states) for (name, partition), state in sorted(states.items())],
            'final_residency': [resident.snapshot() for resident in residents]}

def classify_issue(event, node):
    instruction = re.sub(r'^\s*@\s*!?\s*\S+\s+', '', event['instruction']).strip()
    opcode = instruction.split()[0]; base = opcode.split('.')[0]
    if base not in ['mov', 'add', 'sub', 'addc', 'subc', 'mad', 'madc', 'mul', 'and', 'or', 'xor', 'not',
                    'shl', 'shr', 'setp', 'selp', 'cvt', 'cvta', 'fns', 'bra', 'ret', 'ld', 'st',
                    'shfl', 'vote', 'redux', 'match', 'fma']:
        raise ValueError('No declared execution pipeline for ' + opcode)
    if node.get('conservative_selection_reads'): raise ValueError('Execution requires exact schema-2 parked reads')
    if node['live_ins'] or node.get('undefined_results'): raise ValueError('Execution requires known register origins and defined collective sources')
    if node.get('collective_wait') or not event['execution_mask']: return 'control', True
    if base in ['bra', 'ret']: return 'control', True
    if base in ['shfl', 'vote', 'redux', 'match']: return 'collective', False
    if base in ['ld', 'st']:
        if any(part in opcode.split('.') for part in ['volatile', 'acquire', 'release', 'relaxed', 'sc']):
            raise ValueError('Memory ordering requires a declared service/readiness model')
        return ('parameter' if '.param.' in opcode else 'memory'), False
    if base == 'cvt': return 'convert', False
    if any(re.fullmatch(r'(?:f|bf)(?:16|64)(?:x2)?', part) for part in opcode.split('.')):
        raise ValueError('Floating type requires an additional declared pipeline class')
    return ('fp32' if 'f32' in opcode.split('.') else 'alu'), False

def analyze_execution(run_directory, config=None, compiler_directory=None, memory_config=None):
    config = config or default_config(); config.validate()
    if memory_config is not None:
        if not isinstance(memory_config, MemoryConfig): raise ValueError('Missing timed memory configuration')
        memory_config.validate()
        if memory_config.sm_count != config.sm_count: raise ValueError('Memory service and scheduler SM counts disagree')
    run = Path(run_directory).resolve()
    graph = analyze_dependencies(run); residency = analyze_residency(run, compiler_directory)
    if graph['event_sha256'] != residency['event_sha256']: raise ValueError('Architecture inputs refer to different event streams')
    event_bytes = (run / 'execution-events.jsonl').read_bytes()
    if hashlib.sha256(event_bytes).hexdigest() != graph['event_sha256']: raise ValueError('Execution stream changed during analysis')
    events = [json.loads(row) for row in event_bytes.decode().splitlines()]
    launches, raw, active = {}, {}, None
    for event in events:
        if event['type'] == 'launch':
            if active is not None: raise ValueError('Overlapping launches require stream readiness capture')
            active = event['launch']; launches[active] = event
        elif event['type'] == 'launch_end': active = None
        elif event['type'] == 'issue':
            if event['launch'] != active: raise ValueError('Issue outside sequential launch envelope')
            raw[event['sequence']] = event
    requests, policies = {}, []
    if memory_config is not None:
        transactions = analyze_transactions(run)
        if transactions['event_sha256'] != graph['event_sha256']: raise ValueError('Timed addresses refer to a different event stream')
        for record in transactions['requests']:
            if record['space'] == 'param': continue
            if record['space'] != 'global' or record['operation'] not in ['load', 'store']:
                raise ValueError('Timed sector service requires global load/store requests')
            sequence = record['issue']
            if sequence in requests: raise ValueError('Timed sector service requires one conserved memory group per issue')
            policy = decode_policy(record['instruction'], record['operation'])
            supported = {'load': ['ca', 'cg'], 'store': ['wb', 'cg']}
            if policy.hints or policy.noncoherent or policy.ordering != 'weak' or policy.scope or policy.cache_operator not in supported[record['operation']]:
                raise ValueError('Unsupported timed cache policy: ' + policy.opcode)
            sectors = tuple(int(s, 16) for s in record['sector_ids'])
            request = SectorRequest(record['operation'], sectors, policy.cache_operator != 'ca')
            request.validate(memory_config); requests[sequence] = request
            policies.append({'issue': sequence, 'launch': record['launch'], 'policy': asdict(policy), 'sectors': len(sectors)})
    plans = {identity: [] for identity in launches}; live = {}
    for node in graph['nodes']:
        event = raw[node['sequence']]; launch = launches[event['launch']]
        if event['frame'] != 0 or event['function'] != event['kernel']: raise ValueError('Frame/control transfers require additional readiness semantics')
        pipeline, control = classify_issue(event, node)
        key = (event['launch'], tuple(event['block']), event['warp'])
        if key not in live:
            threads = launch['block'][0] * launch['block'][1] * launch['block'][2]
            live[key] = (1 << min(32, threads - 32 * event['warp'])) - 1
        if event['path_mask'] & ~live[key]: raise ValueError('Issue includes previously retired kernel lanes')
        instruction = re.sub(r'^\s*@\s*!?\s*\S+\s+', '', event['instruction']).strip()
        if instruction.split()[0].split('.')[0] == 'ret': live[key] &= ~event['execution_mask']
        dependencies = tuple(sorted({(edge['source'], edge['kind']) for edge in node['edges']}))
        request = requests.get(event['sequence'])
        if memory_config is not None and (pipeline == 'memory') != (request is not None):
            raise ValueError('Timed memory issue disagrees with conserved physical-address capture')
        plans[event['launch']].append(Issue(event['sequence'], tuple(event['block']), event['warp'], pipeline, dependencies, control, request))
    if any(live.values()): raise ValueError('Captured warp does not retire every kernel lane')
    results = []; origin = 0; profile = profile_for(residency['profile']['gpu']); pipeline_readiness = memory_readiness = None
    for record in residency['launches']:
        plan = plans[record['launch']]
        if len({issue.block for issue in plan}) != record['grid_blocks']: raise ValueError('Missing block issue stream')
        footprint = BlockFootprint(**record['footprint'])
        result = simulate_launch(profile, footprint, plan, config, origin, memory_config, pipeline_readiness, memory_readiness)
        result.update(launch=record['launch'], kernel=record['kernel'], register_provenance=record['register_provenance'], footprint=record['footprint'],
                      ptx_estimated_registers_per_thread=record['ptx_estimated_registers_per_thread'], compiler_allocation=record['compiler_allocation'])
        results.append(result); origin = result['end_cycle']
        pipeline_readiness = result['next_pipeline_readiness']
        memory_readiness = result['memory_service']['next_port_readiness'] if memory_config else None
    return {'schema': 1, 'kind': 'configured SM execution replay of observed PTX paths; uncalibrated model cycles',
            'event_sha256': graph['event_sha256'], 'ptx_sha256': graph['ptx_sha256'], 'config': config.record(),
            'profile': residency['profile'], 'model_cycles': origin, 'launches': results,
            'memory_config': memory_config.record() if memory_config else None, 'memory_policies': policies,
            'memory_policy_source': CACHE_POLICY_SOURCE if memory_config else None,
            'physical_register_allocation_verified': residency['physical_register_allocation_verified'],
            'compiler_artifact_sha256': residency['compiler_artifact_sha256'],
            'hardware_cycles': False, 'complete_scoreboard_dependencies': False, 'calibrated': False,
            'assumptions': ['recorded per-warp SIMT path order; control operations retire before the next selected path issue',
                            'one issue per partition per model cycle, round-robin ready warps and rotating partition sweep',
                            'SDK-checked capacities with synthetic rotating balanced warp placement',
                            'operands consumed at issue, results written at completion; RAW/WAW wait completion and WAR waits issue',
                            'configured nonmemory warp-instruction latencies, initiation intervals and bounded in-flight queues',
                            'sequential launches; block resources released only after every in-flight instruction drains'] +
                           ['pipeline and memory port readiness persists across launch boundaries'] +
                           (['actual sector requests use asynchronous private-L1/shared-L2/backing queues with explicit synthetic geometry and ports',
                             'requested-sector fills, read-for-ownership store misses, clean/dirty LRU replacement and FIFO backing service',
                             'global stores bypass L1; no L1 data coherence or weak-memory outcomes are inferred',
                             'cold caches per launch; dirty evictions and terminal drains retain backing reservations until completion'] if memory_config else
                            ['memory instructions use fixed configured latency tokens']),
            'limits': ['no regeneration of SIMT paths, complete rendezvous/control readiness or independent-thread scheduling',
                       ('timed service supports weak global ca/cg loads and wb/cg stores; data/coherence/ordering, atomics and other spaces remain open' if memory_config else
                        'memory latency is a configured instruction token; enable an explicit memory configuration for staged global load/store service'),
                       'virtual dependencies do not establish complete opcode extraction or physical-register/SASS dependencies',
                       'compiler allocation affects admission only; generated cubins are not executed',
                       'model cycles and resident-warp fractions are not NVIDIA measurements']}
