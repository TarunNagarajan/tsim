"""Lane-aware observed PTX register hazards, independent of cycle parameters."""
from collections import Counter
import hashlib
import json
from math import prod
from pathlib import Path
import re

from execution_events import summarize_events, validate_dependency_issue
from collective_events import CollectiveEvents, collective_kind
from memory_model import validate_compilation_artifacts

class CompiledInstructions:
    """Check instruction spelling and start line against verified PTX text."""
    def __init__(self, ptx):
        self.clean = re.sub(r'/\*.*?\*/|//[^\n]*', lambda match: '\n' * match[0].count('\n'), ptx, flags=re.S)
        self.positions = [i for i, char in enumerate(self.clean) if not char.isspace()]
        self.normalized = ''.join(self.clean[i] for i in self.positions)
        self.checked = {}

    def matches(self, line, instruction):
        key = (line, re.sub(r'\s+', '', instruction).rstrip(';') + ';')
        if key in self.checked: return self.checked[key]
        start = 0; matched = False
        while True:
            position = self.normalized.find(key[1], start)
            if position < 0: break
            original = self.positions[position]
            if ((not original or not (self.clean[original - 1].isalnum() or self.clean[original - 1] in '_.$%')) and
                    self.clean.count('\n', 0, original) + 1 == line):
                matched = True; break
            start = position + 1
        self.checked[key] = matched
        return matched

class LaneDependencies:
    """Read-after-write, write-after-read and write-after-write predecessors.

    Register resources are frame-local; CC.CF is per warp and per lane. A
    resource read before any writer becomes an explicit live-in, never a
    fabricated predecessor. This records observed ordering, not GPU latency.
    """
    def __init__(self):
        self.writers = {}
        self.readers = {}
        self.contexts = {}
        self.last_sequence = 0

    def issue(self, event, collective=None, collective_schema=2):
        validate_dependency_issue(event)
        sequence = event['sequence']
        if type(sequence) is not int or sequence <= self.last_sequence:
            raise ValueError('Dependency issues must have increasing sequence identities')
        warp = (event['launch'], tuple(event['block']), event['warp'])
        frame = warp + (event['frame'],)
        layout = (event['function'], event['virtual_registers32'], event['virtual_registers64'], event['declared_local_bytes'])
        if frame in self.contexts and self.contexts[frame] != layout: raise ValueError('Frame aliases different register layouts')
        kind = collective_kind(event['instruction'])
        groups, transfers, waiting = {}, [], False
        if kind:
            if collective is None: raise ValueError('Collective dependencies require executed lane-transfer records')
            validator = CollectiveEvents({sequence: event}, collective_schema)
            previous = sequence
            for child in collective:
                if child.get('issue') != sequence or type(child.get('sequence')) is not int or child['sequence'] <= previous:
                    raise ValueError('Invalid collective child ownership or sequence')
                previous = child['sequence']; validator.record(child)
                if child['type'] == 'collective_group': groups[child['sequence']] = child
                elif child['type'] == 'collective_transfer': transfers.append(child)
                elif child['type'] == 'collective_wait': waiting = True
            validator.finish(sequence, 'completed')
            roles = {(group['data_use'], tuple(group['data_defs'])) for group in groups.values()}
            if len(roles) > 1: raise ValueError('Collective groups disagree on operand roles')
            if any(group['data_use'] >= 0 and not group['data_defs'] for group in groups.values()):
                raise ValueError('Discarded collective results require additional source-read capture')
        elif collective: raise ValueError('Ordinary instruction has collective children')
        self.last_sequence = sequence
        self.contexts[frame] = layout
        reads, writes = {}, {}
        def add(target, resource, mask): target[resource] = target.get(resource, 0) | mask
        def reg(key): return ('register', event['frame'], key)
        execution = 0 if waiting else event['execution_mask']
        uses = list(event['uses'])
        if groups and next(iter(groups.values()))['data_use'] >= 0: uses.pop(0)
        if waiting and collective_schema == 2:
            uses = []
            wait = next(child for child in collective if child['type'] == 'collective_wait')
            if wait['member_use'] >= 0: add(reads, reg(wait['member_use']), wait['member_lane_mask'])
        # Legacy captures do not identify the pre-dispatch membership read.
        # Retain their explicitly conservative selected-operand dependencies.
        for key in uses: add(reads, reg(key), event['execution_mask'])
        for key in event['defs']: add(writes, reg(key), execution)
        if event['predicate_use'] != -1: add(reads, reg(event['predicate_use']), event['path_mask'])
        for key in event['implicit_uses']: add(reads, ('implicit', key), event['execution_mask'])
        for key in event['implicit_defs']: add(writes, ('implicit', key), event['execution_mask'])
        edges, live_ins, undefined = {}, {}, []
        def lanes(mask):
            while mask:
                bit = mask & -mask
                yield bit.bit_length() - 1, bit
                mask ^= bit
        def edge(source, kind, resource, bit, destination=(), destination_mask=0):
            if source == sequence: return
            key = (source, kind, resource, destination, destination_mask)
            edges[key] = edges.get(key, 0) | bit
        def read(resource, mask, destination=(), destination_mask=0):
            for lane, bit in lanes(mask):
                identity = (warp, resource, lane)
                writer = self.writers.get(identity)
                if writer is None:
                    key = (resource, destination, destination_mask)
                    live_ins[key] = live_ins.get(key, 0) | bit
                else: edge(writer, 'RAW', resource, bit, destination, destination_mask)
                self.readers.setdefault(identity, set()).add(sequence)
        for resource, mask in reads.items():
            read(resource, mask)
        # Resolve every input against the old versions before any destination
        # writes, including in-place shuffles and disjoint dispatch groups.
        for transfer in transfers:
            group = groups[transfer['group']]
            for key in group['data_defs']:
                destination = reg(key); bit = 1 << transfer['destination_lane']
                if not transfer['source_mask']:
                    undefined.append({'resource': list(destination), 'lane_mask': bit, 'reason': 'inactive shuffle source'})
                read(reg(group['data_use']), transfer['source_mask'], destination, bit)
        for resource, mask in writes.items():
            for lane, bit in lanes(mask):
                identity = (warp, resource, lane)
                writer = self.writers.get(identity)
                if writer is not None: edge(writer, 'WAW', resource, bit)
                for reader in self.readers.pop(identity, set()): edge(reader, 'WAR', resource, bit)
                self.writers[identity] = sequence
        def route(destination, mask):
            return {'destination_resource': list(destination), 'destination_lane_mask': mask} if destination else {}
        node = {'sequence': sequence, 'launch': event['launch'], 'block': event['block'], 'warp': event['warp'],
                'frame': event['frame'], 'function': event['function'],
                'edges': [{'source': source, 'target': sequence, 'kind': kind, 'resource': list(resource), 'lane_mask': mask}
                          | route(destination, destination_mask)
                          for (source, kind, resource, destination, destination_mask), mask in sorted(edges.items())],
                'live_ins': [{'resource': list(resource), 'lane_mask': mask} | route(destination, destination_mask)
                             for (resource, destination, destination_mask), mask in sorted(live_ins.items())]}
        if kind:
            node.update(collective_kind=kind, collective_wait=waiting, actual_execution_mask=execution,
                        conservative_selection_reads=waiting and collective_schema == 1, undefined_results=undefined)
        return node

def analyze_dependencies(run_directory):
    run = Path(run_directory).resolve()
    metadata = validate_compilation_artifacts(run)
    events_path = run / 'execution-events.jsonl'
    summary = summarize_events(events_path)
    if metadata['exit_code'] or metadata['timed_out'] or not summary['complete_supported_memory']:
        raise ValueError('Dependency graph requires successful complete capture; call ABI and coverage gaps remain unsupported')
    if summary['dependency_schema'] != 2: raise ValueError('Dependency graph requires schema-2 implicit state capture')
    instructions = CompiledInstructions((run / 'kernel.ptx').read_text())
    graph = LaneDependencies(); nodes = []; launches = {}; counts = Counter()
    # Completion order may interleave across CPU workers. Preserve issue order.
    children = {}; events = []
    with events_path.open() as source:
        for row in source:
            event = json.loads(row)
            if event['type'] in ['launch', 'issue']: events.append(event)
            if event['type'].startswith('collective_'): children.setdefault(event['issue'], []).append(event)
    for event in events:
        if event['type'] == 'launch': launches[event['launch']] = event
        if event['type'] != 'issue': continue
        if not instructions.matches(event['ptx_line'], event['instruction']):
            raise ValueError('Dependency instruction differs from compiled PTX spelling or line')
        instruction = re.sub(r'^\s*@\s*!?\s*\S+\s+', '', event['instruction']).strip()
        opcode = instruction.split()[0]
        if opcode.startswith('bar.red') or opcode.split('.')[0] == 'elect':
            raise ValueError('Cross-lane operands require a lane-transfer dependency model')
        if collective_kind(event['instruction']) and not summary['complete_supported_collectives']:
            raise ValueError('Collective dependencies require complete executed lane-transfer capture')
        launch = launches[event['launch']]
        if event['kernel'] != launch['kernel']: raise ValueError('Dependency issue kernel differs from launch')
        block = event['block']
        if not isinstance(block, list) or len(block) != 3 or any(type(n) is not int or n < 0 or n >= cap for n, cap in zip(block, launch['grid'])):
            raise ValueError('Dependency block outside launch grid')
        threads = prod(launch['block']); warps = (threads + 31) // 32
        if event['warp'] >= warps: raise ValueError('Dependency warp outside block')
        valid_lanes = min(32, threads - event['warp'] * 32)
        if event['path_mask'] >= 1 << valid_lanes: raise ValueError('Dependency path has nonexistent block lanes')
        node = graph.issue(event, children.pop(event['sequence'], []), summary['collective_schema'] or 2); nodes.append(node)
        for edge in node['edges']: counts[edge['kind']] += 1
        counts['live_in_lane_reads'] += sum(item['lane_mask'].bit_count() for item in node['live_ins'])
    return {'schema': 2, 'kind': 'observed lane-aware PTX register hazard graph; not a complete GPU scoreboard',
            'event_sha256': hashlib.sha256(events_path.read_bytes()).hexdigest(),
            'ptx_sha256': metadata['compilation_sha256']['kernel.ptx'],
            'counts': dict(counts), 'nodes': nodes, 'complete_scoreboard_dependencies': False,
            'hardware_cycles': False,
            'limits': ['explicit virtual operands plus CC.CF; no physical register mapping',
                       'register layout and frame checks do not prove all opcode operand extraction complete',
                       'call ABI, control transitions, election and block synchronization ordering require additional modeling',
                       'legacy schema-1 parked selection reads are conservative; inactive shuffle values have unknown origins',
                       'memory dependence and latency are separate from register hazards',
                       'live-in lane reads are preserved as unknown origins rather than assumed ready values']}
