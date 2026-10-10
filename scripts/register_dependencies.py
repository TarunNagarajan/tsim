"""Lane-aware observed PTX register hazards, independent of cycle parameters."""
from collections import Counter
import hashlib
import json
from math import prod
from pathlib import Path
import re

from execution_events import summarize_events, validate_dependency_issue
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

    def issue(self, event):
        validate_dependency_issue(event)
        sequence = event['sequence']
        if type(sequence) is not int or sequence <= self.last_sequence:
            raise ValueError('Dependency issues must have increasing sequence identities')
        warp = (event['launch'], tuple(event['block']), event['warp'])
        frame = warp + (event['frame'],)
        layout = (event['function'], event['virtual_registers32'], event['virtual_registers64'], event['declared_local_bytes'])
        if frame in self.contexts and self.contexts[frame] != layout: raise ValueError('Frame aliases different register layouts')
        self.last_sequence = sequence
        self.contexts[frame] = layout
        reads, writes = {}, {}
        def add(target, resource, mask): target[resource] = target.get(resource, 0) | mask
        def reg(key): return ('register', event['frame'], key)
        for key in event['uses']: add(reads, reg(key), event['execution_mask'])
        for key in event['defs']: add(writes, reg(key), event['execution_mask'])
        if event['predicate_use'] != -1: add(reads, reg(event['predicate_use']), event['path_mask'])
        for key in event['implicit_uses']: add(reads, ('implicit', key), event['execution_mask'])
        for key in event['implicit_defs']: add(writes, ('implicit', key), event['execution_mask'])
        edges, live_ins = {}, {}
        def lanes(mask):
            while mask:
                bit = mask & -mask
                yield bit.bit_length() - 1, bit
                mask ^= bit
        def edge(source, kind, resource, bit):
            if source == sequence: return
            key = (source, kind, resource)
            edges[key] = edges.get(key, 0) | bit
        for resource, mask in reads.items():
            for lane, bit in lanes(mask):
                identity = (warp, resource, lane)
                writer = self.writers.get(identity)
                if writer is None: live_ins[resource] = live_ins.get(resource, 0) | bit
                else: edge(writer, 'RAW', resource, bit)
                self.readers.setdefault(identity, set()).add(sequence)
        for resource, mask in writes.items():
            for lane, bit in lanes(mask):
                identity = (warp, resource, lane)
                writer = self.writers.get(identity)
                if writer is not None: edge(writer, 'WAW', resource, bit)
                for reader in self.readers.pop(identity, set()): edge(reader, 'WAR', resource, bit)
                self.writers[identity] = sequence
        return {'sequence': sequence, 'launch': event['launch'], 'block': event['block'], 'warp': event['warp'],
                'frame': event['frame'], 'function': event['function'],
                'edges': [{'source': source, 'target': sequence, 'kind': kind, 'resource': list(resource), 'lane_mask': mask}
                          for (source, kind, resource), mask in sorted(edges.items())],
                'live_ins': [{'resource': list(resource), 'lane_mask': mask} for resource, mask in sorted(live_ins.items())]}

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
    with events_path.open() as source:
        for row in source:
            event = json.loads(row)
            if event['type'] == 'launch': launches[event['launch']] = event
            if event['type'] != 'issue': continue
            if not instructions.matches(event['ptx_line'], event['instruction']):
                raise ValueError('Dependency instruction differs from compiled PTX spelling or line')
            instruction = re.sub(r'^\s*@\s*!?\s*\S+\s+', '', event['instruction']).strip()
            opcode = instruction.split()[0]
            if opcode.split('.')[0] in ['shfl', 'vote', 'redux', 'match'] or opcode.startswith('bar.red'):
                raise ValueError('Cross-lane operands require a lane-transfer dependency model')
            launch = launches[event['launch']]
            if event['kernel'] != launch['kernel']: raise ValueError('Dependency issue kernel differs from launch')
            block = event['block']
            if not isinstance(block, list) or len(block) != 3 or any(type(n) is not int or n < 0 or n >= cap for n, cap in zip(block, launch['grid'])):
                raise ValueError('Dependency block outside launch grid')
            threads = prod(launch['block']); warps = (threads + 31) // 32
            if event['warp'] >= warps: raise ValueError('Dependency warp outside block')
            valid_lanes = min(32, threads - event['warp'] * 32)
            if event['path_mask'] >= 1 << valid_lanes: raise ValueError('Dependency path has nonexistent block lanes')
            node = graph.issue(event); nodes.append(node)
            for edge in node['edges']: counts[edge['kind']] += 1
            counts['live_in_lane_reads'] += sum(item['lane_mask'].bit_count() for item in node['live_ins'])
    return {'schema': 1, 'kind': 'observed lane-aware PTX register hazard graph; not a complete GPU scoreboard',
            'event_sha256': hashlib.sha256(events_path.read_bytes()).hexdigest(),
            'ptx_sha256': metadata['compilation_sha256']['kernel.ptx'],
            'counts': dict(counts), 'nodes': nodes, 'complete_scoreboard_dependencies': False,
            'hardware_cycles': False,
            'limits': ['explicit virtual operands plus CC.CF; no physical register mapping',
                       'register layout and frame checks do not prove all opcode operand extraction complete',
                       'call ABI, cross-lane shuffle/reduction transfers and synchronization ordering require additional modeling',
                       'memory dependence and latency are separate from register hazards',
                       'live-in lane reads are preserved as unknown origins rather than assumed ready values']}
