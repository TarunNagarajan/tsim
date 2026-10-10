"""Resource ceilings and an explicit synthetic SM admission allocator.

Input register counts are allocation slots; PTX liveness estimates are not
physical NVIDIA allocations. No SM issue scheduling or GPU cycles are claimed.
"""
from dataclasses import asdict, dataclass
import json
from math import prod
from pathlib import Path
import re

from memory_model import analyze_transactions, positive_integer

PROGRAMMING = 'https://docs.nvidia.com/cuda/archive/12.6.3/cuda-c-programming-guide/index.html#technical-specifications-per-compute-capability'

def nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0: raise ValueError(name + ' must be a nonnegative integer')
    return value

def round_up(value, quantum):
    return (value + quantum - 1) // quantum * quantum

@dataclass(frozen=True)
class SMProfile:
    gpu: str
    shared_bytes: int
    shared_user_bytes_per_block: int
    warps: int = 64
    blocks: int = 32
    registers: int = 65536
    registers_per_block: int = 65536
    threads_per_block: int = 1024
    warp_size: int = 32
    partitions: int = 4
    register_quantum_per_warp: int = 256
    shared_quantum: int = 128
    reserved_shared_per_block: int = 1024

def profile_for(gpu):
    if gpu == 'nvidia/a100': return SMProfile(gpu, 164 * 1024, 163 * 1024)
    if gpu == 'nvidia/h100': return SMProfile(gpu, 228 * 1024, 227 * 1024)
    raise ValueError('SM residency currently supports A100/H100 profiles')

@dataclass(frozen=True)
class BlockFootprint:
    threads: int
    warps: int
    registers_per_warp: int
    registers: int
    registers_admission_check: int
    shared_bytes: int
    shared_user_bytes: int

def block_footprint(profile, block, registers_per_thread, static_shared=0, dynamic_shared=0):
    # Profiles are published constraints and fixed SDK rules, not arbitrary
    # capacities with an A100/H100 name attached.
    if profile != profile_for(profile.gpu): raise ValueError('Altered SM profile requires separate provenance')
    if not isinstance(block, (list, tuple)) or len(block) != 3: raise ValueError('Block must have three dimensions')
    for n, cap in zip(block, [1024, 1024, 64]):
        positive_integer(n, 'block dimension')
        if n > cap: raise ValueError('Block dimension exceeds public limit')
    threads = prod(block)
    if threads > profile.threads_per_block: raise ValueError('Block exceeds maximum threads')
    nonnegative(registers_per_thread, 'register allocation slots per thread')
    # The CUDA 12.6 occupancy SDK permits 256 allocated slots on SM8/9,
    # whereas the ISA exposes 255 logical registers. Do not invent spills.
    if registers_per_thread > 256: raise ValueError('PTX estimate requires a physical allocation/spill result')
    static_shared = nonnegative(static_shared, 'static shared')
    dynamic_shared = nonnegative(dynamic_shared, 'dynamic shared')
    if static_shared > 48 * 1024: raise ValueError('Static shared exceeds portable static allocation limit')
    shared_user = static_shared + dynamic_shared
    if shared_user > profile.shared_user_bytes_per_block: raise ValueError('Block shared memory exceeds public per-block limit')
    warps = (threads + profile.warp_size - 1) // profile.warp_size
    per_warp = round_up(registers_per_thread * profile.warp_size, profile.register_quantum_per_warp)
    return BlockFootprint(threads, warps, per_warp, per_warp * warps,
                          per_warp * round_up(warps, profile.partitions),
                          round_up(shared_user + profile.reserved_shared_per_block, profile.shared_quantum), shared_user)

def resource_ceiling(profile, footprint):
    validate_footprint(profile, footprint)
    if footprint.registers_admission_check > profile.registers_per_block:
        register_limit = 0
    elif footprint.registers_per_warp:
        partition_warps = (profile.registers // profile.partitions) // footprint.registers_per_warp
        register_limit = partition_warps * profile.partitions // footprint.warps
    else: register_limit = profile.blocks
    ceilings = {'warps': profile.warps // footprint.warps, 'blocks': profile.blocks,
                'registers': register_limit, 'shared': profile.shared_bytes // footprint.shared_bytes}
    blocks = min(ceilings.values())
    return {'blocks_per_sm': blocks, 'warps_per_sm': blocks * footprint.warps,
            'active_threads_per_sm': blocks * footprint.threads,
            'warp_occupancy_fraction': blocks * footprint.warps / profile.warps,
            'active_thread_fraction': blocks * footprint.threads / (profile.warps * profile.warp_size),
            'limits': ceilings, 'limiting_resources': sorted(k for k, n in ceilings.items() if n == blocks)}

def validate_footprint(profile, footprint):
    for key, value in asdict(footprint).items(): nonnegative(value, key)
    canonical = block_footprint(profile, [footprint.threads, 1, 1], footprint.registers_per_warp // 32,
                                dynamic_shared=footprint.shared_user_bytes)
    if canonical != footprint: raise ValueError('Inconsistent block footprint')

class SMResidency:
    """Atomic admission/release with a declared balanced-warp placement policy.

    Warp register placement rotates over four partitions; this is a synthetic
    scheduling choice. Exact hardware placement and fragmentation are unknown.
    """
    def __init__(self, profile):
        if profile != profile_for(profile.gpu): raise ValueError('Altered SM profile requires separate provenance')
        self.profile = profile
        self.partition_registers = [0] * profile.partitions
        self.used_warps = self.used_shared = self.used_threads = 0
        self.live = {}
        self.next_partition = 0

    def admit(self, identity, footprint):
        validate_footprint(self.profile, footprint)
        if identity in self.live: raise ValueError('Duplicate resident block identity')
        p = self.profile
        if (len(self.live) == p.blocks or self.used_warps + footprint.warps > p.warps or
                self.used_shared + footprint.shared_bytes > p.shared_bytes or
                footprint.registers_admission_check > p.registers_per_block): return False
        partition_cap = p.registers // p.partitions
        for shift in range(p.partitions):
            start = (self.next_partition + shift) % p.partitions
            charges = [0] * p.partitions
            for warp in range(footprint.warps): charges[(start + warp) % p.partitions] += footprint.registers_per_warp
            if any(used + needed > partition_cap for used, needed in zip(self.partition_registers, charges)): continue
            self.partition_registers = [used + needed for used, needed in zip(self.partition_registers, charges)]
            self.used_warps += footprint.warps; self.used_shared += footprint.shared_bytes; self.used_threads += footprint.threads
            self.live[identity] = (footprint, charges)
            self.next_partition = (start + footprint.warps) % p.partitions
            return True
        return False

    def release(self, identity):
        if identity not in self.live: raise ValueError('Unknown resident block identity')
        footprint, charges = self.live.pop(identity)
        self.partition_registers = [used - charge for used, charge in zip(self.partition_registers, charges)]
        self.used_warps -= footprint.warps; self.used_shared -= footprint.shared_bytes; self.used_threads -= footprint.threads

    def snapshot(self):
        return {'blocks': len(self.live), 'warps': self.used_warps, 'threads': self.used_threads,
                'shared_bytes': self.used_shared, 'partition_registers': list(self.partition_registers)}

def analyze_residency(run_directory):
    # Require the same checked stream/artifact provenance as other architecture
    # reports; unsupported memory paths are not secretly complete inputs.
    transactions = analyze_transactions(run_directory)
    run = Path(run_directory)
    metadata = json.loads((run / 'run.json').read_text())
    profile = profile_for(metadata['gpu'])
    events = list(map(json.loads, (run / 'execution-events.jsonl').read_text().splitlines()))
    for event in events:
        if event['type'] != 'issue': continue
        instruction = re.sub(r'^\s*@\s*!?\s*%[A-Za-z0-9_$]+\s+', '', event['instruction']).strip()
        if instruction.startswith('setmaxnreg.'):
            raise ValueError('Dynamic register redistribution requires a separate residency model')
        if instruction.startswith('bar.') and not instruction.startswith('bar.warp.'):
            if not re.fullmatch(r'bar\.(?:sync|arrive)(?:\.aligned)?\s+0(?:\s*,.*)?', instruction):
                raise ValueError('Additional named or dynamic block barrier resources are not captured')
    records = []
    for event in events:
        if event['type'] != 'launch': continue
        if event['register_allocation_provenance'] != 'PTX liveness estimate, not NVIDIA ptxas':
            raise ValueError('Unknown captured register allocation provenance')
        # Dynamic shared prefix alignment is not in this event schema yet.
        if event['static_shared_bytes'] and event['dynamic_shared_bytes']:
            raise ValueError('Combined static/dynamic shared requires captured aligned storage prefix')
        footprint = block_footprint(profile, event['block'], event['estimated_registers_per_thread'],
                                    event['static_shared_bytes'], event['dynamic_shared_bytes'])
        grid = event['grid']
        if len(grid) != 3: raise ValueError('Invalid captured grid')
        for n, cap in zip(grid, [(1 << 31) - 1, 65535, 65535]):
            positive_integer(n, 'grid dimension')
            if n > cap: raise ValueError('Grid exceeds public limit')
        records.append({'launch': event['launch'], 'kernel': event['kernel'], 'grid_blocks': prod(grid),
                        'footprint': asdict(footprint), 'register_provenance': event['register_allocation_provenance'],
                        'resource_ceiling': resource_ceiling(profile, footprint)})
    return {'schema': 1, 'kind': 'SM resource ceiling from PTX estimates; not measured achieved occupancy',
            'event_sha256': transactions['event_sha256'], 'profile': asdict(profile), 'launches': records,
            'hardware_cycles': False, 'physical_register_allocation_verified': False,
            'provenance': {'capacity_source': PROGRAMMING,
                           'tuning_source': 'https://docs.nvidia.com/cuda/archive/12.6.3/' + ('ampere' if profile.gpu.endswith('a100') else 'hopper') + '-tuning-guide/index.html#occupancy',
                           'allocation_rules': 'CUDA 12.6.3 cuda_occupancy.h standalone SDK reference',
                           'carveout_and_optin': 'maximum shared carveout and permitted dynamic opt-in assumed',
                           'barrier_resource_usage': 'one ordinary block barrier assumed; additional named barriers unmodeled'},
            'limits': ['PTX register liveness is not NVIDIA physical register allocation', 'no invented spill allocation',
                       'no SM placement, queues, scheduling or achieved-occupancy simulation',
                       'launch bounds, carveout/opt-in attributes, clusters and dynamic register redistribution unmodeled',
                       'combined static/dynamic shared requires additional aligned-prefix capture']}
