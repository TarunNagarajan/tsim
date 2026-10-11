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
from cta_barrier_events import barrier_form

PROGRAMMING = 'https://docs.nvidia.com/cuda/archive/12.6.3/cuda-c-programming-guide/index.html#technical-specifications-per-compute-capability'

def validate_estimated_cta(instruction):
    if barrier_form(instruction) and not re.fullmatch(r'bar(?:rier)?(?:\.cta)?\.(?:sync|arrive)(?:\.aligned)?\s+0(?:\s*,.*)?', instruction.rstrip(';')):
        raise ValueError('Additional named or dynamic block barrier resources are not captured')

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
    # The pinned SDK applies the SM barrier ceiling on Hopper onwards.
    # Zero means this SDK model does not establish an Ampere barrier ceiling.
    barriers: int = 0

def profile_for(gpu):
    if gpu == 'nvidia/a100': return SMProfile(gpu, 164 * 1024, 163 * 1024)
    if gpu == 'nvidia/h100': return SMProfile(gpu, 228 * 1024, 227 * 1024, barriers=64)
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
    barriers: int

def block_footprint(profile, block, registers_per_thread, static_shared=0, dynamic_shared=0, barriers=1):
    static_shared = nonnegative(static_shared, 'static shared')
    if static_shared > 48 * 1024: raise ValueError('Static shared exceeds portable static allocation limit')
    return allocation_footprint(profile, block, registers_per_thread, static_shared, dynamic_shared, barriers)

def allocation_footprint(profile, block, registers_per_thread, shared_prefix=0, dynamic_shared=0, barriers=1):
    """Charge an allocated storage prefix, which may include compiler padding."""
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
    shared_prefix = nonnegative(shared_prefix, 'shared storage prefix')
    dynamic_shared = nonnegative(dynamic_shared, 'dynamic shared')
    nonnegative(barriers, 'block barriers')
    if barriers > 16: raise ValueError('Block barrier allocation exceeds supported limit')
    shared_user = shared_prefix + dynamic_shared
    if shared_user > profile.shared_user_bytes_per_block: raise ValueError('Block shared memory exceeds public per-block limit')
    warps = (threads + profile.warp_size - 1) // profile.warp_size
    per_warp = round_up(registers_per_thread * profile.warp_size, profile.register_quantum_per_warp)
    return BlockFootprint(threads, warps, per_warp, per_warp * warps,
                          per_warp * round_up(warps, profile.partitions),
                          round_up(shared_user + profile.reserved_shared_per_block, profile.shared_quantum), shared_user, barriers)

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
    if profile.barriers and footprint.barriers: ceilings['barriers'] = profile.barriers // footprint.barriers
    blocks = min(ceilings.values())
    return {'blocks_per_sm': blocks, 'warps_per_sm': blocks * footprint.warps,
            'active_threads_per_sm': blocks * footprint.threads,
            'warp_occupancy_fraction': blocks * footprint.warps / profile.warps,
            'active_thread_fraction': blocks * footprint.threads / (profile.warps * profile.warp_size),
            'limits': ceilings, 'limiting_resources': sorted(k for k, n in ceilings.items() if n == blocks)}

def validate_footprint(profile, footprint):
    for key, value in asdict(footprint).items(): nonnegative(value, key)
    canonical = allocation_footprint(profile, [footprint.threads, 1, 1], footprint.registers_per_warp // 32,
                                     dynamic_shared=footprint.shared_user_bytes, barriers=footprint.barriers)
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
        self.used_warps = self.used_shared = self.used_threads = self.used_barriers = 0
        self.live = {}
        self._warp_partitions = {}
        self.next_partition = 0

    def admit(self, identity, footprint):
        validate_footprint(self.profile, footprint)
        if identity in self.live: raise ValueError('Duplicate resident block identity')
        p = self.profile
        if (len(self.live) == p.blocks or self.used_warps + footprint.warps > p.warps or
                self.used_shared + footprint.shared_bytes > p.shared_bytes or
                (p.barriers and self.used_barriers + footprint.barriers > p.barriers) or
                footprint.registers_admission_check > p.registers_per_block): return False
        partition_cap = p.registers // p.partitions
        for shift in range(p.partitions):
            start = (self.next_partition + shift) % p.partitions
            charges = [0] * p.partitions
            for warp in range(footprint.warps): charges[(start + warp) % p.partitions] += footprint.registers_per_warp
            if any(used + needed > partition_cap for used, needed in zip(self.partition_registers, charges)): continue
            self.partition_registers = [used + needed for used, needed in zip(self.partition_registers, charges)]
            self.used_warps += footprint.warps; self.used_shared += footprint.shared_bytes; self.used_threads += footprint.threads
            self.used_barriers += footprint.barriers
            self.live[identity] = (footprint, charges)
            self._warp_partitions[identity] = tuple((start + warp) % p.partitions for warp in range(footprint.warps))
            self.next_partition = (start + footprint.warps) % p.partitions
            return True
        return False

    def release(self, identity):
        if identity not in self.live: raise ValueError('Unknown resident block identity')
        footprint, charges = self.live.pop(identity)
        self._warp_partitions.pop(identity)
        self.partition_registers = [used - charge for used, charge in zip(self.partition_registers, charges)]
        self.used_warps -= footprint.warps; self.used_shared -= footprint.shared_bytes; self.used_threads -= footprint.threads
        self.used_barriers -= footprint.barriers

    def warp_partitions(self, identity):
        if identity not in self.live: raise ValueError('Unknown resident block identity')
        return self._warp_partitions[identity]

    def snapshot(self):
        return {'blocks': len(self.live), 'warps': self.used_warps, 'threads': self.used_threads,
                'shared_bytes': self.used_shared, 'barriers': self.used_barriers,
                'partition_registers': list(self.partition_registers)}

def analyze_residency(run_directory, compiler_directory=None):
    # Require the same checked stream/artifact provenance as other architecture
    # reports; unsupported memory paths are not secretly complete inputs.
    transactions = analyze_transactions(run_directory)
    run = Path(run_directory)
    metadata = json.loads((run / 'run.json').read_text())
    profile = profile_for(metadata['gpu'])
    compiler = None
    if compiler_directory is not None:
        from compiler_resources import load_resources
        compiler = load_resources(compiler_directory, run)
    events = list(map(json.loads, (run / 'execution-events.jsonl').read_text().splitlines()))
    for event in events:
        if event['type'] != 'issue': continue
        instruction = re.sub(r'^\s*@\s*!?\s*%[A-Za-z0-9_$]+\s+', '', event['instruction']).strip()
        if instruction.startswith('setmaxnreg.'):
            raise ValueError('Dynamic register redistribution requires a separate residency model')
        if compiler is None: validate_estimated_cta(instruction)
    records = []
    for event in events:
        if event['type'] != 'launch': continue
        if event['register_allocation_provenance'] != 'PTX liveness estimate, not NVIDIA ptxas':
            raise ValueError('Unknown captured register allocation provenance')
        # Dynamic shared prefix alignment is not in this event schema yet.
        if compiler is None and event['static_shared_bytes'] and event['dynamic_shared_bytes']:
            raise ValueError('Combined static/dynamic shared requires captured aligned storage prefix')
        if compiler is None:
            allocation = None
            footprint = block_footprint(profile, event['block'], event['estimated_registers_per_thread'],
                                        event['static_shared_bytes'], event['dynamic_shared_bytes'])
            provenance = event['register_allocation_provenance']
        else:
            allocation = compiler['resources']['entries'].get(event['kernel'])
            if allocation is None: raise ValueError('Captured kernel missing from compiler resources')
            footprint = allocation_footprint(profile, event['block'], allocation['registers'],
                allocation['shared_prefix_bytes'], event['dynamic_shared_bytes'], allocation['barriers'])
            provenance = 'NVIDIA ptxas allocation cross-checked with cuobjdump'
        grid = event['grid']
        if len(grid) != 3: raise ValueError('Invalid captured grid')
        for n, cap in zip(grid, [(1 << 31) - 1, 65535, 65535]):
            positive_integer(n, 'grid dimension')
            if n > cap: raise ValueError('Grid exceeds public limit')
        records.append({'launch': event['launch'], 'kernel': event['kernel'], 'grid_blocks': prod(grid),
                        'footprint': asdict(footprint), 'register_provenance': provenance,
                        'ptx_estimated_registers_per_thread': event['estimated_registers_per_thread'],
                        'compiler_allocation': allocation,
                        'resource_ceiling': resource_ceiling(profile, footprint)})
    return {'schema': 1, 'kind': 'SM resource ceiling from ' + ('compiler allocations' if compiler else 'PTX estimates') + '; not measured achieved occupancy',
            'event_sha256': transactions['event_sha256'], 'profile': asdict(profile), 'launches': records,
            'hardware_cycles': False, 'physical_register_allocation_verified': compiler is not None,
            'compiler_artifact_sha256': compiler['artifact_sha256'] if compiler else None,
            'provenance': {'capacity_source': PROGRAMMING,
                           'tuning_source': 'https://docs.nvidia.com/cuda/archive/12.6.3/' + ('ampere' if profile.gpu.endswith('a100') else 'hopper') + '-tuning-guide/index.html#occupancy',
                           'allocation_rules': 'CUDA 12.6.3 cuda_occupancy.h standalone SDK reference',
                           'carveout_and_optin': 'maximum shared carveout and permitted dynamic opt-in assumed',
                           'barrier_resource_usage': ('ptxas block barrier count' if compiler else 'one ordinary block barrier assumed') +
                               '; pinned SDK applies the 64-barrier SM ceiling only on Hopper'},
            'limits': ['PTX register liveness is not NVIDIA physical register allocation', 'no dynamic spill traffic inferred from compiler counts',
                       'no SM placement, queues, scheduling or achieved-occupancy simulation',
                       'launch bounds, carveout/opt-in attributes, clusters and dynamic register redistribution unmodeled',
                       'compiler lowering may change the instruction and memory stream relative to PTX',
                       'Ampere SM barrier capacity is not established by this SDK occupancy model'] +
                       ([] if compiler else ['combined static/dynamic shared requires additional aligned-prefix capture'])}
