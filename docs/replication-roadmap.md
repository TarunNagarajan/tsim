# NVIDIA replication program

The goal is a progressively comprehensive NVIDIA CUDA execution and GPU architecture simulator on a CPU. A passing demonstration is a milestone, not completion of this program. Original execution code remains attributed to the pinned PantheonSim engine; tsim's changes are independently tracked and verified.

The initial functional baseline compiles CUDA 12.6.3 source into compute_80 PTX, runs 16 integration checks, and records actual interpreter warp state. It has no hardware timing model. Upstream capability claims are leads for verification, not automatically claims about this build.

The implementation tracks are:

| Track | Required behavior | Evidence required |
| --- | --- | --- |
| CUDA/PTX conformance | Typed arithmetic, conversions, calls, control flow, memory spaces, collectives, atomics and supported API behavior | Independent references, boundary cases, schedule variants, explicit failures for unsupported behavior |
| SIMT and synchronization | Per-thread control flow and state, warp participation, block/cluster/grid barriers and forward progress | Valid divergent synchronization tests; invalid participation diagnostics; adversarial schedules |
| Memory consistency | Scoped acquire/release, fences, atomicity, ordered communication, memory lifetime and access checks | PTX memory-model litmus tests and race diagnostics; documented legal outcomes |
| Architecture events | Actual per-lane memory addresses and widths, register dependencies, resource requirements, block/warp identity | Trace alignment and conservation invariants; checks against interpreter counters |
| SM model | Block placement, residency, warp schedulers, scoreboards, execution pipelines and contention | Small analytically solvable workloads and published architectural constraints |
| Memory hierarchy | Coalescing, sectors, shared-memory banks, configurable L1/L2, misses and memory queues | Address-pattern microbenchmarks with exact transaction references |
| Timing and calibration | Explicit latency/throughput parameters, cycle simulation and calibration provenance | Hardware measurements where available; predicted-versus-measured error distributions |
| Debugging and differential validation | State inspection, repeatable replay, failure minimization, hardware comparison | Reproducible artifacts; identical functional checks on simulator and physical GPUs |
| Compatibility inventory | Versioned supported/tested/unverified/unsupported states | Machine-readable manifests linked to executed checks; no inferred blanket compatibility score |

PTX behavior follows the [PTX ISA 8.5 specification](https://docs.nvidia.com/cuda/archive/12.6.3/parallel-thread-execution/index.html) corresponding to the local compiler, with later ISA versions tracked separately. Architecture constraints begin with the [Ampere tuning guide](https://docs.nvidia.com/cuda/archive/12.6.3/ampere-tuning-guide/index.html). Parameters from public specifications, measured hardware, and model assumptions must carry separate provenance. A modeled cycle is not automatically a measured NVIDIA cycle.

The expanded functional baseline now passes 343 upstream core checks and 60 CUDA source cases across two profiles and three schedules. Shared-memory publication and counted barriers have a warp-level happens-before detector with explicit limits. The compatibility inventory ties each tested subset to executed checks. Architecture modeling next follows actual addresses, typed register dependencies and resource events captured during execution. An instruction name alone cannot establish cache activity.

The broad goal stays active while these tracks remain incomplete. Exact equivalence to undisclosed proprietary microarchitecture cannot be established solely from public documentation. Hardware differential results are currently unavailable on this CPU-only machine and must be reported as unverified.
