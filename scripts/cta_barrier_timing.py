"""Finite CTA release service over captured phase causes and memory prefixes."""
from dataclasses import dataclass
import heapq

@dataclass(frozen=True)
class Phase:
    identity: int
    block: tuple
    barrier: int
    generation: int
    arrivals: tuple
    exits: tuple = ()
    waiters: tuple = ()  # (blocking arrival issue, next selected issue)

class BarrierService:
    def __init__(self, phases, issues, streams, config, origin, readiness=None):
        if not isinstance(phases, tuple): raise ValueError('CTA phases require an immutable plan')
        if phases and config is None: raise ValueError('CTA phases require declared release service assumptions')
        if config is not None:
            config.validate()
            if config.scope != 'sm': raise ValueError('CTA release service must have SM scope')
        self.config, self.origin, self.initial = config, origin, readiness
        self.plans, self.gates, self.records, self.heap = {}, {}, {}, []
        self.ports, self.active, self.peaks = {}, {}, {}
        if readiness is not None:
            if not isinstance(readiness, list): raise ValueError('Malformed CTA port readiness')
            for row in readiness:
                if not isinstance(row, dict) or set(row) != {'sm', 'next_release_cycle'}:
                    raise ValueError('Malformed CTA port readiness record')
                sm, deadline = row['sm'], row['next_release_cycle']
                if type(sm) is not int or sm < 0 or sm in self.ports or type(deadline) is not int or deadline < 0:
                    raise ValueError('Invalid CTA port readiness')
                self.ports[sm] = deadline
        previous, arrival_owners = {}, set()
        for phase in phases:
            if not isinstance(phase, Phase) or type(phase.identity) is not int or phase.identity <= 0 or phase.identity in issues or phase.identity in self.plans:
                raise ValueError('Invalid or repeated CTA phase identity')
            if type(phase.barrier) is not int or not 0 <= phase.barrier < 16 or type(phase.generation) is not int or phase.generation < 0:
                raise ValueError('Invalid CTA resource or generation')
            if not isinstance(phase.block, tuple) or len(phase.block) != 3 or any(type(v) is not int or v < 0 for v in phase.block):
                raise ValueError('Invalid CTA block identity')
            if not isinstance(phase.arrivals, tuple) or not phase.arrivals or not isinstance(phase.exits, tuple) or not isinstance(phase.waiters, tuple):
                raise ValueError('Malformed CTA phase causes')
            causes = phase.arrivals + phase.exits
            if len(set(causes)) != len(causes): raise ValueError('Repeated CTA cause')
            warps, memory = set(), set()
            for source in causes:
                if type(source) is not int or source not in issues: raise ValueError('Missing CTA cause instruction')
                issue = issues[source]
                if issue.block != phase.block or issue.warp in warps or not issue.control or source >= phase.identity:
                    raise ValueError('CTA causes must be earlier distinct converged warp controls in the same block')
                warps.add(issue.warp)
                for prior in streams[(issue.block, issue.warp)]:
                    if prior.sequence >= source: break
                    if prior.pipeline in ['memory', 'parameter']: memory.add(prior.sequence)
            if arrival_owners.intersection(phase.arrivals): raise ValueError('CTA arrival borrowed by another phase')
            arrival_owners.update(phase.arrivals)
            key = (phase.block, phase.barrier)
            prior = previous.get(key)
            if phase.generation != (self.plans[prior]['phase'].generation + 1 if prior else 0):
                raise ValueError('CTA generation skipped or repeated')
            previous[key] = phase.identity
            waiter_sources = set()
            for pair in phase.waiters:
                if not isinstance(pair, tuple) or len(pair) != 2: raise ValueError('Malformed CTA waiter gate')
                source, target = pair
                if source not in phase.arrivals or source in waiter_sources or type(target) is not int or target not in issues or target <= phase.identity:
                    raise ValueError('CTA waiter lacks a unique arrival and subsequent issue')
                a, b = issues[source], issues[target]
                stream = streams[(a.block, a.warp)]
                position = next(i for i, entry in enumerate(stream) if entry.sequence == source)
                if (a.block, a.warp) != (b.block, b.warp) or position + 1 >= len(stream) or stream[position+1].sequence != target:
                    raise ValueError('CTA waiter gate is not its next selected instruction')
                waiter_sources.add(source)
                self.gates.setdefault(target, set()).add(phase.identity)
            if prior:
                for source in phase.arrivals: self.gates.setdefault(source, set()).add(prior)
            self.plans[phase.identity] = {'phase': phase, 'memory': sorted(memory), 'prior': prior,
                'causes': sorted(set(causes) | memory)}

    def gate(self, sequence):
        required = self.gates.get(sequence, ())
        if any(identity not in self.records for identity in required): return None
        return max((self.records[identity]['complete_cycle'] for identity in required), default=self.origin)

    def advance(self, cycle, scheduled, placements):
        retired = []
        while self.heap and self.heap[0][0] <= cycle:
            end, identity, sm = heapq.heappop(self.heap)
            self.active[sm] -= 1
            retired.append((self.plans[identity]['phase'].block, end))
        for identity, plan in self.plans.items():
            if identity in self.records: continue
            phase = plan['phase']
            if phase.block not in placements: continue
            if any(source not in scheduled or scheduled[source]['complete_cycle'] is None for source in plan['causes']): continue
            prior = plan['prior']
            if prior is not None and prior not in self.records: continue
            ready = max([self.origin] + [scheduled[source]['complete_cycle'] for source in plan['causes']] +
                        ([self.records[prior]['complete_cycle']] if prior else []))
            sm = placements[phase.block]['sm']
            if ready > cycle or self.ports.get(sm, self.origin) > cycle or self.active.get(sm, 0) >= self.config.capacity: continue
            end = cycle + self.config.latency
            self.records[identity] = {'identity': identity, 'block': list(phase.block), 'barrier': phase.barrier,
                'phase': phase.generation, 'arrival_issues': list(phase.arrivals), 'exit_issues': list(phase.exits),
                'waiter_gates': [list(pair) for pair in phase.waiters], 'memory_prior_issues': plan['memory'],
                'previous_phase': prior, 'sm': sm, 'ready_cycle': ready, 'start_cycle': cycle, 'complete_cycle': end}
            self.ports[sm] = cycle + self.config.interval
            self.active[sm] = self.active.get(sm, 0) + 1
            self.peaks[sm] = max(self.peaks.get(sm, 0), self.active[sm])
            heapq.heappush(self.heap, (end, identity, sm))
        return retired

    def wakeups(self, cycle, scheduled, placements):
        result = [self.heap[0][0]] if self.heap else []
        for identity, plan in self.plans.items():
            if identity in self.records or plan['phase'].block not in placements: continue
            result.extend(scheduled[source]['complete_cycle'] for source in plan['causes']
                          if source in scheduled and scheduled[source]['complete_cycle'] is not None)
            result.append(self.ports.get(placements[plan['phase'].block]['sm'], self.origin))
        return [value for value in result if value > cycle]

    def report(self):
        if self.heap or len(self.records) != len(self.plans) or any(self.active.values()):
            raise AssertionError('CTA release service has not drained')
        return {'phases': list(self.records.values()), 'initial_port_readiness': self.initial,
            'next_port_readiness': [{'sm': sm, 'next_release_cycle': time} for sm, time in sorted(self.ports.items())],
            'peak_inflight_per_sm': dict(self.peaks), 'final_inflight_per_sm': dict(self.active)}

def captured_phases(graph, launch):
    gates = {gate['issue']: gate['release'] for gate in graph['cta_control_gates']}
    target_by_release_warp = {}
    nodes = {node['sequence']: node for node in graph['nodes']}
    for target, release in gates.items(): target_by_release_warp[(release, nodes[target]['warp'])] = target
    return tuple(Phase(phase['release'], tuple(phase['block']), phase['barrier'], phase['phase'],
        tuple(phase['arrival_issues']), tuple(phase['exit_issues']),
        tuple((p['issue'], target_by_release_warp[(phase['release'], p['warp'])])
              for p in phase['participants'] if p['blocking']))
        for phase in graph['cta_phases'] if phase['launch'] == launch)
