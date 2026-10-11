"""Indivisible LSU admission and completion joins for global/shared PTX traffic."""
from collections import Counter
from dataclasses import asdict,dataclass
from sector_memory_service import MemoryConfig, SectorMemoryService
from shared_memory_service import SharedConfig, SharedRequest, SharedMemoryService
from memory_model import positive_integer
from global_atomic_service import AtomicConfig,GlobalAtomicRequest,AtomicMemoryService


@dataclass(frozen=True)
class SectorRequest:
    operation: str
    sectors: tuple
    bypass_l1: bool = False

    def validate(self, config):
        if not isinstance(config,MemoryConfig): raise ValueError('Global fragment needs a sector-service configuration')
        if self.operation not in ['load','store'] or type(self.bypass_l1) is not bool: raise ValueError('Unsupported modeled sector request')
        if not isinstance(self.sectors,tuple) or not self.sectors: raise ValueError('Missing immutable sector request')
        if any(type(s) is not int or s<0 for s in self.sectors): raise ValueError('Invalid modeled sector identity')
        if len(set(self.sectors))!=len(self.sectors) or len(self.sectors)>config.max_sectors_per_request:
            raise ValueError('Invalid or excessive modeled sector request width')


@dataclass(frozen=True)
class CompositeRequest:
    global_fragment: SectorRequest | GlobalAtomicRequest
    shared_fragment: SharedRequest


def memory_record(request):
    if isinstance(request,CompositeRequest):return {'global_fragment':memory_record(request.global_fragment),'shared_fragment':memory_record(request.shared_fragment)}
    result=asdict(request)
    if isinstance(request,GlobalAtomicRequest):result['sectors']=request.sectors
    return result


class MemorySystem:
    """The common LSU slot survives until the last routed fragment returns.

    Shared/global bank/cache capacities remain independent. Common LSU capacity
    uses the global configuration if present; otherwise the shared request cap.
    The initiation interval is the maximum of participating route intervals.
    These are explicit assumptions, not measured NVIDIA port sharing.
    """
    def __init__(self, memory_config=None, shared_config=None, readiness=None,atomic_config=None):
        if memory_config is None and shared_config is None: raise ValueError('Missing memory services')
        for config,kind in [(memory_config,MemoryConfig),(shared_config,SharedConfig)]:
            if config is not None:
                if not isinstance(config,kind):raise ValueError('Invalid memory service configuration')
                config.validate()
        if atomic_config is not None:
            if not isinstance(atomic_config,AtomicConfig) or memory_config is None:raise ValueError('Global atomics need atomic and sector configurations')
            atomic_config.validate()
        self.count=(memory_config or shared_config).sm_count
        if any(c and c.sm_count!=self.count for c in [memory_config,shared_config]): raise ValueError('Memory service SM counts disagree')
        self.capacity=memory_config.lsu_slots_per_sm if memory_config else shared_config.request_slots_per_sm
        if readiness is not None:
            if not isinstance(readiness,dict) or set(readiness)!={'lsu','global','shared'} or not isinstance(readiness['lsu'],list) or len(readiness['lsu'])!=self.count or any(type(v) is not int or v<0 for v in readiness['lsu']):
                raise ValueError('Malformed common LSU readiness')
            if (readiness['global'] is None)!=(memory_config is None) or (readiness['shared'] is None)!=(shared_config is None): raise ValueError('Memory readiness route mismatch')
        self.global_service=(AtomicMemoryService(memory_config,atomic_config) if atomic_config else SectorMemoryService(memory_config)) if memory_config else None
        if self.global_service and readiness is not None:self.global_service.restore_port_readiness(readiness['global'])
        self.shared_service=SharedMemoryService(shared_config,readiness['shared'] if readiness else None) if shared_config else None
        self.next_port=list(readiness['lsu']) if readiness else [0]*self.count
        self.initial=self.port_readiness() if readiness is not None else None
        self.used=[0]*self.count;self.cycle=0;self.requests={};self.finished={};self.resource_records=[];self.stats=Counter();self.peak=0

    def fragments(self,request):
        if isinstance(request,(SectorRequest,GlobalAtomicRequest)):return {'global':request}
        if isinstance(request,SharedRequest):return {'shared':request}
        if isinstance(request,CompositeRequest):return {'global':request.global_fragment,'shared':request.shared_fragment}
        raise ValueError('Missing typed memory request')

    def validate(self,request):
        fragments=self.fragments(request)
        for name,fragment in fragments.items():
            provider=self.global_service if name=='global' else self.shared_service
            if provider is None:raise ValueError('Undeclared '+name+' memory service')
            if not isinstance(fragment,(SectorRequest,GlobalAtomicRequest) if name=='global' else SharedRequest):raise ValueError('Invalid typed memory fragment')
            if isinstance(fragment,GlobalAtomicRequest):
                if not isinstance(provider,AtomicMemoryService):raise ValueError('Undeclared global atomic service')
                fragment.validate(provider.config,provider.atomic_config)
            else:fragment.validate(provider.config)
        if len({fragment.operation for fragment in fragments.values()})!=1:raise ValueError('Mixed instruction operations disagree')

    def submit(self,identity,sm,request):
        positive_integer(identity,'memory request identity')
        if identity in self.requests or identity in self.finished:raise ValueError('Duplicate common memory identity')
        if type(sm) is not int or not 0<=sm<self.count:raise ValueError('Common LSU outside modeled SMs')
        self.validate(request);fragments=self.fragments(request)
        providers={name:self.global_service if name=='global' else self.shared_service for name in fragments}
        if self.used[sm]>=self.capacity or self.next_port[sm]>self.cycle or any(not p.can_submit(sm) for p in providers.values()):
            self.stats['admission_blocked_probes']+=1;return False
        # Every route is validated and preflighted before any reservation.
        for name,fragment in fragments.items():
            if isinstance(fragment,GlobalAtomicRequest):accepted=providers[name].submit_atomic(identity,sm,fragment)
            elif name=='global':accepted=providers[name].submit(identity,sm,fragment.sectors,operation=fragment.operation,bypass_l1=fragment.bypass_l1)
            else:accepted=providers[name].submit(identity,sm,fragment)
            if not accepted:raise AssertionError('Preflighted fragment was partially admitted')
        interval=max(p.config.lsu_interval if name=='global' else p.config.request_interval for name,p in providers.items())
        self.next_port[sm]=self.cycle+interval;self.used[sm]+=1;self.peak=max(self.peak,sum(self.used))
        self.requests[identity]={'identity':identity,'sm':sm,'submit_cycle':self.cycle,'interval':interval,'routes':list(fragments),'fragments':{}}
        self.resource_records.append({'cycle':self.cycle,'sm':sm,'identity':identity,'delta':1})
        self.stats['accepted_requests']+=1;return True

    def next_wakeup(self):
        wakes=[p.next_wakeup() for p in [self.global_service,self.shared_service] if p]
        return min((w for w in wakes if w is not None),default=None)

    def next_admission(self,sm):
        if type(sm) is not int or not 0<=sm<self.count:raise ValueError('Invalid common LSU SM')
        if self.used[sm]>=self.capacity:return self.next_wakeup()
        if self.next_port[sm]>self.cycle:return self.next_port[sm]
        wakes=[p.next_admission(sm) for p in [self.global_service,self.shared_service] if p]
        return min((w for w in wakes if w is not None and w>self.cycle),default=self.next_wakeup())

    def advance(self,target):
        if type(target) is not int or target<self.cycle:raise ValueError('Common memory time cannot move backwards')
        for name,provider in [('global',self.global_service),('shared',self.shared_service)]:
            if provider:
                for response in provider.advance(target):
                    request=self.requests[response['identity']]
                    if name in request['fragments']:raise AssertionError('Duplicate fragment response')
                    request['fragments'][name]=response['complete_cycle']
        complete=[]
        ready=[r for r in self.requests.values() if set(r['fragments'])==set(r['routes'])]
        for r in sorted(ready,key=lambda r:(max(r['fragments'].values()),r['identity'])):
            r['complete_cycle']=max(r['fragments'].values());identity=r['identity'];sm=r['sm']
            self.requests.pop(identity);self.finished[identity]=r;self.used[sm]-=1
            self.resource_records.append({'cycle':r['complete_cycle'],'sm':sm,'identity':identity,'delta':-1})
            complete.append({'identity':identity,'complete_cycle':r['complete_cycle']})
        self.cycle=target;return complete

    def port_readiness(self):
        return {'lsu':list(self.next_port),'global':self.global_service.port_readiness() if self.global_service else None,
            'shared':self.shared_service.port_readiness() if self.shared_service else None}

    def finish(self):
        if self.requests or any(self.used):raise AssertionError('Common LSU responses have not drained')
        if self.global_service:self.global_service.finish();self.cycle=max(self.cycle,self.global_service.cycle)
        if self.shared_service:self.shared_service.report()

    def report(self):
        if self.requests or any(self.used):raise AssertionError('Common memory is not drained')
        return {'requests':list(self.finished.values()),'resource_records':self.resource_records,'capacity_per_sm':self.capacity,
            'stats':dict(self.stats),'peak_requests':self.peak,'initial_port_readiness':self.initial,'next_port_readiness':self.port_readiness(),
            'drained':True,'assumption':'One common LSU slot per PTX issue through the last fragment response; participating route intervals take their maximum'}

    def global_report(self):
        p=self.global_service
        if p is None:return None
        return {'config':p.config.record(),'stats':dict(p.stats),'peaks':dict(p.peak),'requests':list(p.finished.values()),
            'dram_reads':p.dram_records,'writebacks':p.writeback_records,'store_commits':p.commit_records,
            'resource_records':p.resource_records,'lookup_records':p.lookup_records,
            'initial_port_readiness':self.initial['global'] if self.initial else None,'next_port_readiness':p.port_readiness(),
            'final_snapshot':p.snapshot(),'closed':p.closed,
            'atomic_service':p.atomic_report() if isinstance(p,AtomicMemoryService) else None}
