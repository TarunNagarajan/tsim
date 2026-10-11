"""Finite scalar global RMW jobs integrated with lookup, fill and dirty queues.

Word exclusion follows overlapping default GPU-scope atomicity. L2 execution,
sector lookup fanout, line pins, arbitration, unconditional dirtying and every
capacity/latency are explicit model assumptions; no data values or calibration.
"""
from collections import Counter,deque
from dataclasses import asdict,dataclass
import heapq,json
from pathlib import Path
from memory_model import positive_integer
from sector_memory_service import MemoryConfig,SectorMemoryService


@dataclass(frozen=True)
class AtomicAccess:
    lane:int
    element:int
    address:int
    width:int


@dataclass(frozen=True)
class GlobalAtomicRequest:
    accesses:tuple
    operation:str='atomic'
    bypass_l1:bool=True

    @property
    def sectors(self):return tuple(sorted({a.address//32 for a in self.accesses}))

    def validate(self,memory_config=None,atomic_config=None):
        if self.operation!='atomic' or self.bypass_l1 is not True:raise ValueError('Global atomic request must bypass L1')
        if not isinstance(self.accesses,tuple) or not self.accesses:raise ValueError('Missing immutable atomic accesses')
        lanes=set();widths=set()
        for a in self.accesses:
            if not isinstance(a,AtomicAccess) or type(a.lane) is not int or not 0<=a.lane<32 or type(a.element) is not int or a.element!=0:
                raise ValueError('Atomic timing requires distinct scalar warp lanes')
            if a.lane in lanes:raise ValueError('Duplicate atomic lane update')
            if type(a.width) is not int or a.width not in [4,8] or type(a.address) is not int or a.address<0 or a.address+a.width>1<<64 or a.address%a.width:
                raise ValueError('Atomic timing requires naturally aligned scalar32/64 addresses')
            lanes.add(a.lane);widths.add(a.width)
        if len(widths)!=1:raise ValueError('Atomic lanes disagree on scalar width')
        if memory_config is not None:
            if not isinstance(memory_config,MemoryConfig):raise ValueError('Global atomic timing requires sector configuration')
            if len(self.sectors)>memory_config.max_sectors_per_request:raise ValueError('Atomic sector fanout exceeds configured bound')
        if atomic_config is not None and len(self.accesses)>atomic_config.max_updates_per_request:raise ValueError('Atomic updates exceed configured bound')

    @classmethod
    def captured(cls,rows):
        if not rows or any(r['space']!='global' or r['operation']!='atomic' for r in rows):raise ValueError('Atomic fragment mixes space/operation')
        request=cls(tuple(AtomicAccess(r['lane'],r['element'],int(r['address'],16),r['bytes']) for r in rows))
        request.validate();return request


@dataclass(frozen=True)
class AtomicConfig:
    queue:int=8
    slots:int=4
    interval:int=1
    latency:int=4
    max_updates_per_request:int=32
    provenance:str='Synthetic global L2 scalar RMW service with finite FIFO/word locks/line pins; unconditional dirtying and no data/coherence/calibration'

    def validate(self):
        for name in self.__dataclass_fields__:
            if name!='provenance':positive_integer(getattr(self,name),name)
        if not isinstance(self.provenance,str) or not self.provenance.strip():raise ValueError('Missing atomic assumptions')
        if self.max_updates_per_request>32:raise ValueError('Atomic request bound exceeds scalar warp width')

    def record(self):self.validate();return {'schema':1,'provenance_kind':'assumption',**asdict(self)}


def load_atomic_config(path):
    data=json.loads(Path(path).read_text())
    if not isinstance(data,dict) or set(data)!=set(AtomicConfig.__dataclass_fields__)|{'schema','provenance_kind'}:raise ValueError('Malformed atomic configuration')
    schema=data.pop('schema')
    if type(schema) is not int or schema!=1 or data.pop('provenance_kind')!='assumption':raise ValueError('Unsupported atomic provenance')
    config=AtomicConfig(**data);config.validate();return config


class AtomicMemoryService(SectorMemoryService):
    def __init__(self,memory_config=None,atomic_config=None):
        super().__init__(memory_config);self.atomic_config=atomic_config or AtomicConfig();self.atomic_config.validate()
        self.atomic_requests={};self.atomic_jobs={};self.atomic_pending=deque();self.atomic_queue=deque();self.atomic_active={}
        self.atomic_locks={};self.atomic_pins={};self.atomic_remaining={};self.atomic_ready=set();self.next_atomic=0
        self.atomic_resources=[];self.atomic_records=[];self.atomic_fills=[];self.atomic_stats=Counter();self.atomic_peaks=Counter();self.atomic_order=0
        self.initial_atomic=0

    def _atomic_resource(self,name,token,delta):
        self.atomic_order+=1
        self.atomic_resources.append({'cycle':self.cycle,'ordinal':self.atomic_order,'resource':name,'token':list(token),'delta':delta})

    def submit_atomic(self,identity,sm,request):
        if not isinstance(request,GlobalAtomicRequest):raise ValueError('Missing typed global atomic request')
        request.validate(self.config,self.atomic_config)
        # The ordinary provider validates identity and admission before mutation.
        accepted=super().submit(identity,sm,request.sectors,operation='load',bypass_l1=True)
        if not accepted:return False
        self.requests[identity]['operation']='atomic'
        self.atomic_requests[identity]={'identity':identity,'sm':sm,'submit_cycle':self.cycle,'accesses':asdict(request)['accesses'],'sectors':list(request.sectors)}
        for number,a in enumerate(sorted(request.accesses,key=lambda a:a.lane)):
            token=(identity,number);sector=a.address//32
            words=list(range(a.address//4,(a.address+a.width-1)//4+1))
            self.atomic_jobs[token]={'identity':identity,'job':number,'lane':a.lane,'element':a.element,'address':a.address,'width':a.width,'sector':sector,'words':words,'ready_cycle':None}
            self.atomic_remaining[(identity,sector)]=self.atomic_remaining.get((identity,sector),0)+1
            self._atomic_resource('upstream',token,1)
        self.atomic_stats['accepted_requests']+=1;self.atomic_stats['updates']+=len(request.accesses)
        return True

    def _event(self,when,stage,payload):
        if stage!='atomic':return super()._event(when,stage,payload)
        self._serial+=1;heapq.heappush(self._events,(when,0,self._serial,stage,payload))

    def _response(self,pair):
        if self.requests[pair[0]]['operation']!='atomic':return super()._response(pair)
        if pair in self.atomic_ready:raise AssertionError('Duplicate atomic sector availability')
        self.atomic_ready.add(pair);line=pair[1]//self.l2.sectors_per_line
        self.atomic_pins.setdefault(line,set()).add(pair);self._atomic_resource('pins',pair,1)
        self.atomic_peaks['pins']=max(self.atomic_peaks['pins'],sum(len(v) for v in self.atomic_pins.values()))
        for token,job in self.atomic_jobs.items():
            if token[0]==pair[0] and job['sector']==pair[1]:
                job['ready_cycle']=self.cycle;self.atomic_pending.append(token)

    def _l2_hit(self,pair):
        self._active_lookups[pair]['outcome']='hit';super()._l2_hit(pair)

    def _lookup_result(self,pair):
        self._active_lookups[pair]['outcome']='miss';return super()._lookup_result(pair)

    def _pin_blocks_fill(self,sector):
        line=sector//self.l2.sectors_per_line;bucket=self.l2.sets[line%self.l2.set_count]
        if ('global',line) in bucket or len(bucket)<self.l2.ways:return False
        (namespace,victim),_=next(iter(bucket.items()))
        return bool(self.atomic_pins.get(victim))

    def _try_fill(self,sector):
        if self._pin_blocks_fill(sector):self.atomic_stats['pin_blocked_fill_probes']+=1;return False
        dirty=self._victim(sector)
        if len(self._writebacks)+len(dirty)>self.config.writeback_slots:return False
        line=sector//self.l2.sectors_per_line;bucket=self.l2.sets[line%self.l2.set_count]
        victim=next(iter(bucket))[1] if ('global',line) not in bucket and len(bucket)==self.l2.ways else None
        self.atomic_order+=1
        self.atomic_fills.append({'cycle':self.cycle,'ordinal':self.atomic_order,'sector':sector,'victim_line':victim})
        if not super()._try_fill(sector):raise AssertionError('Preflighted atomic-cache fill changed capacity')
        return True

    def _complete_miss(self,sector):
        if not self._try_fill(sector):
            self._ready_fills.append(sector);self.stats['fill_backpressure_blocked_probes']+=1
            self.peak['ready_fills']=max(self.peak['ready_fills'],len(self._ready_fills))

    def _pump(self):
        super()._pump()
        while self.atomic_pending and len(self.atomic_queue)<self.atomic_config.queue:
            token=self.atomic_pending.popleft();self._atomic_resource('upstream',token,-1)
            self.atomic_queue.append(token);self._atomic_resource('queue',token,1)
            self.atomic_peaks['queue']=max(self.atomic_peaks['queue'],len(self.atomic_queue))
        if self.atomic_queue and len(self.atomic_active)<self.atomic_config.slots and self.next_atomic<=self.cycle:
            token=self.atomic_queue[0];job=self.atomic_jobs[token]
            if any(w in self.atomic_locks for w in job['words']):return
            self.atomic_queue.popleft();self._atomic_resource('queue',token,-1)
            for word in job['words']:self.atomic_locks[word]=token
            self.atomic_active[token]=job;self._atomic_resource('slots',token,1)
            self.atomic_peaks['slots']=max(self.atomic_peaks['slots'],len(self.atomic_active))
            self.next_atomic=self.cycle+self.atomic_config.interval
            record=dict(job,start_cycle=self.cycle,complete_cycle=self.cycle+self.atomic_config.latency)
            self.atomic_records.append(record);self._event(record['complete_cycle'],'atomic',token)

    def _dispatch(self,stage,payload):
        if stage!='atomic':return super()._dispatch(stage,payload)
        job=self.atomic_active.pop(payload);identity,sector=job['identity'],job['sector']
        lookup=self.l2.request({sector},allocate=False)
        if lookup['hits']!={sector}:raise AssertionError('Atomic cache reservation lost its data')
        self.l2.request({sector},write=True)
        for word in job['words']:del self.atomic_locks[word]
        self._atomic_resource('slots',payload,-1);self.atomic_stats['committed_updates']+=1
        pair=(identity,sector);self.atomic_remaining[pair]-=1
        if not self.atomic_remaining[pair]:
            del self.atomic_remaining[pair];line=sector//self.l2.sectors_per_line
            self.atomic_pins[line].remove(pair)
            if not self.atomic_pins[line]:del self.atomic_pins[line]
            self._atomic_resource('pins',pair,-1)
            # One sector response follows its last distinct scalar RMW update.
            super()._response(pair)
        self._retry_fills()

    def next_wakeup(self):
        candidates=[super().next_wakeup()]
        if self.atomic_queue and len(self.atomic_active)<self.atomic_config.slots and self.next_atomic>self.cycle:candidates.append(self.next_atomic)
        return min((w for w in candidates if w is not None),default=None)

    def _assert_drained(self):
        super()._assert_drained()
        if self.atomic_pending or self.atomic_queue or self.atomic_active or self.atomic_locks or self.atomic_pins or self.atomic_remaining:raise AssertionError('Global atomic resources have not drained')

    def snapshot(self):
        return {**super().snapshot(),'atomic_queue':len(self.atomic_queue),'atomic_upstream':len(self.atomic_pending),
            'atomic_slots':len(self.atomic_active),'atomic_locks':len(self.atomic_locks),'atomic_pins':sum(len(v) for v in self.atomic_pins.values())}

    def port_readiness(self):return {**super().port_readiness(),'atomic':self.next_atomic}

    def restore_port_readiness(self,readiness):
        if not isinstance(readiness,dict) or set(readiness)!={'lsu','l2','backing','atomic'} or type(readiness['atomic']) is not int or readiness['atomic']<0:raise ValueError('Malformed atomic port readiness')
        super().restore_port_readiness({k:v for k,v in readiness.items() if k!='atomic'})
        self.next_atomic=self.initial_atomic=readiness['atomic']

    def atomic_report(self):
        self._assert_drained()
        return {'config':self.atomic_config.record(),'requests':list(self.atomic_requests.values()),'jobs':self.atomic_records,
            'resources':self.atomic_resources,'fills':self.atomic_fills,'stats':dict(self.atomic_stats),'peaks':dict(self.atomic_peaks),
            'initial_port_readiness':self.initial_atomic,'next_port_readiness':self.next_atomic,'drained':True,
            'assumptions':['One sector lookup/fill fans out to every captured scalar lane update; updates are never merged',
                'One finite global FIFO and RMW port, word locks and conservative LRU victim-line pin stalls',
                'All modeled atomic updates conservatively dirty the sector, including conditional/no-change operations',
                'Default GPU-scope scalar32/64 subset; no data/coherence/scoped ordering/SASS or hardware latency prediction']}
