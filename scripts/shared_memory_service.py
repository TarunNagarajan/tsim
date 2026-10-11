"""Finite shared bank service over logical PTX byte spans; no data/timing calibration."""
from collections import Counter, deque
from dataclasses import asdict, dataclass
import heapq
import json
from pathlib import Path
from memory_model import SHARED_BASE, positive_integer

@dataclass(frozen=True)
class SharedAccess:
    lane: int
    element: int
    offset: int
    width: int

@dataclass(frozen=True)
class SharedRequest:
    operation: str
    block: tuple
    accesses: tuple

    def validate(self, config=None):
        if self.operation not in ['load','store','atomic']: raise ValueError('Unsupported shared operation')
        if not isinstance(self.block,tuple) or len(self.block)!=3 or any(type(v) is not int or v<0 for v in self.block):
            raise ValueError('Invalid shared CTA owner')
        if not isinstance(self.accesses,tuple) or not self.accesses: raise ValueError('Missing immutable shared accesses')
        seen, lanes = set(), {}
        for a in self.accesses:
            if not isinstance(a,SharedAccess) or type(a.lane) is not int or not 0<=a.lane<32 or type(a.element) is not int or not 0<=a.element<4:
                raise ValueError('Invalid shared lane/element identity')
            if type(a.width) is not int or a.width not in [1,2,4,8,16] or type(a.offset) is not int or a.offset<0 or a.offset+a.width>1<<24 or a.offset%a.width:
                raise ValueError('Unsupported shared width, alignment or local address')
            if (a.lane,a.element) in seen: raise ValueError('Duplicate shared lane/element')
            seen.add((a.lane,a.element)); lanes.setdefault(a.lane,[]).append(a)
            if self.operation=='atomic' and (a.element or a.width not in [4,8]): raise ValueError('Shared atomic service requires scalar 32/64-bit updates')
        layouts=set()
        for group in lanes.values():
            ordered=sorted(group,key=lambda a:a.element); first=ordered[0]
            if len(ordered) not in [1,2,4] or first.offset%(len(ordered)*first.width) or [a.element for a in ordered]!=list(range(len(ordered))) or any(a.width!=first.width or a.offset!=first.offset+a.element*first.width for a in ordered):
                raise ValueError('Shared vector elements must preserve complete contiguous lane spans')
            layouts.add((len(ordered),first.width))
        if len(layouts)!=1: raise ValueError('Shared participating lanes disagree on width/vector layout')
        if config and sum(a.width for a in self.accesses)>config.max_request_bytes: raise ValueError('Shared request exceeds declared byte bound')

    @classmethod
    def captured(cls, operation, block, rows):
        if any(r['space']!='shared' or r['operation']!=operation for r in rows): raise ValueError('Shared fragment mixes memory spaces or operations')
        result=cls(operation,tuple(block),tuple(SharedAccess(r['lane'],r['element'],int(r['address'],16)-SHARED_BASE,r['bytes']) for r in rows))
        result.validate(); return result

def word_jobs(request):
    """Merge ordinary same-word traffic; retain each atomic update separately."""
    request.validate(); words, jobs = {}, []
    for a in sorted(request.accesses,key=lambda a:(a.lane,a.element)):
        touched=[]
        for word in range(a.offset//4,(a.offset+a.width-1)//4+1):
            low=max(a.offset,word*4)-word*4; high=min(a.offset+a.width,word*4+4)-word*4
            owner={'lane':a.lane,'element':a.element,'byte_mask':((1<<(high-low))-1)<<low}
            touched.append((word,owner))
            if request.operation!='atomic': words.setdefault(word,[]).append(owner)
        if request.operation=='atomic':
            jobs.append({'words':[w for w,_ in touched], 'owners':[owner for _,owner in touched], 'atomic':True})
    if request.operation!='atomic': jobs=[{'words':[w],'owners':owners,'atomic':False} for w,owners in sorted(words.items())]
    for job in jobs: job['banks']=sorted({w%32 for w in job['words']})
    return jobs

def shared_topology(request):
    jobs=word_jobs(request); banks=Counter(); owners={}; byte_owners={}
    for a in request.accesses:
        for byte in range(a.offset,a.offset+a.width): byte_owners.setdefault(byte,set()).add(a.lane)
        for w in range(a.offset//4,(a.offset+a.width-1)//4+1): owners.setdefault(w,set()).add(a.lane)
    for job in jobs: banks.update(job['banks'])
    return {'banks_touched':len(banks),'distinct_words':len(owners),'logical_word_jobs':len(jobs),
        'bank_work_counts':[banks[b] for b in range(32)],'max_bank_work':max(banks.values(),default=0),
        'same_word_extra_lanes':sum(len(v)-1 for v in owners.values()),
        'write_alias_bytes':sum(len(v)>1 for v in byte_owners.values()) if request.operation=='store' else 0,
        'atomic_updates':len(request.accesses) if request.operation=='atomic' else 0,
        'requested_bytes':sum(a.width for a in request.accesses)}

@dataclass(frozen=True)
class SharedConfig:
    sm_count: int=1
    request_slots_per_sm: int=8
    request_interval: int=1
    front_latency: int=1
    bank_queue: int=4
    bank_slots: int=2
    bank_interval: int=1
    bank_latency: int=2
    atomic_latency: int=4
    return_latency: int=2
    max_request_bytes: int=4096
    provenance: str='Synthetic logical PTX word service with finite bank queues; no SASS splitting, data values or hardware calibration'

    def validate(self):
        for name in self.__dataclass_fields__:
            if name!='provenance': positive_integer(getattr(self,name),name)
        if not isinstance(self.provenance,str) or not self.provenance.strip(): raise ValueError('Missing shared service assumptions')

    def record(self):
        self.validate(); return {'schema':1,'provenance_kind':'assumption',**asdict(self)}

def load_shared_config(path):
    data=json.loads(Path(path).read_text())
    if not isinstance(data,dict) or set(data)!=set(SharedConfig.__dataclass_fields__)|{'schema','provenance_kind'}:
        raise ValueError('Malformed shared service configuration')
    schema=data.pop('schema')
    if type(schema) is not int or schema!=1 or data.pop('provenance_kind')!='assumption': raise ValueError('Unsupported shared service provenance')
    config=SharedConfig(**data);config.validate();return config

class SharedMemoryService:
    def __init__(self,config=None,readiness=None):
        self.config=config or SharedConfig();self.config.validate();n=self.config.sm_count
        self.cycle=0;self.requests={};self.finished={};self._completed=[];self._events=[];self._serial=0
        self.used=[0]*n;self.next_request=[0]*n;self.next_bank=[[0]*32 for _ in range(n)]
        if readiness is not None:
            if not isinstance(readiness,dict) or set(readiness)!={'request','bank'} or not isinstance(readiness['request'],list) or not isinstance(readiness['bank'],list) or len(readiness['request'])!=n or len(readiness['bank'])!=n:
                raise ValueError('Malformed shared port readiness')
            if any(not isinstance(row,list) or len(row)!=32 for row in readiness['bank']): raise ValueError('Incomplete shared bank readiness')
            if any(type(v) is not int or v<0 for v in readiness['request']+[v for row in readiness['bank'] for v in row]): raise ValueError('Invalid shared port deadline')
            self.next_request=list(readiness['request']);self.next_bank=[list(row) for row in readiness['bank']]
        self.initial=self.port_readiness() if readiness is not None else None;self.queues=[[deque() for _ in range(32)] for _ in range(n)]
        self.active=[[set() for _ in range(32)] for _ in range(n)];self.locks={};self.jobs={}
        self.resource_records=[];self.job_records=[];self.stats=Counter();self.peaks=Counter()

    def _resource(self,kind,sm,bank,token,delta):
        self.resource_records.append({'cycle':self.cycle,'resource':kind,'sm':sm,'bank':bank,'token':list(token),'delta':delta})

    def _event(self,time,kind,value):
        self._serial+=1;heapq.heappush(self._events,(time,self._serial,kind,value))

    def can_submit(self,sm):
        if type(sm) is not int or not 0<=sm<self.config.sm_count: raise ValueError('Shared request outside modeled SMs')
        return self.used[sm]<self.config.request_slots_per_sm and self.next_request[sm]<=self.cycle

    def submit(self,identity,sm,request):
        if type(identity) is not int or identity<=0 or identity in self.requests or identity in self.finished: raise ValueError('Invalid shared request identity')
        if not isinstance(request,SharedRequest): raise ValueError('Missing typed shared request')
        request.validate(self.config)
        if not self.can_submit(sm):self.stats['admission_blocked_probes']+=1;return False
        jobs=word_jobs(request);ready=self.cycle+self.config.front_latency
        self.requests[identity]={'identity':identity,'sm':sm,'request':asdict(request),'submit_cycle':self.cycle,'front_ready_cycle':ready,
            'pending':deque((identity,i) for i in range(len(jobs))),'remaining':len(jobs)}
        for i,job in enumerate(jobs):
            token=(identity,i);self.jobs[token]=dict(job,sm=sm,block=request.block,ready=ready)
            self._resource('upstream',sm,-1,token,1)
        self.used[sm]+=1;self.next_request[sm]=self.cycle+self.config.request_interval
        self._resource('requests',sm,-1,(identity,),1)
        self.peaks['requests']=max(self.peaks['requests'],sum(self.used));self.stats['accepted_requests']+=1;self.stats['logical_word_jobs']+=len(jobs)
        self._event(ready,'front',None);return True

    def _pump(self):
        for request in self.requests.values():
            if request['front_ready_cycle']>self.cycle:continue
            sm=request['sm']
            while request['pending']:
                token=request['pending'][0];job=self.jobs[token]
                if any(len(self.queues[sm][b])>=self.config.bank_queue for b in job['banks']):break
                request['pending'].popleft();self._resource('upstream',sm,-1,token,-1)
                for b in job['banks']:
                    self.queues[sm][b].append(token);self._resource('queue',sm,b,token,1)
                    self.peaks['bank_queue']=max(self.peaks['bank_queue'],len(self.queues[sm][b]))
        for sm in range(self.config.sm_count):
            for bank in range(32):
                if not self.queues[sm][bank]:continue
                token=self.queues[sm][bank][0];job=self.jobs[token];banks=job['banks']
                if any(not self.queues[sm][b] or self.queues[sm][b][0]!=token or self.next_bank[sm][b]>self.cycle or len(self.active[sm][b])>=self.config.bank_slots for b in banks):continue
                lock_keys=[(sm,job['block'],word) for word in job['words']] if job['atomic'] else []
                if any(key in self.locks for key in lock_keys):continue
                end=self.cycle+(self.config.atomic_latency if job['atomic'] else self.config.bank_latency)
                for b in banks:
                    self.queues[sm][b].popleft();self._resource('queue',sm,b,token,-1)
                    self.active[sm][b].add(token);self._resource('bank',sm,b,token,1);self.next_bank[sm][b]=self.cycle+self.config.bank_interval
                    self.peaks['bank_slots']=max(self.peaks['bank_slots'],len(self.active[sm][b]))
                for key in lock_keys:self.locks[key]=token
                self.job_records.append({'identity':token[0],'job':token[1],'sm':sm,'block':list(job['block']),
                    'words':job['words'],'banks':banks,'owners':job['owners'],'atomic':job['atomic'],
                    'ready_cycle':job['ready'],'start_cycle':self.cycle,'complete_cycle':end})
                self._event(end,'job',token)

    def next_wakeup(self):
        candidates=[self._events[0][0]] if self._events else []
        for sm in range(self.config.sm_count):
            for b in range(32):
                if self.queues[sm][b] and self.next_bank[sm][b]>self.cycle:candidates.append(self.next_bank[sm][b])
        return min(candidates) if candidates else None

    def next_admission(self,sm):
        if self.used[sm]<self.config.request_slots_per_sm:return max(self.cycle,self.next_request[sm])
        return self.next_wakeup()

    def advance(self,target):
        if type(target) is not int or target<self.cycle:raise ValueError('Shared service cannot move backwards')
        while (wake:=self.next_wakeup()) is not None and wake<=target:
            self.cycle=wake
            while self._events and self._events[0][0]<=wake:
                _,_,kind,token=heapq.heappop(self._events)
                if kind=='job':
                    job=self.jobs[token];sm=job['sm']
                    for b in job['banks']:self.active[sm][b].remove(token);self._resource('bank',sm,b,token,-1)
                    if job['atomic']:
                        for word in job['words']:del self.locks[(sm,job['block'],word)]
                    request=self.requests[token[0]];request['remaining']-=1
                    if not request['remaining']:self._event(self.cycle+self.config.return_latency,'response',token[0])
                elif kind=='response':
                    request=self.requests.pop(token);sm=request['sm'];self.used[sm]-=1;self._resource('requests',sm,-1,(token,),-1)
                    request.pop('pending');request.pop('remaining');request['complete_cycle']=self.cycle
                    self.finished[token]=request;self._completed.append({'identity':token,'complete_cycle':self.cycle})
            self._pump()
        self.cycle=target;self._pump();result,self._completed=self._completed,[];return result

    def port_readiness(self):return {'request':list(self.next_request),'bank':[list(row) for row in self.next_bank]}

    def report(self):
        if self.requests or self._events or any(self.used) or self.locks or any(q for row in self.queues for q in row) or any(a for row in self.active for a in row):
            raise AssertionError('Shared service resources have not drained')
        return {'config':self.config.record(),'requests':list(self.finished.values()),'jobs':self.job_records,
            'resource_records':self.resource_records,'stats':dict(self.stats),'peaks':dict(self.peaks),
            'initial_port_readiness':self.initial,'next_port_readiness':self.port_readiness(),'drained':True,
            'limits':['Logical PTX words are not predicted SASS instructions or 128-byte transactions',
                'Ordinary same-word traffic merges; overlapping store values remain undefined',
                'Atomic word locks serialize modeled overlapping updates in captured lane order; no data values or weak-memory outcomes',
                'Queue sizes, port sharing and latencies are uncalibrated assumptions']}
