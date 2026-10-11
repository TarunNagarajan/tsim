"""Hand-solved logical bank traffic, queue pressure and atomic word locks."""
from dataclasses import replace
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from shared_memory_service import SharedAccess,SharedRequest,SharedConfig,SharedMemoryService,shared_topology
from shared_service_reference import verify_shared_service
from copy import deepcopy
import random

def verify_shared_components():
    checks=[]
    def check(name,condition):
        if not condition:raise AssertionError(name)
        checks.append(name)
    def reject(name,action):
        try:action()
        except ValueError:checks.append(name)
        else:raise AssertionError(name)
    def request(stride=1,width=4,elements=1,lanes=32,operation='load',block=(0,0,0)):
        return SharedRequest(operation,block,tuple(SharedAccess(lane,e,lane*stride*width*elements+e*width,width) for lane in range(lanes) for e in range(elements)))
    def solve(req,config=None,readiness=None):
        service=SharedMemoryService(config,readiness);assert service.submit(1,0,req)
        while service.requests:
            next_cycle=service.next_wakeup();assert next_cycle is not None
            service.advance(next_cycle)
        result=service.report();assert verify_shared_service(result);return result
    unit=solve(request())
    check('thirty-two consecutive words use thirty-two banks in parallel',len(unit['jobs'])==32 and all(j['start_cycle']==1 and j['complete_cycle']==3 for j in unit['jobs']) and unit['requests'][0]['complete_cycle']==5)
    two=solve(request(2));check('stride-two bank work needs two initiation rounds',two['requests'][0]['complete_cycle']==6 and len({j['start_cycle'] for j in two['jobs']})==2)
    serial=solve(request(32));check('thirty-two distinct words in one bank retain all conflicts',serial['requests'][0]['complete_cycle']==36 and [j['start_cycle'] for j in serial['jobs']]==list(range(1,33)))
    limited=solve(request(32),SharedConfig(bank_slots=1));check('finite bank completion slots throttle below port initiation bandwidth',limited['requests'][0]['complete_cycle']==67 and limited['peaks']['bank_slots']==1)
    broadcast=solve(request(0));check('same-word load broadcasts once to all lane owners',len(broadcast['jobs'])==1 and len(broadcast['jobs'][0]['owners'])==32 and broadcast['requests'][0]['complete_cycle']==5)
    byte=solve(request(width=1));check('distinct byte addresses in the same word share one bank word',len(byte['jobs'])==8 and byte['requests'][0]['complete_cycle']==5)
    half=solve(request(width=2));check('halfword addresses preserve sixteen word jobs',len(half['jobs'])==16 and half['requests'][0]['complete_cycle']==5)
    wide=solve(request(width=8));check('contiguous scalar64 spans preserve sixty-four word jobs',len(wide['jobs'])==64 and wide['requests'][0]['complete_cycle']==6)
    vector=solve(request(elements=4));check('v4 scalar32 conserves all vector elements and word traffic',len(vector['jobs'])==128 and vector['requests'][0]['complete_cycle']==8 and shared_topology(request(elements=4))['requested_bytes']==512)
    wide_vector=solve(request(width=8,elements=4));check('v4 scalar64 spans are bounded logical traffic rather than invented SASS instructions',len(wide_vector['jobs'])==256 and wide_vector['requests'][0]['complete_cycle']==12)
    check('finite bank queues retain the upstream tail',serial['peaks']['bank_queue']==4 and sum(e['delta'] for e in serial['resource_records'] if e['resource']=='upstream')==0)
    alias=shared_topology(request(0,operation='store'));check('overlapping stores remain explicit without choosing a hardware winning lane',alias['write_alias_bytes']==4 and alias['same_word_extra_lanes']==31)
    atomic=solve(request(0,operation='atomic'));check('same-word atomic updates serialize until prior RMW completion',len(atomic['jobs'])==32 and atomic['requests'][0]['complete_cycle']==131 and [j['start_cycle'] for j in atomic['jobs']]==list(range(1,126,4)))
    atomic64=solve(request(0,width=8,operation='atomic'));check('scalar64 atomic retains joint two-bank reservation and word locks',all(j['banks']==[0,1] for j in atomic64['jobs']) and atomic64['requests'][0]['complete_cycle']==131)
    independent=solve(request(32,lanes=3,operation='atomic'));check('distinct-word atomics can overlap within finite bank slots',[j['start_cycle'] for j in independent['jobs']]==[1,2,5] and independent['requests'][0]['complete_cycle']==11)
    service=SharedMemoryService(SharedConfig(request_slots_per_sm=1));assert service.submit(1,0,request())
    check('request capacity is held through response transmission',not service.submit(2,0,request()) and service.next_admission(0)==1)
    assert service.advance(3)==[];check('bank completion alone does not free an ingress slot',not service.submit(2,0,request()))
    assert service.advance(5)==[{'identity':1,'complete_cycle':5}];assert service.submit(2,0,request())
    while service.requests:service.advance(service.next_wakeup())
    check('freed request storage is reused without losing first response',service.finished[2]['complete_cycle']==10 and service.report()['drained'])
    service=SharedMemoryService(SharedConfig(sm_count=2));assert service.submit(1,0,request(32));assert service.submit(2,1,request(32))
    while service.requests:service.advance(service.next_wakeup())
    check('different SMs have independent bank ports',service.finished[1]['complete_cycle']==service.finished[2]['complete_cycle']==36)
    service=SharedMemoryService(SharedConfig(bank_slots=2));assert service.submit(1,0,request(0,lanes=1,operation='atomic'));service.advance(1)
    assert service.submit(2,0,request(0,lanes=1,operation='atomic',block=(1,0,0)))
    while service.requests:service.advance(service.next_wakeup())
    check('shared atomic locks name CTA ownership while banks still contend',[j['start_cycle'] for j in service.job_records]==[1,2])
    first=solve(request(lanes=1),SharedConfig(bank_interval=20,request_interval=20))
    next_service=SharedMemoryService(SharedConfig(bank_interval=20,request_interval=20),first['next_port_readiness'])
    next_service.advance(5);check('cross-launch ingress deadline persists',not next_service.submit(2,0,request(lanes=1)) and next_service.next_admission(0)==20)
    next_service.advance(20);assert next_service.submit(2,0,request(lanes=1))
    while next_service.requests:next_service.advance(next_service.next_wakeup())
    check('cross-launch bank deadline persists independently of return tail',next_service.job_records[0]['start_cycle']==21 and next_service.finished[2]['complete_cycle']==25)
    for label,config in [('boolean bank latency',replace(SharedConfig(),bank_latency=True)),('zero bank slots',replace(SharedConfig(),bank_slots=0)),('zero queue',replace(SharedConfig(),bank_queue=0)),('missing assumptions',replace(SharedConfig(),provenance=''))]:
        reject(label,lambda config=config:SharedMemoryService(config))
    for label,req in [('misaligned scalar',SharedRequest('load',(0,0,0),(SharedAccess(0,0,2,4),))),
        ('misaligned vector',SharedRequest('load',(0,0,0),(SharedAccess(0,0,4,4),SharedAccess(0,1,8,4)))),
        ('missing vector element',SharedRequest('load',(0,0,0),(SharedAccess(0,1,4,4),))),
        ('duplicate element',SharedRequest('load',(0,0,0),(SharedAccess(0,0,0,4),SharedAccess(0,0,0,4)))),
        ('remote shared owner',SharedRequest('load',(0,0,0),(SharedAccess(0,0,1<<30,4),))),
        ('unsupported atomic width',request(width=2,operation='atomic')),('boolean lane',SharedRequest('load',(0,0,0),(SharedAccess(True,0,0,4),)))]:
        reject(label,lambda req=req:req.validate())
    reject('request byte bound',lambda:request(elements=4).validate(SharedConfig(max_request_bytes=128)))
    reject('backward service time',lambda:next_service.advance(24))
    reject('duplicate completed identity',lambda:next_service.submit(2,0,request(lanes=1)))
    for label,mutation in [
        ('missing bank work',lambda r:r['jobs'].pop()),
        ('invented owner bytes',lambda r:r['jobs'][0]['owners'][0].update(byte_mask=0)),
        ('invented bank route',lambda r:r['jobs'][0]['banks'].__setitem__(0,31)),
        ('early bank completion',lambda r:r['jobs'][0].update(complete_cycle=1)),
        ('early response',lambda r:r['requests'][0].update(complete_cycle=2)),
        ('missing resource release',lambda r:r['resource_records'].pop()),
        ('borrowed queue capacity',lambda r:r['peaks'].update(bank_queue=0)),
        ('erased bank deadline',lambda r:r['next_port_readiness']['bank'][0].__setitem__(0,0)),
        ('invented job counter',lambda r:r['stats'].update(logical_word_jobs=1))]:
        altered=deepcopy(serial);mutation(altered)
        try:verify_shared_service(altered)
        except AssertionError:checks.append('independent audit rejects '+label)
        else:raise AssertionError('Audit accepted '+label)
    rng=random.Random(831)
    for seed in range(40):
        config=SharedConfig(sm_count=2,bank_queue=rng.randint(1,4),bank_slots=rng.randint(1,3),
            bank_interval=rng.randint(1,4),bank_latency=rng.randint(1,7),atomic_latency=rng.randint(1,9),request_slots_per_sm=rng.randint(1,4))
        service=SharedMemoryService(config);pending=[]
        for identity in range(1,17):
            operation=rng.choice(['load','store','atomic']);width=rng.choice([4,8]) if operation=='atomic' else rng.choice([1,2,4,8,16])
            elements=1 if operation=='atomic' else rng.choice([1,2,4])
            pending.append((identity,rng.randrange(2),request(rng.choice([0,1,2,32]),width,elements,rng.randint(1,32),operation,(rng.randrange(3),0,0))))
        while pending or service.requests:
            for item in list(pending):
                if service.submit(*item):pending.remove(item)
            wakeups=[service.next_wakeup()]+[service.next_admission(sm) for sm in range(2)]
            future=[w for w in wakeups if w is not None and w>service.cycle]
            if pending or service.requests:
                assert future, 'Shared randomized progress';service.advance(min(future))
        check('independent randomized bank/atomic/resource audit '+str(seed),verify_shared_service(service.report()))
    return checks

if __name__=='__main__':print('PASS shared bank references checks='+str(len(verify_shared_components())))
