"""Hand-solved global RMW, overlapping words and pinned victim backpressure."""
from pathlib import Path
from dataclasses import asdict,replace
from copy import deepcopy
import sys
import random
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from global_atomic_service import AtomicAccess,GlobalAtomicRequest,AtomicConfig,AtomicMemoryService
from sector_memory_service import MemoryConfig
from global_atomic_reference import verify_global_atomics
from sector_service_reference import verify_service


def verify_global_atomic_components():
    checks=[]
    def check(name,test):
        if not test:raise AssertionError(name)
        checks.append(name)
    def reject(name,action):
        try:action()
        except ValueError:checks.append(name)
        else:raise AssertionError(name)
    def request(lanes=32,stride=0,width=4,address=0):return GlobalAtomicRequest(tuple(AtomicAccess(lane,0,address+lane*stride*width,width) for lane in range(lanes)))
    def report(s):
        return {'config':s.config.record(),'stats':dict(s.stats),'peaks':dict(s.peak),'requests':list(s.finished.values()),
            'dram_reads':s.dram_records,'writebacks':s.writeback_records,'store_commits':s.commit_records,'resource_records':s.resource_records,
            'lookup_records':s.lookup_records,'final_snapshot':s.snapshot(),'closed':s.closed,'atomic_service':s.atomic_report()}
    def solve(req,config=None,atomic=None):
        s=AtomicMemoryService(config,atomic);assert s.submit_atomic(1,0,req);s.finish();r=report(s)
        rows=[{'sequence':1,'sm':0,'issue_cycle':0,'complete_cycle':s.finished[1]['complete_cycle'],'memory':dict(asdict(req),sectors=req.sectors)}]
        assert verify_service(r,rows,0,s.cycle) and verify_global_atomics(r);return s,r
    s,hot=solve(request());j=hot['atomic_service']['jobs']
    check('thirty-two same-address lanes remain thirty-two serialized updates',len(j)==32 and [v['start_cycle'] for v in j]==list(range(16,141,4)) and s.finished[1]['complete_cycle']==146)
    check('atomic updates share one lookup/fill without merging RMW work',s.stats['l2_lookups']==s.stats['dram_reads']==1 and s.atomic_stats['committed_updates']==32)
    check('dirty atomic backing tail follows last RMW and response',s.cycle==157 and s.writeback_records[0]['reserve_cycle']==146 and s.writeback_records[0]['sector']==0)
    wide,_=solve(request(width=8));check('scalar64 locks both words until each overlapping update completes',all(v['words']==[0,1] for v in wide.atomic_records) and wide.finished[1]['complete_cycle']==146)
    distinct,_=solve(request(stride=1));check('independent words overlap within finite RMW slots',[v['start_cycle'] for v in distinct.atomic_records[:8]]==list(range(16,24)) and distinct.finished[1]['complete_cycle']==53)
    serial,_=solve(request(lanes=8,stride=1),atomic=AtomicConfig(slots=1));check('RMW completion slots can throttle independent words',serial.finished[1]['complete_cycle']==50)
    narrow,_=solve(request(lanes=8,stride=1),atomic=AtomicConfig(queue=1));check('finite RMW FIFO retains the upstream fanout',narrow.atomic_peaks['queue']==1 and narrow.atomic_stats['committed_updates']==8)
    s=AtomicMemoryService(MemoryConfig(sm_count=2));assert s.submit_atomic(1,0,request(lanes=1,width=8));s.advance(1);assert s.submit_atomic(2,1,request(lanes=1,width=4,address=4));s.finish();r=report(s)
    check('mixed-size overlapping RMW excludes the other SM high-word update',[(v['start_cycle'],v['complete_cycle']) for v in s.atomic_records]==[(16,20),(20,24)] and verify_global_atomics(r))
    s=AtomicMemoryService(MemoryConfig(cache_sets=1,cache_ways=1));assert s.submit_atomic(1,0,request(lanes=8));s.advance(1);assert s.submit(2,0,[4],operation='load',bypass_l1=True);s.finish();r=report(s)
    check('unrelated cache fill waits until pinned victim atomic work drains',s.atomic_stats['pin_blocked_fill_probes']>0 and any(f['sector']==4 and f['cycle']==48 and f['victim_line']==0 for f in s.atomic_fills) and verify_global_atomics(r))
    check('pin-stalled fill preserves MSHR and dirty eviction backing',s.finished[1]['complete_cycle']==50 and s.finished[2]['complete_cycle']==50 and s.writeback_records[0]['origin']=='eviction' and s.writeback_records[0]['reserve_cycle']==48)
    s=AtomicMemoryService(MemoryConfig(cache_sets=1,cache_ways=1,mshrs=1,l2_slots=1,l2_queue=1),AtomicConfig(queue=1,slots=1));assert s.submit_atomic(1,0,request(lanes=8,stride=8));s.finish()
    check('multi-sector atomic fanout avoids pin-induced upstream deadlock',s.atomic_stats['committed_updates']==8 and verify_global_atomics(report(s)))
    config=AtomicConfig(interval=100);s,r=solve(request(lanes=1),atomic=config);continuation=AtomicMemoryService(atomic_config=config);continuation.restore_port_readiness(s.port_readiness());continuation.advance(s.cycle);assert continuation.submit_atomic(2,0,request(lanes=1));continuation.finish()
    check('atomic initiation deadline survives cold next-launch caches',continuation.atomic_records[0]['start_cycle']==116 and continuation.finished[2]['complete_cycle']==122 and verify_global_atomics(report(continuation),s.cycle))
    one=AtomicMemoryService(MemoryConfig(lsu_slots_per_sm=1));assert one.submit_atomic(1,0,request(lanes=1));one.advance(20)
    check('atomic RMW completion retains LSU until response transmission',not one.submit_atomic(2,0,request(lanes=1)));one.advance(22);assert one.submit_atomic(2,0,request(lanes=1));one.finish()
    check('completed atomic data can feed a warm next RMW without backing refetch',one.stats['dram_reads']==1 and one.finished[2]['complete_cycle']==33)
    for label,req in [('duplicate lane',GlobalAtomicRequest((AtomicAccess(0,0,0,4),AtomicAccess(0,0,4,4)))),('misaligned64',request(lanes=1,width=8,address=4)),('vector atomic',GlobalAtomicRequest((AtomicAccess(0,1,0,4),))),('boolean address',GlobalAtomicRequest((AtomicAccess(0,0,True,4),))),('mixed lane width',GlobalAtomicRequest((AtomicAccess(0,0,0,4),AtomicAccess(1,0,8,8)))),('unsupported halfwidth',request(lanes=1,width=2))]:
        reject(label,lambda req=req:req.validate())
    for label,cfg in [('boolean RMW latency',replace(AtomicConfig(),latency=True)),('zero FIFO',replace(AtomicConfig(),queue=0)),('more than scalar warp bound',replace(AtomicConfig(),max_updates_per_request=33)),('missing assumptions',replace(AtomicConfig(),provenance=''))]:reject(label,lambda cfg=cfg:AtomicMemoryService(atomic_config=cfg))
    reject('update fanout bound enforced before admission',lambda:AtomicMemoryService(atomic_config=AtomicConfig(max_updates_per_request=1)).submit_atomic(1,0,request()))
    for label,mutation in [('missing lane update',lambda r:r['atomic_service']['jobs'].pop()),('wrong word lock',lambda r:r['atomic_service']['jobs'][0]['words'].__setitem__(0,9)),('early RMW completion',lambda r:r['atomic_service']['jobs'][0].update(complete_cycle=1)),('early response',lambda r:r['requests'][0].update(complete_cycle=22)),('missing pin release',lambda r:r['atomic_service']['resources'].pop()),('invented slot peak',lambda r:r['atomic_service']['peaks'].update(slots=0)),('erased deadline',lambda r:r['atomic_service'].update(next_port_readiness=0))]:
        damaged=deepcopy(hot);mutation(damaged)
        try:verify_global_atomics(damaged)
        except AssertionError:checks.append('independent atomic audit rejects '+label)
        else:raise AssertionError('Accepted '+label)
    damaged=deepcopy(hot);damaged['atomic_service']['fills'][0].update(victim_line=0,ordinal=max(e['ordinal'] for e in damaged['atomic_service']['resources'])+10)
    try:verify_global_atomics(damaged)
    except AssertionError:checks.append('independent atomic audit rejects pinned-line eviction')
    else:raise AssertionError('Accepted pinned-line eviction')
    rng=random.Random(937)
    for seed in range(40):
        cfg=MemoryConfig(sm_count=2,lsu_slots_per_sm=rng.randint(1,4),cache_sets=rng.randint(1,2),cache_ways=rng.randint(1,2),
            mshrs=rng.randint(1,4),l2_slots=rng.randint(1,4),l2_queue=rng.randint(1,4),writeback_slots=rng.choice([4,8]))
        atomic=AtomicConfig(queue=rng.randint(1,4),slots=rng.randint(1,4),interval=rng.randint(1,3),latency=rng.randint(1,9))
        s=AtomicMemoryService(cfg,atomic);pending=[];payloads={}
        for identity in range(1,17):
            sm=rng.randrange(2);address=rng.randrange(12)*128;op=rng.choice(['load','store','atomic'])
            if op=='atomic':payload=request(rng.randint(1,8),rng.choice([0,1,8]),rng.choice([4,8]),address);payloads[identity]=dict(asdict(payload),sectors=payload.sectors)
            else:payload=(address//32,);payloads[identity]={'operation':op,'sectors':payload,'bypass_l1':True}
            pending.append((identity,sm,op,payload))
        steps=0
        while pending or s.requests:
            for identity,sm,op,payload in list(pending):
                accepted=s.submit_atomic(identity,sm,payload) if op=='atomic' else s.submit(identity,sm,payload,operation=op,bypass_l1=True)
                if accepted:pending.remove((identity,sm,op,payload))
            candidates=[s.next_wakeup()]+[s.next_admission(sm) for sm in range(2)];future=[w for w in candidates if w is not None and w>s.cycle]
            if pending or s.requests:
                assert future,'Random atomic service deadlock';s.advance(min(future));steps+=1;assert steps<100000
        s.finish();r=report(s)
        rows=[{'sequence':identity,'sm':v['sm'],'issue_cycle':v['submit_cycle'],'complete_cycle':v['complete_cycle'],'memory':payloads[identity]} for identity,v in s.finished.items()]
        check('independent randomized RMW/cache/pin/resource audit '+str(seed),verify_global_atomics(r) and verify_service(r,rows,0,s.cycle))
    return checks

if __name__=='__main__':print('PASS global atomic references checks='+str(len(verify_global_atomic_components())))
