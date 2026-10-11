"""Hand-solved common LSU, mixed responses, shared bank tails and CTA gates."""
from dataclasses import asdict,replace
from copy import deepcopy
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from memory_system import MemorySystem,SectorRequest,CompositeRequest
from shared_memory_service import SharedRequest,SharedAccess,SharedConfig
from sector_memory_service import MemoryConfig
from sm_execution import Issue,Pipeline,TimingConfig,simulate_launch
from sm_residency import profile_for,block_footprint
from cta_barrier_timing import Phase
from memory_system_reference import verify_memory_system
from sector_service_reference import verify_service


def verify_sm_shared():
    checks=[]
    def check(name,condition):
        if not condition:raise AssertionError(name)
        checks.append(name)
    def reject(name,action):
        try:action()
        except ValueError:checks.append(name)
        else:raise AssertionError(name)
    profile=profile_for('nvidia/a100');fp=block_footprint(profile,[32,1,1],16)
    timing=TimingConfig(1,{'memory':Pipeline(999,1,8),'alu':Pipeline(4,1,4),'control':Pipeline(1,1,4)},
        'Independent shared and composite completion references',cta_barrier=Pipeline(3,1,16,'sm'))
    def shared(stride=1,operation='load',lanes=32,block=(0,0,0)):
        return SharedRequest(operation,block,tuple(SharedAccess(lane,0,lane*stride*4,4) for lane in range(lanes)))
    def mem(n,request,warp=0,block=(0,0,0)):
        return Issue(n,block,warp,'memory',memory=request)
    def op(n,deps=(),warp=0,control=False):return Issue(n,(0,0,0),warp,'control' if control else 'alu',deps,control)
    def solve(issues,shared_config=None,memory_config=None,phases=(),fp=fp,origin=0,readiness=None):
        r=simulate_launch(profile,fp,issues,timing,origin,memory_config or MemoryConfig(),phases=phases,
            shared_config=shared_config or SharedConfig(),system_readiness=readiness)
        global_rows=verify_memory_system(r['memory_system'],r['issues'],r['memory_service'],r['shared_service'],origin,r['end_cycle'],readiness)
        assert verify_service(r['memory_service'],global_rows,origin,r['end_cycle']);return r
    def times(r):return {row['sequence']:(row['issue_cycle'],row['complete_cycle']) for row in r['issues']}
    r=solve([mem(1,shared()),op(2,((1,'RAW'),))]);check('shared RAW uses actual word response instead of fixed token',times(r)=={1:(0,5),2:(5,9)})
    r=solve([mem(1,shared(32)),op(2,control=True)]);check('shared conflict tail holds block after early return',times(r)=={1:(0,36),2:(1,2)} and r['blocks'][0]['release_cycle']==36)
    for kind in ['WAW','WAR']:
        r=solve([mem(1,shared()),op(2,((1,kind),))]);check('shared '+kind+' uses correct write/read stage',times(r)[2]==((5,9) if kind=='WAW' else (1,5)))
    composite=CompositeRequest(SectorRequest('load',(0,)),shared(32))
    r=solve([mem(1,composite),op(2,((1,'RAW'),))]);check('mixed instruction waits shared tail after early global response',r['memory_system']['requests'][0]['fragments']=={'global':21,'shared':36} and times(r)[2]==(36,40))
    r=solve([mem(1,replace(composite,shared_fragment=shared())),op(2,((1,'RAW'),))]);check('mixed instruction waits global tail after early shared response',r['memory_system']['requests'][0]['fragments']=={'global':21,'shared':5} and times(r)[2]==(21,25))
    one=MemoryConfig(lsu_slots_per_sm=1)
    r=solve([mem(1,composite),mem(2,SectorRequest('load',(0,)))],memory_config=one)
    check('freed global component slot cannot bypass the occupied common LSU',times(r)=={1:(0,36),2:(36,39)} and r['memory_system']['peak_requests']==1)
    two=block_footprint(profile,[64,1,1],16)
    issues=[mem(1,shared(32,'store')),op(2,control=True),op(3,warp=1,control=True),op(5,control=True),op(6,warp=1,control=True)]
    phase=Phase(4,(0,0,0),0,0,(2,3),waiters=((2,5),(3,6)))
    r=solve(issues,phases=(phase,),fp=two)
    check('actual shared response joins CTA memory prefix and both waiter gates',r['cta_barrier_service']['phases'][0]['start_cycle']==36 and times(r)[5]==times(r)[6]==(39,40))
    atomic=solve([mem(1,shared(0,'atomic')),op(2,control=True)])
    check('shared atomic RMW serialization affects instruction and resident tails',times(atomic)[1]==(0,131) and atomic['blocks'][0]['release_cycle']==131)
    slow=SharedConfig(request_interval=100,bank_interval=120)
    first=solve([mem(1,shared(lanes=1))],slow);second=solve([mem(2,shared(lanes=1))],slow,origin=first['end_cycle'],readiness=first['memory_system']['next_port_readiness'])
    check('common and shared bank port deadlines persist separately across launches',times(second)=={2:(100,125)} and second['shared_service']['jobs'][0]['start_cycle']==121)
    system=MemorySystem(MemoryConfig(),SharedConfig(request_slots_per_sm=1));assert system.submit(1,0,shared())
    system.advance(1);check('mixed admission preflight leaves all global state untouched on shared pressure',not system.submit(2,0,composite) and not system.global_service.requests and not system.global_service.stats['accepted_requests'])
    reject('malformed mixed fragment rejected before any route mutates',lambda:system.submit(3,0,CompositeRequest(SectorRequest('load',(7,)),SharedRequest('load',(0,0,0),(SharedAccess(0,0,2,4),)))))
    check('rejected typed fragment has no partial global reservation',not system.global_service.requests)
    reject('shared request owner differs from scheduled CTA',lambda:solve([mem(1,shared(block=(1,0,0)))]))
    reject('shared count must agree with scheduler',lambda:solve([mem(1,shared())],SharedConfig(sm_count=2)))
    for label,mutation in [('early common response',lambda s:s['requests'][0].update(complete_cycle=21)),
        ('missing fragment',lambda s:s['requests'][0]['fragments'].pop('shared')),
        ('missing common release',lambda s:s['resource_records'].pop()),
        ('invented common peak',lambda s:s.update(peak_requests=0)),
        ('erased common deadline',lambda s:s['next_port_readiness']['lsu'].__setitem__(0,0))]:
        damaged=deepcopy(r['memory_system']);mutation(damaged)
        try:verify_memory_system(damaged,r['issues'],r['memory_service'],r['shared_service'],0,r['end_cycle'])
        except AssertionError:checks.append('independent common audit rejects '+label)
        else:raise AssertionError('Accepted '+label)
    return checks

if __name__=='__main__':print('PASS shared SM references checks='+str(len(verify_sm_shared())))
