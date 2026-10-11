"""Hand-solved global atomic retirement, common LSU joins and CTA prefixes."""
from dataclasses import replace
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from global_atomic_service import AtomicAccess,GlobalAtomicRequest,AtomicConfig
from sector_memory_service import MemoryConfig
from memory_system import CompositeRequest
from shared_memory_service import SharedAccess,SharedRequest,SharedConfig
from sm_execution import Issue,Pipeline,TimingConfig,simulate_launch
from sm_residency import profile_for,block_footprint
from cta_barrier_timing import Phase
from memory_system_reference import verify_memory_system
from sector_service_reference import verify_service


def verify_sm_atomics():
    checks=[]
    def check(name,condition):
        if not condition:raise AssertionError(name)
        checks.append(name)
    def reject(name,action):
        try:action()
        except ValueError:checks.append(name)
        else:raise AssertionError(name)
    profile=profile_for('nvidia/a100');fp=block_footprint(profile,[32,1,1],16)
    timing=TimingConfig(1,{'memory':Pipeline(999,1,8),'alu':Pipeline(4,1,4),'control':Pipeline(1,1,4)},'Independent scalar RMW retirement references',cta_barrier=Pipeline(3,1,16,'sm'))
    def request(lanes=1):return GlobalAtomicRequest(tuple(AtomicAccess(lane,0,0,4) for lane in range(lanes)))
    def mem(n,req=None,warp=0):return Issue(n,(0,0,0),warp,'memory',memory=req or request())
    def op(n,deps=(),warp=0,control=False):return Issue(n,(0,0,0),warp,'control' if control else 'alu',deps,control)
    def solve(stream,atomic=None,memory=None,shared=None,phases=(),fp=fp,origin=0,readiness=None):
        r=simulate_launch(profile,fp,stream,timing,origin,memory or MemoryConfig(),phases=phases,
            atomic_config=atomic or AtomicConfig(),shared_config=shared,system_readiness=readiness)
        rows=verify_memory_system(r['memory_system'],r['issues'],r['memory_service'],r['shared_service'],origin,r['end_cycle'],readiness)
        assert verify_service(r['memory_service'],rows,origin,r['end_cycle']);return r
    def times(r):return {v['sequence']:(v['issue_cycle'],v['complete_cycle']) for v in r['issues']}
    r=solve([mem(1),op(2,((1,'RAW'),))]);check('atomic RAW waits RMW and response transmission',times(r)=={1:(0,22),2:(22,26)})
    check('atomic backing drain follows retirement without prolonging resident lifetime',r['blocks'][0]['release_cycle']==26 and r['end_cycle']==37)
    for kind in ['WAR','WAW']:
        r=solve([mem(1),op(2,((1,kind),))]);check('global RMW '+kind+' stage readiness',times(r)[2]==((1,5) if kind=='WAR' else (22,26)))
    r=solve([mem(1,request(32)),op(2,control=True)]);check('same-word atomic lane tail survives early warp return',times(r)=={1:(0,146),2:(1,2)} and r['blocks'][0]['release_cycle']==146)
    r=solve([mem(1),mem(2)],memory=MemoryConfig(lsu_slots_per_sm=1));check('common LSU slot persists until last atomic response',times(r)=={1:(0,22),2:(22,33)} and r['memory_system']['peak_requests']==1)
    mixed=CompositeRequest(request(),SharedRequest('atomic',(0,0,0),tuple(SharedAccess(lane,0,0,4) for lane in range(32))))
    r=solve([mem(1,mixed),op(2,((1,'RAW'),))],shared=SharedConfig())
    check('mixed global/shared atomics retain separate updates and joint response',r['memory_system']['requests'][0]['fragments']=={'global':22,'shared':131} and times(r)[2]==(131,135))
    two=block_footprint(profile,[64,1,1],16)
    phase=Phase(4,(0,0,0),0,0,(2,3),waiters=((2,5),(3,6)))
    r=solve([mem(1,request(32)),op(2,control=True),op(3,warp=1,control=True),op(5,control=True),op(6,warp=1,control=True)],phases=(phase,),fp=two)
    check('CTA phase prefix waits all global atomic updates and transmission',r['cta_barrier_service']['phases'][0]['start_cycle']==146 and times(r)[5]==times(r)[6]==(149,150))
    slow=AtomicConfig(interval=100);first=solve([mem(1)],atomic=slow);second=solve([mem(2)],atomic=slow,origin=first['end_cycle'],readiness=first['memory_system']['next_port_readiness'])
    check('SM preserves atomic deadline across launches and cold cache fills',times(second)=={2:(33,122)} and second['memory_service']['atomic_service']['jobs'][0]['start_cycle']==116)
    reject('global RMW cannot use ordinary sector service without atomic assumptions',lambda:simulate_launch(profile,fp,[mem(1)],timing,memory_config=MemoryConfig()))
    reject('atomic configuration requires explicit global service',lambda:simulate_launch(profile,fp,[mem(1)],timing,atomic_config=AtomicConfig()))
    reject('atomic operation cannot use fixed tokens without captured updates',lambda:solve([Issue(1,(0,0,0),0,'memory')]))
    return checks

if __name__=='__main__':print('PASS SM atomic references checks='+str(len(verify_sm_atomics())))
