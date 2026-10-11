"""Hand-solved CTA release, memory-prefix, reuse and finite-port references."""
from dataclasses import replace
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from cta_barrier_timing import Phase
from sm_execution import Issue, Pipeline, TimingConfig, SectorRequest, simulate_launch
from sm_residency import profile_for, block_footprint, validate_estimated_cta
from sector_memory_service import MemoryConfig
from register_dependencies import LaneDependencies
from register_dependency_components import issue as register_issue

def verify_cta_timing():
    checks = []
    def check(name, condition):
        if not condition: raise AssertionError(name)
        checks.append(name)
    def reject(name, action):
        try: action()
        except ValueError: checks.append(name)
        else: raise AssertionError(name)
    profile = profile_for('nvidia/a100'); fp = block_footprint(profile, [64,1,1],16)
    timing = TimingConfig(1, {'memory':Pipeline(10,1,8),'parameter':Pipeline(5,1,8),
        'control':Pipeline(1,1,8),'alu':Pipeline(1,1,8)}, 'Hand-solved CTA service reference',
        cta_barrier=Pipeline(3,1,16,'sm'))
    def op(sequence, warp=0, pipeline='control', memory=None, block=(0,0,0)):
        return Issue(sequence,block,warp,pipeline,(),pipeline=='control',memory)
    def solve(issues, phases, config=timing, memory=None, origin=0, readiness=None):
        return simulate_launch(profile,fp,issues,config,origin,memory,phases=tuple(phases),barrier_readiness=readiness)
    def times(report): return {r['sequence']:(r['issue_cycle'],r['complete_cycle']) for r in report['issues']}
    stream = [op(1,pipeline='memory'),op(2),op(3,1),op(5),op(6,1)]
    phase = Phase(4,(0,0,0),0,0,(2,3),waiters=((2,5),(3,6)))
    result=solve(stream,[phase]); p=result['cta_barrier_service']['phases'][0]
    check('arrival controls do not consume the ten-cycle memory response',times(result)[2]==(1,2) and times(result)[3]==(0,1))
    check('CTA release waits every captured prior memory access',p['memory_prior_issues']==[1] and (p['ready_cycle'],p['start_cycle'],p['complete_cycle'])==(10,10,13))
    check('both waiters resume at release completion rather than final arrival',times(result)[5]==times(result)[6]==(13,14))
    check('resident lifetime includes waiter continuation',result['blocks'][0]['release_cycle']==14)
    staged=[replace(stream[0],memory=SectorRequest('store',(0,)))]+stream[1:]
    result=solve(staged,[phase],memory=MemoryConfig()); p=result['cta_barrier_service']['phases'][0]
    check('CTA release follows actual L2 store acknowledgement instead of fixed memory token',times(result)[1]==(0,18) and p['start_cycle']==18 and p['complete_cycle']==21)
    check('store visibility token gates waiters without waiting terminal DRAM writeback',times(result)[5]==(21,22) and result['end_cycle']==33)
    staged=[replace(staged[0],memory=SectorRequest('load',(0,)))]+staged[1:]
    result=solve(staged,[phase],memory=MemoryConfig())
    check('CTA release waits asynchronous read transmission',times(result)[1]==(0,21) and times(result)[5]==(24,25))
    result=solve([op(1,pipeline='parameter')]+stream[1:],[phase])
    check('prior parameter accesses remain explicit completion causes',result['cta_barrier_service']['phases'][0]['memory_prior_issues']==[1] and times(result)[5]==(8,9))
    exited=[op(1,1,pipeline='memory'),op(2),op(3,1),op(5)]
    result=solve(exited,[replace(phase,arrivals=(2,),exits=(3,),waiters=((2,5),))])
    check('whole-CTA exit causes retain the exited warp memory prefix',result['cta_barrier_service']['phases'][0]['memory_prior_issues']==[1] and times(result)[5]==(13,14))
    producer=[op(1),op(2,1),op(4,pipeline='memory'),op(5,1),op(6)]
    p=Phase(3,(0,0,0),1,0,(1,2),waiters=((2,5),))
    result=solve(producer,[p])
    check('a nonblocking producer continues without inheriting waiter acquire',times(result)[4]==(1,11) and times(result)[5]==(4,5))
    check('post-arrival memory is excluded from the released phase prefix',result['cta_barrier_service']['phases'][0]['memory_prior_issues']==[])
    reused=[op(1),op(2,1),op(4),op(5,1),op(7),op(8,1)]
    phases=[Phase(3,(0,0,0),1,0,(1,2)),Phase(6,(0,0,0),1,1,(4,5),waiters=((4,7),(5,8)))]
    result=solve(reused,phases)
    check('nonblocking next-generation arrivals wait the previous phase reset',times(result)[4]==times(result)[5]==(4,5))
    check('phase reuse preserves separate completion records',[(p['phase'],p['start_cycle'],p['complete_cycle']) for p in result['cta_barrier_service']['phases']]==[(0,1,4),(1,5,8)])
    result=solve([op(1),op(2,1)], [Phase(3,(0,0,0),0,0,(1,2))])
    check('a phase without waiters still holds block storage through release',result['instruction_end_cycle']==1 and result['execution_end_cycle']==result['blocks'][0]['release_cycle']==4)
    independent=[Phase(3,(0,0,0),1,0,(1,)),Phase(4,(0,0,0),2,0,(2,))]
    result=solve([op(1),op(2,1)],independent,replace(timing,cta_barrier=Pipeline(3,1,1,'sm')))
    check('finite release capacity serializes independent IDs',[(p['start_cycle'],p['complete_cycle']) for p in result['cta_barrier_service']['phases']]==[(1,4),(4,7)])
    result=solve([op(1),op(2,1)],independent,replace(timing,cta_barrier=Pipeline(3,5,16,'sm')))
    check('release initiation interval remains independent of capacity',[(p['start_cycle'],p['complete_cycle']) for p in result['cta_barrier_service']['phases']]==[(1,4),(6,9)])
    result=solve([op(1),op(2,1)],[Phase(3,(0,0,0),0,0,(1,2))],origin=10,readiness=[{'sm':0,'next_release_cycle':19}])
    check('prior-launch release port deadline gates an otherwise ready phase',result['cta_barrier_service']['phases'][0]['start_cycle']==19 and result['end_cycle']==22)
    for label, changed in [('missing cause',replace(phase,arrivals=(99,))),('borrowed exit warp',replace(phase,exits=(3,))),
        ('missing waiter continuation',replace(phase,waiters=((2,6),))),('late waiter gate',replace(phase,waiters=((2,2),))),
        ('skipped phase',replace(phase,generation=1)),('invalid ID',replace(phase,barrier=16)),
        ('boolean identity',replace(phase,identity=True)),('repeated arrival',replace(phase,arrivals=(2,2)))]:
        reject(label,lambda changed=changed:solve(stream,[changed]))
    reject('CTA release assumptions cannot be omitted',lambda:solve(stream,[phase],replace(timing,cta_barrier=None)))
    reject('CTA service cannot be partition scoped',lambda:solve(stream,[phase],replace(timing,cta_barrier=Pipeline(3,1,2))))
    reject('CTA port cannot name nonexistent SM',lambda:solve(stream,[phase],readiness=[{'sm':1,'next_release_cycle':0}]))
    # Actual action roles, not selected opcode names, control virtual writes.
    g=LaneDependencies(); g.issue(register_issue(1,defs=(0,4)))
    text='bar.red.popc.u32 %r0, 0, %p2;'
    arrival=g.issue(register_issue(2,defs=(0,),uses=(4,),instruction=text),cta={'type':'cta_barrier_arrive','issue':2,'execution_mask':3})
    check('reduction arrival reads vote operands without writing its result',all(e['kind']=='RAW' for e in arrival['edges']) and g.writers[((1,(0,0,0),0),('register',0,0),0)]==1)
    result=g.issue(register_issue(3,defs=(0,),uses=(4,),instruction=text),cta={'type':'cta_barrier_result','issue':3,'execution_mask':3})
    check('saved reduction collection writes once without rereading live predicate registers',any(e['kind']=='WAW' and e['source']==1 for e in result['edges']) and not any(e['kind']=='RAW' for e in result['edges']))
    g=LaneDependencies()
    parked=g.issue(register_issue(1,uses=(4,),instruction='bar.sync %r2;'),cta={'type':'cta_barrier_wait','issue':1,'execution_mask':3})
    check('convergence parking consumes no ID/count/vote operand before actual arrival',not parked['live_ins'] and not parked['edges'])
    reject('CTA operand action cannot borrow another parent',lambda:LaneDependencies().issue(register_issue(1,instruction='bar.sync 0;'),cta={'type':'cta_barrier_wait','issue':2,'execution_mask':3}))
    check('zero-ID barrier alias preserves the explicit PTX estimate subset',validate_estimated_cta('barrier.cta.sync.aligned 0;') is None)
    reject('named barrier aliases cannot bypass compiler resource evidence',lambda:validate_estimated_cta('barrier.cta.sync 5;'))
    reject('reduction aliases cannot imply allocated resource counts',lambda:validate_estimated_cta('barrier.red.popc.u32 %r0, 0, %p0;'))
    return checks

if __name__=='__main__': print('PASS CTA timing references checks='+str(len(verify_cta_timing())))
