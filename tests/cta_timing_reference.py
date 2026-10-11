"""Independent timestamp and causal-prefix audit; imports no timing service."""
from collections import defaultdict

def verify_cta_service(report, graph, launch, previous):
    service = launch['cta_barrier_service']
    expected = {p['release']:p for p in graph['cta_phases'] if p['launch']==launch['launch']}
    observed = {p['identity']:p for p in service['phases']}
    assert len(observed)==len(service['phases']) and set(observed)==set(expected), 'CTA phase conservation'
    records = {r['sequence']:r for r in launch['issues']}
    nodes = {n['sequence']:n for n in graph['nodes'] if n['launch']==launch['launch']}
    gates = defaultdict(list)
    for g in graph['cta_control_gates']:
        if g['release'] in expected: gates[g['release']].append(g['issue'])
    assert service['initial_port_readiness']==previous, 'Persistent CTA port ownership'
    initial = {r['sm']:r['next_release_cycle'] for r in previous or []}
    final = {r['sm']:r['next_release_cycle'] for r in service['next_port_readiness']}
    assert len(final)==len(service['next_port_readiness']), 'Unique CTA port deadline'
    starts, events = defaultdict(list), defaultdict(list)
    block_ends, peak = {}, defaultdict(int)
    prior = {}
    for identity in sorted(expected):
        phase, row = expected[identity], observed[identity]
        assert row['block']==phase['block'] and row['barrier']==phase['barrier'] and row['phase']==phase['phase'], 'CTA resource ownership'
        assert row['arrival_issues']==phase['arrival_issues'] and row['exit_issues']==phase['exit_issues'], 'CTA exact causes'
        key=(tuple(phase['block']),phase['barrier']); last=prior.get(key)
        assert row['previous_phase']==last, 'CTA generation reset cause'
        assert phase['phase']==(expected[last]['phase']+1 if last else 0), 'CTA phase sequence'
        prior[key]=identity
        needed=set()
        for source in phase['arrival_issues']+phase['exit_issues']:
            cause=nodes[source]
            assert cause['block']==phase['block'], 'CTA source block'
            for sequence,node in nodes.items():
                if sequence<source and node['block']==cause['block'] and node['warp']==cause['warp'] and node['memory_access']:
                    needed.add(sequence)
        assert row['memory_prior_issues']==sorted(needed), 'CTA prior-memory prefix conservation'
        causes=set(phase['arrival_issues']+phase['exit_issues'])|needed
        ready=max([launch['start_cycle']]+[records[s]['complete_cycle'] for s in causes]+([observed[last]['complete_cycle']] if last else []))
        assert row['ready_cycle']==ready, 'CTA exact completion readiness'
        cfg=report['config'].get('cta_barrier')
        assert cfg and cfg['scope']=='sm', 'Declared CTA service'
        begin,end=row['start_cycle'],row['complete_cycle']; sm=row['sm']
        assert type(begin) is int and type(end) is int and begin>=ready and end==begin+cfg['latency'], 'CTA release latency and memory readiness'
        assert begin>=initial.get(sm,launch['start_cycle']), 'Prior CTA port deadline'
        assert all(records[s]['sm']==sm for s in causes), 'Resident CTA release placement'
        starts[sm].append(begin); events[sm].extend([(begin,1),(end,-1)])
        block=tuple(phase['block']); block_ends[block]=max(block_ends.get(block,0),end)
        waiter_pairs=[]
        for p in phase['participants']:
            if not p['blocking']: continue
            own=sorted(s for s,n in nodes.items() if n['block']==phase['block'] and n['warp']==p['warp'] and s>p['issue'])
            assert own and own[0] in gates[identity], 'CTA next selected waiter gate'
            assert records[own[0]]['issue_cycle']>=end, 'CTA waiter release readiness'
            waiter_pairs.append([p['issue'],own[0]])
        assert row['waiter_gates']==waiter_pairs and set(gates[identity])=={p[1] for p in waiter_pairs}, 'CTA exact waiter ownership'
        if last:
            assert all(records[s]['issue_cycle']>=observed[last]['complete_cycle'] for s in phase['arrival_issues']), 'CTA reuse waits reset'
    for sm, values in starts.items():
        ordered=sorted(values); cfg=report['config']['cta_barrier']
        assert all(b-a>=cfg['interval'] for a,b in zip(ordered,ordered[1:])), 'CTA release initiation interval'
        active=0
        for _,delta in sorted(events[sm]):
            active+=delta; assert 0<=active<=cfg['capacity'], 'CTA finite release capacity'
            peak[sm]=max(peak[sm],active)
        assert active==0, 'CTA release drain'
        initial[sm]=max(values)+cfg['interval']
    assert final==initial, 'Next CTA release deadline'
    assert {int(sm):p for sm,p in service['peak_inflight_per_sm'].items()}==dict(peak), 'CTA in-flight peaks'
    assert all(v==0 for v in service['final_inflight_per_sm'].values()), 'CTA final resources'
    return block_ends
