"""Audit staged reduction operands directly against captured CTA children."""
import json

def verify_cta_dependencies(events, graph):
    issues={r['sequence']:r for r in events if r['type']=='issue'}
    nodes={n['sequence']:n for n in graph['nodes']}
    arrivals={r['sequence']:r for r in events if r['type']=='cta_barrier_arrive'}
    releases={a:r for r in events if r['type']=='cta_barrier_release' for a in r['arrivals']}
    actions={r['issue']:r for r in events if r['type'] in ['cta_barrier_wait','cta_barrier_arrive','cta_barrier_result']}
    encode=lambda rows:sorted(json.dumps(r,sort_keys=True) for r in rows)
    for sequence,node in nodes.items():
        action=actions.get(sequence)
        assert node.get('cta_action')==(action['type'] if action else None), 'Actual CTA operand stage'
        data=[e for e in node['edges'] if e['kind']=='CTA_REDUCTION']
        if not action or action['type']!='cta_barrier_result':
            assert not data and 'cta_result' not in node, 'No speculative CTA result'
        if not action: continue
        parent=issues[sequence]
        if action['type']=='cta_barrier_arrive':
            assert all(e['kind']=='RAW' for e in node['edges']), 'Arrival cannot write its deferred result'
        elif action['type']=='cta_barrier_wait':
            assert all(e['kind']=='RAW' and e['resource']==['register',parent['frame'],parent['predicate_use']] for e in node['edges']), 'Convergence wait cannot read barrier data operands'
            assert all(r['resource']==['register',parent['frame'],parent['predicate_use']] for r in node['live_ins']), 'Convergence wait cannot invent operand live-ins'
        else:
            release=releases[action['arrival']]
            assert node['cta_result']=={'release':release['sequence'],'result':release['result']}, 'Saved reduction phase/value'
            expected=[]
            for reference in release['arrivals']:
                arrival=arrivals[reference]
                for destination in parent['defs']:
                    expected.append({'source':arrival['issue'],'target':sequence,'kind':'CTA_REDUCTION',
                        'resource':['cta',release['barrier'],release['phase']],'lane_mask':arrival['execution_mask'],
                        'destination_resource':['register',parent['frame'],destination],
                        'destination_lane_mask':parent['execution_mask']})
            assert encode(data)==encode(expected), 'Exact reduction contributors and destination lanes'
            assert all(e['resource']==['register',parent['frame'],parent['predicate_use']] for e in node['edges'] if e['kind']=='RAW'), 'Saved result cannot reread its vote operands'
    return True
