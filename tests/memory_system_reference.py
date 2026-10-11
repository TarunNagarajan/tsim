"""Independent common LSU reservation and fragment-response join audit."""
from collections import defaultdict
from shared_service_reference import verify_shared_service
from global_atomic_reference import verify_global_atomics


def verify_captured_memory(events,report):
    """Compare routed issue payloads directly with original per-lane captures."""
    captured=defaultdict(list);parents={e['sequence']:e for e in events if e['type']=='issue'}
    for event in events:
        if event['type']=='memory' and event['space']!='param':captured[event['issue']].append(event)
    issues={r['sequence']:r for launch in report['launches'] for r in launch['issues'] if r.get('memory')}
    assert set(captured)==set(issues), 'Captured memory issue conservation'
    for identity,row in issues.items():
        payload=row['memory'];fragments=({'global':payload['global_fragment'],'shared':payload['shared_fragment']} if 'global_fragment' in payload else
            {'global':payload} if 'sectors' in payload else {'shared':payload})
        assert set(fragments)=={e['space'] for e in captured[identity]}, 'Captured route conservation'
        for name,fragment in fragments.items():
            accesses=[e for e in captured[identity] if e['space']==name]
            assert all(e['operation']==fragment['operation'] for e in accesses) and parents[identity]['block']==row['block'], 'Captured route operation and CTA'
            if name=='global':
                expected={byte//32 for e in accesses for byte in range(int(e['address'],16),int(e['address'],16)+e['bytes'])}
                assert list(fragment['sectors'])==sorted(expected), 'Captured global byte sectors'
                if fragment['operation']=='atomic':
                    expected=sorted((e['lane'],e['element'],int(e['address'],16),e['bytes']) for e in accesses)
                    actual=sorted((a['lane'],a['element'],a['address'],a['width']) for a in fragment['accesses'])
                    assert actual==expected, 'Captured global scalar RMW bytes'
            else:
                expected=sorted((e['lane'],e['element'],int(e['address'],16)-0x6ffe00000000,e['bytes']) for e in accesses)
                actual=sorted((a['lane'],a['element'],a['offset'],a['width']) for a in fragment['accesses'])
                assert expected==actual and list(fragment['block'])==row['block'], 'Captured shared byte spans'
    return True


def verify_memory_system(system,rows,global_service,shared_service,start,end,previous=None):
    requests={r['identity']:r for r in system['requests']};issues={r['sequence']:r for r in rows if r.get('memory')}
    assert len(requests)==len(system['requests']) and set(requests)==set(issues), 'Common issue conservation'
    assert system['initial_port_readiness']==previous, 'Common port continuity'
    global_requests={r['identity']:r for r in global_service['requests']} if global_service else {}
    shared_requests={r['identity']:r for r in shared_service['requests']} if shared_service else {}
    global_rows=[];shared_rows=[];starts=defaultdict(list)
    for identity,r in requests.items():
        row=issues[identity];memory=row['memory']
        if 'global_fragment' in memory:fragments={'global':memory['global_fragment'],'shared':memory['shared_fragment']}
        elif 'sectors' in memory:fragments={'global':memory}
        else:fragments={'shared':memory}
        assert len(r['routes'])==len(fragments) and set(r['routes'])==set(fragments)==set(r['fragments']), 'Whole memory routing'
        assert r['sm']==row['sm'] and r['submit_cycle']==row['issue_cycle'] and r['complete_cycle']==row['complete_cycle'], 'Common response readiness'
        assert r['complete_cycle']==max(r['fragments'].values()), 'Common last-fragment join'
        intervals=[]
        for name,payload in fragments.items():
            provider=global_requests if name=='global' else shared_requests
            component=provider[identity]
            assert (component['sm'],component['submit_cycle'],component['complete_cycle'])==(r['sm'],r['submit_cycle'],r['fragments'][name]), 'Indivisible route admission'
            target=global_rows if name=='global' else shared_rows
            target.append(dict(row,memory=payload,complete_cycle=r['fragments'][name]))
            intervals.append(global_service['config']['lsu_interval'] if name=='global' else shared_service['config']['request_interval'])
        assert r['interval']==max(intervals), 'Common participating route interval'
        starts[r['sm']].append(r)
    assert set(global_requests)=={r['sequence'] for r in global_rows} and set(shared_requests)=={r['sequence'] for r in shared_rows}, 'No invented route fragments'
    if shared_service:assert verify_shared_service(shared_service,shared_rows,start,end)
    if global_service and global_service.get('atomic_service'):assert verify_global_atomics(global_service,start,end,global_rows)
    count=(global_service or shared_service)['config']['sm_count']
    capacity=(global_service['config']['lsu_slots_per_sm'] if global_service else shared_service['config']['request_slots_per_sm'])
    assert system['capacity_per_sm']==capacity, 'Common capacity assumption'
    initial=previous['lsu'] if previous else [0]*count;following=system['next_port_readiness']
    assert following['global']==(global_service['next_port_readiness'] if global_service else None), 'Common global port state'
    assert following['shared']==(shared_service['next_port_readiness'] if shared_service else None), 'Common shared port state'
    for sm in range(count):
        own=sorted(starts[sm],key=lambda r:r['submit_cycle'])
        assert not own or own[0]['submit_cycle']>=initial[sm], 'Prior common LSU deadline'
        assert all(b['submit_cycle']>=a['submit_cycle']+a['interval'] for a,b in zip(own,own[1:])), 'Common LSU bandwidth'
        assert following['lsu'][sm]==(own[-1]['submit_cycle']+own[-1]['interval'] if own else initial[sm]), 'Next common LSU deadline'
    live=defaultdict(set);reservations={};finished={};peak=0;previous_cycle=start
    for e in system['resource_records']:
        cycle,sm,identity,delta=e['cycle'],e['sm'],e['identity'],e['delta']
        assert type(cycle) is int and previous_cycle<=cycle<=end and type(sm) is int and 0<=sm<count and type(delta) is int and delta in [-1,1], 'Common resource event'
        assert identity in requests and requests[identity]['sm']==sm, 'Common token ownership'
        if delta==1:
            assert identity not in live[sm] and identity not in reservations, 'Duplicate common reservation'
            live[sm].add(identity);reservations[identity]=cycle
        else:
            assert identity in live[sm] and identity not in finished, 'Common release ownership'
            live[sm].remove(identity);finished[identity]=cycle
        assert len(live[sm])<=capacity, 'Common LSU finite capacity'
        peak=max(peak,sum(len(v) for v in live.values()));previous_cycle=cycle
    assert all(not v for v in live.values()) and set(reservations)==set(finished)==set(requests), 'Common ledger conservation'
    assert all((reservations[i],finished[i])==(r['submit_cycle'],r['complete_cycle']) for i,r in requests.items()), 'Common slot survives all fragments'
    assert system['peak_requests']==peak and system['stats'].get('accepted_requests',0)==len(requests) and system['drained'] is True, 'Common resource summary'
    return global_rows
