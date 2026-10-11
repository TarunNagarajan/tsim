"""Independent scalar RMW byte conservation, word locks and pinned-line audit."""
from collections import Counter,defaultdict,deque


def verify_global_atomics(memory,start=0,end=None,rows=None):
    service=memory['atomic_service'];config=service['config'];base=memory['config']
    requests={r['identity']:r for r in service['requests']}
    outer={r['identity']:r for r in memory['requests'] if r['operation']=='atomic'}
    assert len(requests)==len(service['requests']) and set(requests)==set(outer), 'Atomic request conservation'
    if rows is not None:
        modeled={r['sequence']:r for r in rows if r.get('memory',{}).get('operation')=='atomic'}
        assert set(modeled)==set(requests), 'Atomic captured issue conservation'
        for identity,r in requests.items():assert list(r['accesses'])==list(modeled[identity]['memory']['accesses']), 'Atomic captured lane bytes'
    end=end if end is not None else memory['final_snapshot']['cycle']
    expected={};sectors=defaultdict(list)
    for identity,r in requests.items():
        assert r['sm']==outer[identity]['sm'] and r['submit_cycle']==outer[identity]['submit_cycle'], 'Atomic request ownership'
        accesses=sorted(r['accesses'],key=lambda a:a['lane'])
        assert len(accesses)<=config['max_updates_per_request'] and len({a['lane'] for a in accesses})==len(accesses), 'Scalar lane update conservation'
        for number,a in enumerate(accesses):
            assert type(a['width']) is int and a['width'] in [4,8] and type(a['address']) is int and a['address']>=0 and a['address']%a['width']==0 and a['element']==0, 'Atomic scalar byte geometry'
            token=(identity,number);words=list(range(a['address']//4,(a['address']+a['width']-1)//4+1));sector=a['address']//32
            expected[token]=dict(a,identity=identity,job=number,words=words,sector=sector);sectors[(identity,sector)].append(token)
        assert r['sectors']==outer[identity]['sectors']==sorted({a['address']//32 for a in accesses}), 'Atomic sector lookup fanout'
    jobs={(j['identity'],j['job']):j for j in service['jobs']}
    assert len(jobs)==len(service['jobs']) and set(jobs)==set(expected), 'No merged or invented RMW updates'
    lookups={(r['identity'],r['sector']):r for r in memory['lookup_records']}
    locks=defaultdict(list)
    for token,j in jobs.items():
        assert all(type(j[name]) is int for name in ['identity','job','lane','element','address','width','sector','ready_cycle','start_cycle','complete_cycle']) and all(type(w) is int for w in j['words']), 'Typed atomic job coordinates'
        for name,value in expected[token].items():assert j[name]==value, 'Exact atomic job bytes: '+name
        assert start<=requests[token[0]]['submit_cycle']<=j['ready_cycle']<=j['start_cycle']<j['complete_cycle']<=end, 'Atomic envelope'
        assert j['complete_cycle']==j['start_cycle']+config['latency'], 'Atomic RMW latency'
        lookup=lookups[(token[0],j['sector'])]
        if lookup['outcome']=='hit':assert j['ready_cycle']==lookup['lookup_complete_cycle'], 'Atomic completed hit readiness'
        else:
            assert lookup['outcome']=='miss' and j['ready_cycle']>=lookup['lookup_complete_cycle'], 'Atomic miss readiness'
            assert any(f['sector']==j['sector'] and f['cycle']==j['ready_cycle'] for f in service['fills']), 'Atomic successful fill readiness'
        for word in j['words']:locks[word].append((j['start_cycle'],j['complete_cycle']))
    for intervals in locks.values():
        intervals.sort();assert all(b[0]>=a[1] for a,b in zip(intervals,intervals[1:])), 'Overlapping mixed-width atomic exclusion'
    begins=sorted(j['start_cycle'] for j in jobs.values())
    assert not begins or begins[0]>=service['initial_port_readiness'], 'Prior atomic port deadline'
    assert all(b-a>=config['interval'] for a,b in zip(begins,begins[1:])), 'Global atomic initiation bandwidth'
    assert service['next_port_readiness']==(begins[-1]+config['interval'] if begins else service['initial_port_readiness']), 'Next atomic port deadline'
    for identity,r in outer.items():
        assert r['complete_cycle']==max(j['complete_cycle'] for token,j in jobs.items() if token[0]==identity)+base['return_latency'], 'Whole atomic response join'
    live=defaultdict(set);reservations={};intervals=defaultdict(list);queue=deque();peaks=Counter();previous=start;previous_order=0
    for e in service['resources']:
        cycle,ordinal,kind,delta=e['cycle'],e['ordinal'],e['resource'],e['delta'];token=tuple(e['token'])
        assert type(cycle) is int and previous<=cycle<=end and type(ordinal) is int and ordinal>previous_order and type(delta) is int and delta in [-1,1], 'Atomic resource event order'
        assert kind in ['upstream','queue','slots','pins'] and (token in sectors if kind=='pins' else token in jobs), 'Atomic ledger ownership'
        key=(kind,token)
        if delta==1:
            assert token not in live[kind] and key not in reservations, 'Duplicate atomic reservation'
            live[kind].add(token);reservations[key]=(cycle,ordinal)
            if kind=='queue':queue.append(token)
        else:
            assert token in live[kind], 'Atomic release without reservation'
            live[kind].remove(token);intervals[key].append((reservations.pop(key),(cycle,ordinal)))
            if kind=='queue':assert queue.popleft()==token, 'Atomic global FIFO'
        cap={'queue':config['queue'],'slots':config['slots'],'upstream':base['sm_count']*base['lsu_slots_per_sm']*config['max_updates_per_request'],
             'pins':base['sm_count']*base['lsu_slots_per_sm']*base['max_sectors_per_request']}[kind]
        assert len(live[kind])<=cap, 'Atomic finite '+kind+' storage'
        peaks[kind]=max(peaks[kind],len(live[kind]));previous=cycle;previous_order=ordinal
    assert not reservations and all(not tokens for tokens in live.values()) and not queue, 'Atomic resource drain'
    for token,j in jobs.items():
        upstream=intervals[('upstream',token)];queued=intervals[('queue',token)];active=intervals[('slots',token)]
        assert len(upstream)==len(queued)==len(active)==1, 'Atomic token lifetime conservation'
        assert upstream[0][0][0]==requests[token[0]]['submit_cycle'] and j['ready_cycle']<=upstream[0][1][0], 'Atomic upstream arrival lifetime'
        assert queued[0][0][0]==upstream[0][1][0] and queued[0][1][0]==j['start_cycle'], 'Atomic queue handoff'
        assert active[0][0][0]==j['start_cycle'] and active[0][1][0]==j['complete_cycle'], 'Atomic active slot lifetime'
    pin_lifetimes=[]
    for pair,tokens in sectors.items():
        pins=intervals[('pins',pair)];assert len(pins)==1, 'Atomic sector pin conservation'
        assert len({jobs[t]['ready_cycle'] for t in tokens})==1, 'One availability per atomic sector'
        assert pins[0][0][0]==jobs[tokens[0]]['ready_cycle'] and pins[0][1][0]==max(jobs[t]['complete_cycle'] for t in tokens), 'Atomic pinned sector lifetime'
        pin_lifetimes.append((pair[1]//4,*pins[0]))
    assert len(intervals)==3*len(jobs)+len(sectors), 'No invented atomic ledger tokens'
    fills=service['fills'];assert len({f['ordinal'] for f in fills})==len(fills) and not ({f['ordinal'] for f in fills}&{e['ordinal'] for e in service['resources']}), 'Atomic cache action ordering'
    for f in fills:
        assert type(f['cycle']) is int and type(f['ordinal']) is int and type(f['sector']) is int, 'Typed cache fill coordinates'
        assert start<=f['cycle']<=end and any(d['sector']==f['sector'] and d['complete_cycle']==f['cycle'] or
            d['sector']==f['sector'] and d['complete_cycle']<=f['cycle'] for d in memory['dram_reads']), 'Atomic cache fill after backing'
        if f['victim_line'] is not None:
            assert all(not (line==f['victim_line'] and begin<=(f['cycle'],f['ordinal'])<finish) for line,begin,finish in pin_lifetimes), 'Pinned atomic line cannot be evicted'
    for kind in ['queue','slots','pins']:assert peaks[kind]==service['peaks'].get(kind,0), 'Independent atomic peak: '+kind
    assert service['stats'].get('accepted_requests',0)==len(requests) and service['stats'].get('updates',0)==service['stats'].get('committed_updates',0)==len(jobs), 'Atomic statistics conservation'
    assert service['drained'] is True, 'Atomic drained evidence'
    return True
