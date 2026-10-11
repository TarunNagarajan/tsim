"""Independent byte, queue, bank, atomic-lock and response audit.

No production topology builder or service is imported. Expected traffic comes
from the recorded lane byte spans; reservation histories prove finite storage.
"""
from collections import Counter, defaultdict, deque


def verify_shared_service(service, rows=None, start=0, end=None):
    config=service['config']; count=config['sm_count']
    requests={r['identity']:r for r in service['requests']}
    assert len(requests)==len(service['requests']), 'Unique shared requests'
    if rows is not None:
        modeled={r['sequence']:r for r in rows if r.get('memory')}
        assert set(modeled)==set(requests), 'Shared issue conservation'
        for identity,r in requests.items():
            row=modeled[identity]
            assert (r['sm'],r['submit_cycle'],r['complete_cycle'])==(row['sm'],row['issue_cycle'],row['complete_cycle']), 'Shared response readiness'
            assert r['request']==row['memory'], 'Shared captured spans'
    end=end if end is not None else max((r['complete_cycle'] for r in requests.values()),default=start)
    expected={}; total_bytes=0
    for identity,r in requests.items():
        assert type(identity) is int and identity>0 and type(r['sm']) is int and 0<=r['sm']<count, 'Shared request identity'
        assert start<=r['submit_cycle']<r['complete_cycle']<=end, 'Shared request envelope'
        assert r['front_ready_cycle']==r['submit_cycle']+config['front_latency'], 'Shared front latency'
        req=r['request']; assert req['operation'] in ['load','store','atomic'], 'Shared operation'
        spans=sorted(req['accesses'],key=lambda a:(a['lane'],a['element']))
        assert len({(a['lane'],a['element']) for a in spans})==len(spans), 'Unique shared elements'
        assert sum(a['width'] for a in spans)<=config['max_request_bytes'], 'Shared request byte bound'
        words={}; jobs=[]
        for a in spans:
            assert type(a['offset']) is int and type(a['width']) is int and a['offset']>=0 and a['width'] in [1,2,4,8,16], 'Shared span geometry'
            total_bytes+=a['width']; pieces=[]
            for word in range(a['offset']//4,(a['offset']+a['width']-1)//4+1):
                mask=sum(1<<(byte%4) for byte in range(a['offset'],a['offset']+a['width']) if byte//4==word)
                owner={'lane':a['lane'],'element':a['element'],'byte_mask':mask}
                pieces.append((word,owner)); words.setdefault(word,[]).append(owner)
            if req['operation']=='atomic': jobs.append(([word for word,_ in pieces],[owner for _,owner in pieces]))
        if req['operation']!='atomic': jobs=[([word],owners) for word,owners in sorted(words.items())]
        for number,(word_ids,owners) in enumerate(jobs):
            expected[(identity,number)]={'words':word_ids,'owners':owners,'banks':sorted({w%32 for w in word_ids}),
                'atomic':req['operation']=='atomic','block':list(req['block']),'sm':r['sm']}
    jobs={(j['identity'],j['job']):j for j in service['jobs']}
    assert len(jobs)==len(service['jobs']) and set(jobs)==set(expected), 'Shared logical job conservation'
    bank_starts=defaultdict(list); locks=defaultdict(list)
    for token,j in jobs.items():
        for key,value in expected[token].items(): assert j[key]==value, 'Shared job byte/bank ownership: '+key
        r=requests[token[0]]
        assert j['ready_cycle']==r['front_ready_cycle'] and j['start_cycle']>=j['ready_cycle'], 'Shared bank arrival readiness'
        latency=config['atomic_latency'] if j['atomic'] else config['bank_latency']
        assert j['complete_cycle']==j['start_cycle']+latency, 'Shared bank service latency'
        for bank in j['banks']: bank_starts[(j['sm'],bank)].append(j['start_cycle'])
        if j['atomic']:
            for word in j['words']:locks[(j['sm'],tuple(j['block']),word)].append((j['start_cycle'],j['complete_cycle']))
    for intervals in locks.values():
        intervals.sort(); assert all(b[0]>=a[1] for a,b in zip(intervals,intervals[1:])), 'Atomic word lock lifetime'
    initial=service['initial_port_readiness'] or {'request':[0]*count,'bank':[[0]*32 for _ in range(count)]}
    following=service['next_port_readiness']
    for sm in range(count):
        starts=sorted(r['submit_cycle'] for r in requests.values() if r['sm']==sm)
        assert not starts or starts[0]>=initial['request'][sm], 'Prior shared ingress deadline'
        assert all(b-a>=config['request_interval'] for a,b in zip(starts,starts[1:])), 'Shared ingress bandwidth'
        assert following['request'][sm]==(starts[-1]+config['request_interval'] if starts else initial['request'][sm]), 'Next shared ingress deadline'
        for bank in range(32):
            starts=sorted(bank_starts[(sm,bank)])
            assert not starts or starts[0]>=initial['bank'][sm][bank], 'Prior shared bank deadline'
            assert all(b-a>=config['bank_interval'] for a,b in zip(starts,starts[1:])), 'Shared bank initiation bandwidth'
            assert following['bank'][sm][bank]==(starts[-1]+config['bank_interval'] if starts else initial['bank'][sm][bank]), 'Next shared bank deadline'
    for identity,r in requests.items():
        assert r['complete_cycle']==max(j['complete_cycle'] for token,j in jobs.items() if token[0]==identity)+config['return_latency'], 'Whole shared response join'
    live=defaultdict(set); begins={}; intervals=defaultdict(list); queues=defaultdict(deque); peaks=Counter(); previous=start
    for e in service['resource_records']:
        cycle,kind,sm,bank,delta=e['cycle'],e['resource'],e['sm'],e['bank'],e['delta']; token=tuple(e['token'])
        assert type(cycle) is int and previous<=cycle<=end and type(delta) is int and delta in [-1,1], 'Shared resource event order'
        assert type(sm) is int and 0<=sm<count and kind in ['upstream','queue','bank','requests'], 'Shared resource scope'
        assert (type(bank) is int and 0<=bank<32) if kind in ['queue','bank'] else bank==-1, 'Shared resource bank scope'
        key=(kind,sm,bank); interval_key=(key,token)
        if kind=='requests':
            assert len(token)==1 and token[0] in requests and requests[token[0]]['sm']==sm, 'Shared request ledger owner'
        else:
            assert token in jobs and jobs[token]['sm']==sm, 'Shared job ledger owner'
            if bank!=-1:assert bank in jobs[token]['banks'], 'Shared job ledger bank'
        if delta==1:
            assert token not in live[key], 'Duplicate shared reservation'
            live[key].add(token); begins[interval_key]=cycle
            if kind=='queue': queues[(sm,bank)].append(token)
        else:
            assert token in live[key], 'Shared release without reservation'
            live[key].remove(token); intervals[interval_key].append((begins.pop(interval_key),cycle))
            if kind=='queue':assert queues[(sm,bank)].popleft()==token, 'Shared bank FIFO'
        cap={'requests':config['request_slots_per_sm'],'queue':config['bank_queue'],'bank':config['bank_slots'],
             'upstream':count*config['request_slots_per_sm']*config['max_request_bytes']}[kind]
        assert len(live[key])<=cap, 'Shared finite capacity: '+kind
        peaks['requests']=max(peaks['requests'],sum(len(v) for (k,_,_),v in live.items() if k=='requests'))
        if kind in ['queue','bank']:peaks['bank_queue' if kind=='queue' else 'bank_slots']=max(peaks['bank_queue' if kind=='queue' else 'bank_slots'],len(live[key]))
        previous=cycle
    assert not begins and all(not v for v in live.values()) and all(not q for q in queues.values()), 'Shared resource drain'
    for identity,r in requests.items():
        assert intervals[(('requests',r['sm'],-1),(identity,))]==[(r['submit_cycle'],r['complete_cycle'])], 'Shared response storage lifetime'
    for token,j in jobs.items():
        sm=j['sm']; r=requests[token[0]]; upstream=intervals[(('upstream',sm,-1),token)]
        assert len(upstream)==1 and upstream[0][0]==r['submit_cycle'] and j['ready_cycle']<=upstream[0][1]<=j['start_cycle'], 'Shared upstream lifetime'
        for bank in j['banks']:
            assert intervals[(('queue',sm,bank),token)]==[(upstream[0][1],j['start_cycle'])], 'Shared queue handoff lifetime'
            assert intervals[(('bank',sm,bank),token)]==[(j['start_cycle'],j['complete_cycle'])], 'Shared active bank lifetime'
    assert len(intervals)==len(requests)+sum(1+2*len(j['banks']) for j in jobs.values()), 'No invented shared ledger traffic'
    for key in ['requests','bank_queue','bank_slots']: assert peaks[key]==service['peaks'].get(key,0), 'Independent shared peak: '+key
    assert service['stats'].get('accepted_requests',0)==len(requests) and service['stats'].get('logical_word_jobs',0)==len(jobs), 'Shared statistics conservation'
    assert service['drained'] is True, 'Shared drained claim'
    return True
