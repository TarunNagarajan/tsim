"""Independent cross-lane version, routing and atomic validation references."""
from register_dependency_components import issue
from register_dependencies import LaneDependencies

def verify_collective_dependencies():
    checks = []
    def check(name, condition):
        if not condition: raise AssertionError(name)
        checks.append(name)
    def reject(name, action):
        try: action()
        except ValueError: checks.append(name)
        else: raise AssertionError(name)
    def bundle(sequence, source=0, destinations=(0,), maps=(2, 1), kind='shfl', mask=3):
        group = dict(type='collective_group', sequence=sequence + 1, issue=sequence,
                     kind=kind, execution_mask=mask, data_use=source, data_defs=list(destinations))
        return [group] + [dict(type='collective_transfer', sequence=sequence + 2 + lane, issue=sequence,
                              group=sequence + 1, destination_lane=lane, source_mask=origin)
                          for lane, origin in enumerate(maps)]
    def routed(node):
        return {(edge['source'], tuple(edge['resource']), edge['lane_mask'],
                 tuple(edge['destination_resource']), edge['destination_lane_mask'])
                for edge in node['edges'] if 'destination_resource' in edge}
    shuffle = issue(10, defs=[0, 2], uses=[0], instruction='shfl.sync.bfly.b32 r0|p0,r0,1,31,3')
    model = LaneDependencies(); model.issue(issue(1, defs=[0], mask=1)); model.issue(issue(2, defs=[0], mask=2))
    node = model.issue(shuffle, bundle(10))
    check('in-place swap reads both previous lane versions', routed(node) == {
        (2, ('register', 0, 0), 2, ('register', 0, 0), 1),
        (1, ('register', 0, 0), 1, ('register', 0, 0), 2)})
    check('shuffle predicate excludes transferred data dependencies', all(edge['destination_resource'][-1] == 0
          for edge in node['edges'] if 'destination_resource' in edge))
    check('collective snapshot never depends on own writes', all(edge['source'] != 10 for edge in node['edges']))
    after = model.issue(issue(20, uses=[0, 2]))
    check('collective data and predicate results publish actual writers',
          len(after['edges']) == 2 and all(edge['source'] == 10 and edge['lane_mask'] == 3 for edge in after['edges']))
    model = LaneDependencies(); model.issue(issue(1, defs=[0]))
    model.issue(issue(10, defs=[4], uses=[0], instruction='shfl.sync.idx.b32 r2,r0,0,31,3'), bundle(10, destinations=(4,), maps=(1, 1)))
    overwrite = model.issue(issue(20, defs=[0]))
    check('collective anti-dependency covers only actual source lanes',
          [(edge['source'], edge['lane_mask']) for edge in overwrite['edges'] if edge['kind'] == 'WAR'] == [(10, 1)])
    model = LaneDependencies(); model.issue(issue(1, defs=[0]))
    alias = model.issue(issue(10, defs=[4], uses=[0, 0], instruction='shfl.sync.idx.b32 r2,r0,r0,31,3'),
                        bundle(10, destinations=(4,), maps=(1, 1)))
    check('selector alias retains same-lane reads alongside data routing',
          any(edge['kind'] == 'RAW' and 'destination_resource' not in edge and edge['lane_mask'] == 3 for edge in alias['edges']))
    model = LaneDependencies()
    unknown = model.issue(issue(10, defs=[4], uses=[0], instruction='shfl.sync.idx.b32 r2,r0,0,31,3'), bundle(10, destinations=(4,), maps=(0, 1)))
    check('inactive shuffle source remains explicit undefined output', unknown['undefined_results'] == [
          {'resource': ['register', 0, 4], 'lane_mask': 1, 'reason': 'inactive shuffle source'}])
    check('unknown collective producer preserves exact destination pairing', unknown['live_ins'] == [
          {'resource': ['register', 0, 0], 'lane_mask': 1, 'destination_resource': ['register', 0, 4], 'destination_lane_mask': 2}])
    model = LaneDependencies(); model.issue(issue(1, defs=[1], mask=1)); model.issue(issue(2, defs=[1], mask=2))
    match = model.issue(issue(10, defs=[4, 6], uses=[1], instruction='match.sync.all.b64 r2|p0,r0,3'),
                        bundle(10, source=1, destinations=(4, 6), maps=(3, 3), kind='match'))
    check('wide-source multi-output match preserves all eight directed pairs', len(routed(match)) == 8)
    check('wide collective source never aliases narrow file', all(pair[1][-1] == 1 for pair in routed(match)))
    for destination in (4, 6):
        check('match output ' + str(destination) + ' keeps separate destination lanes',
              {pair[4] for pair in routed(match) if pair[3][-1] == destination} == {1, 2})
    model = LaneDependencies(); model.issue(issue(1, defs=[4]))
    wait = [dict(type='collective_wait', sequence=11, issue=10)]
    parked = model.issue(issue(10, defs=[4], uses=[0], instruction='shfl.sync.idx.b32 r2,r0,0,31,3'), wait, collective_schema=1)
    check('parked attempt has zero executed lanes', parked['collective_wait'] and parked['actual_execution_mask'] == 0)
    check('parked selection uncertainty is explicit', parked['conservative_selection_reads'])
    check('parked destination keeps earlier writer', model.issue(issue(20, uses=[4]))['edges'][0]['source'] == 1)
    model = LaneDependencies()
    constant = model.issue(issue(10, defs=[4], instruction='vote.sync.ballot.b32 r2,1,3'),
                           bundle(10, source=-1, destinations=(4,), maps=(), kind='vote'))
    check('constant collective data invents no virtual producer', not constant['edges'] and not constant['live_ins'])
    model = LaneDependencies(); model.issue(issue(1, defs=[0]))
    broken = bundle(10); broken.pop()
    reject('incomplete maps rejected before graph mutation', lambda: model.issue(shuffle, broken))
    check('failed conservation leaves versions and sequence intact', model.last_sequence == 1 and
          all(writer == 1 for writer in model.writers.values()))
    reject('collective without records rejected', lambda: model.issue(shuffle))
    broken = bundle(10); broken[1]['issue'] = 9
    reject('wrong-parent transfer rejected', lambda: model.issue(shuffle, broken))
    broken = bundle(10); broken[1]['sequence'] = 11
    reject('non-increasing child sequence rejected', lambda: model.issue(shuffle, broken))
    reject('ordinary instruction rejects collective records', lambda: model.issue(issue(10), bundle(10)))
    check('valid issue succeeds after failed bundles', bool(model.issue(shuffle, bundle(10))['edges']))
    model = LaneDependencies(); model.issue(issue(1, defs=[2])); model.issue(issue(2, defs=[0]))
    guarded = model.issue(issue(10, defs=[4], uses=[0], mask=1, path=3, predicate=2,
                         instruction='@p0 shfl.sync.idx.b32 r2,r0,0,31,1'),
                         bundle(10, destinations=(4,), maps=(1,), mask=1))
    check('collective guard still reads full selected path', any(edge['source'] == 1 and edge['lane_mask'] == 3
          and 'destination_resource' not in edge for edge in guarded['edges']))
    check('predicated-off collective result has no writer', bool(model.issue(issue(20, uses=[4], mask=2))['live_ins']))
    model = LaneDependencies(); model.issue(issue(1, defs=[0]))
    split = [dict(type='collective_group', sequence=11, issue=10, kind='shfl', execution_mask=1, data_use=0, data_defs=[0]),
             dict(type='collective_transfer', sequence=12, issue=10, group=11, destination_lane=0, source_mask=1),
             dict(type='collective_group', sequence=13, issue=10, kind='shfl', execution_mask=2, data_use=0, data_defs=[0]),
             dict(type='collective_transfer', sequence=14, issue=10, group=13, destination_lane=1, source_mask=2)]
    multi = model.issue(issue(10, defs=[0], uses=[0], instruction='shfl.sync.idx.b32 r0,r0,0,31,r1'), split)
    check('disjoint groups read snapshot of same preceding issue', routed(multi) == {
          (1, ('register', 0, 0), 1, ('register', 0, 0), 1), (1, ('register', 0, 0), 2, ('register', 0, 0), 2)})
    model = LaneDependencies(); split[2]['data_use'] = -1; split.pop()
    reject('groups cannot change operand class within one issue', lambda: model.issue(
           issue(10, defs=[0], uses=[0], instruction='shfl.sync.idx.b32 r0,r0,0,31,r1'), split))
    reject('discarded data output refuses unrecorded source routing', lambda: model.issue(
           issue(10, defs=[2], uses=[0], instruction='shfl.sync.idx.b32 _|p0,r0,0,31,3'),
           bundle(10, destinations=(), maps=())))
    return checks

if __name__ == '__main__':
    print('PASS collective dependency components checks=' + str(len(verify_collective_dependencies())))
