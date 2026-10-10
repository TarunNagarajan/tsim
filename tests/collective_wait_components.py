"""Exact parked reads and discarded-output source dependencies, with legacy checks."""
from register_dependency_components import issue
from register_dependencies import LaneDependencies
from collective_events import CollectiveEvents

def verify_collective_wait_reads():
    checks = []
    def check(name, condition):
        if not condition: raise AssertionError(name)
        checks.append(name)
    def reject(name, action):
        try: action()
        except ValueError: checks.append(name)
        else: raise AssertionError(name)
    def parked(member=6, lanes=2):
        return [dict(type='collective_wait', sequence=11, issue=10, member_use=member, member_lane_mask=lanes)]
    event = issue(10, defs=[4], uses=[0, 2, 6], mask=6,
                  instruction='shfl.sync.idx.b32 r2,r0,r1,31,r3')
    model = LaneDependencies(); model.issue(issue(1, defs=[0, 2, 6], mask=7))
    node = model.issue(event, parked())
    check('parked attempt reads only membership from first executing lane', node['edges'] == [
          {'source': 1, 'target': 10, 'kind': 'RAW', 'resource': ['register', 0, 6], 'lane_mask': 2}])
    check('new wait capture removes conservative selection flag', not node['conservative_selection_reads'])
    after = model.issue(issue(20, defs=[0, 2], mask=7))
    check('parked data and selector create no anti-dependencies', all(edge['kind'] != 'WAR' for edge in after['edges']))
    after = model.issue(issue(30, defs=[6], mask=7))
    check('parked membership anti-dependency covers exact first lane',
          [edge['lane_mask'] for edge in after['edges'] if edge['kind'] == 'WAR'] == [2])
    model = LaneDependencies()
    constant = model.issue(issue(10, defs=[4], uses=[0], mask=6,
                                 instruction='shfl.sync.idx.b32 r2,r0,0,31,7'), parked(-1, 0))
    check('immediate membership invents no parked virtual reads', not constant['edges'] and not constant['live_ins'])
    model = LaneDependencies(); model.issue(issue(1, defs=[6], mask=7))
    alias = model.issue(issue(10, defs=[4], uses=[6, 6], mask=6,
                              instruction='shfl.sync.idx.b32 r2,r3,0,31,r3'), parked())
    check('data/member alias reads only actual parked membership lane', len(alias['edges']) == 1 and alias['edges'][0]['lane_mask'] == 2)
    model = LaneDependencies(); model.issue(issue(1, defs=[2, 6], mask=7))
    guarded = model.issue(issue(10, defs=[4], uses=[0, 6], mask=6, path=7, predicate=2,
                         instruction='@p0 shfl.sync.idx.b32 r2,r0,0,31,r3'), parked())
    check('parked guard and membership retain different lane masks',
          {tuple(edge['resource']): edge['lane_mask'] for edge in guarded['edges']} == {('register', 0, 2): 7, ('register', 0, 6): 2})
    for name, change in [('data key used as membership', {'member_use': 0}),
                         ('wrong first lane', {'member_lane_mask': 4}), ('all selected lanes', {'member_lane_mask': 6}),
                         ('boolean key', {'member_use': True}), ('boolean mask', {'member_lane_mask': True}),
                         ('overflow mask', {'member_lane_mask': 1 << 32}), ('negative key', {'member_use': -2}),
                         ('nonvirtual with lanes', {'member_use': -1, 'member_lane_mask': 2})]:
        children = parked(); children[0].update(change)
        reject(name + ' rejected before version mutation', lambda children=children: LaneDependencies().issue(event, children))
    reject('new schema refuses missing wait fields', lambda: LaneDependencies().issue(event, [dict(type='collective_wait', sequence=11, issue=10)]))
    reject('legacy shuffle cannot invent membership wait', lambda: LaneDependencies().issue(
           issue(10, defs=[4], uses=[0, 2, 6], mask=6, instruction='shfl.idx.b32 r2,r0,r1,r3'), parked()))
    validator = CollectiveEvents({10: event}, 2); validator.record(parked()[0]); validator.finish(10, 'completed')
    check('exact wait source reads conserve one membership lane', validator.counts['member_lane_reads'] == 1)
    for schema in [True, 0, 3]:
        reject('unsupported collective schema ' + repr(schema), lambda schema=schema: CollectiveEvents({}, schema))
    sink = issue(10, defs=[2], uses=[0], instruction='match.all.sync.b32 _|p0,r0,3')
    children = [dict(type='collective_group', sequence=11, issue=10, kind='match', execution_mask=3, data_use=0, data_defs=[2]),
                dict(type='collective_transfer', sequence=12, issue=10, group=11, destination_lane=0, source_mask=3),
                dict(type='collective_transfer', sequence=13, issue=10, group=11, destination_lane=1, source_mask=3)]
    model = LaneDependencies(); model.issue(issue(1, defs=[0])); node = model.issue(sink, children)
    check('discarded match data keeps predicate dependencies on every contributor', node['edges'] == [
          {'source': 1, 'target': 10, 'kind': 'RAW', 'resource': ['register', 0, 0], 'lane_mask': 3,
           'destination_resource': ['register', 0, 2], 'destination_lane_mask': bit} for bit in [1, 2]])
    check('discarded match data has no fabricated data register result', all(edge['destination_resource'][-1] == 2 for edge in node['edges']))
    overwrite = model.issue(issue(20, defs=[0]))
    check('discarded data source remains outstanding reader', [edge['lane_mask'] for edge in overwrite['edges'] if edge['kind'] == 'WAR'] == [3])
    reject('discarded-source maps still require full conservation', lambda: LaneDependencies().issue(sink, children[:-1]))
    reject('older sink capture does not claim missing source reads', lambda: LaneDependencies().issue(sink, children[:1], collective_schema=1))
    return checks

if __name__ == '__main__': print('PASS exact collective wait components checks=' + str(len(verify_collective_wait_reads())))
