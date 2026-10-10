"""Independent dispatch-group, source-lane and conservation references."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from collective_events import CollectiveEvents

def verify_collective_components():
    checks = []
    def check(name, condition):
        if not condition: raise AssertionError(name)
        checks.append(name)
    def reject(name, action):
        try: action()
        except ValueError: checks.append(name)
        else: raise AssertionError(name)
    def model(kind='shfl', mask=3, defs=None):
        parent = {'instruction': kind + '.sync.b32 %r1,%r0,1,31,3', 'execution_mask': mask,
                  'uses': [0, 6], 'defs': [2] if defs is None else defs}
        return CollectiveEvents({10: parent})
    def group(sequence=11, mask=3, kind='shfl', source=0, defs=None):
        return dict(type='collective_group', sequence=sequence, issue=10, execution_mask=mask,
                    kind=kind, data_use=source, data_defs=[2] if defs is None else defs)
    def transfer(lane, sources, identity=11):
        return dict(type='collective_transfer', issue=10, group=identity, destination_lane=lane, source_mask=sources)
    captured = model(); captured.record(group()); captured.record(transfer(0, 2)); captured.record(transfer(1, 1)); captured.finish(10, 'completed')
    check('two-lane shuffle swaps exact source lanes', captured.counts['source_lane_reads'] == 2 and captured.counts['destination_maps'] == 2)
    check('completed groups release validator state', not captured.groups and not captured.group_ids and not captured.masks)
    captured = model(mask=15)
    captured.record(group(mask=3)); captured.record(transfer(0, 1)); captured.record(transfer(1, 2))
    captured.record(group(sequence=14, mask=12)); captured.record(transfer(2, 4, 14)); captured.record(transfer(3, 8, 14)); captured.finish(10, 'completed')
    check('multiple disjoint groups conserve parent participation', captured.counts['shfl_groups'] == 2 and captured.counts['destination_maps'] == 4)
    for kind in ['vote', 'redux', 'match']:
        captured = model(kind); captured.record(group(kind=kind)); captured.record(transfer(0, 3)); captured.record(transfer(1, 3)); captured.finish(10, 'completed')
        check(kind + ' results depend on all contributing lanes', captured.counts['source_lane_reads'] == 4)
    captured = model(); captured.record(group()); captured.record(transfer(0, 0)); captured.record(transfer(1, 1)); captured.finish(10, 'completed')
    check('inactive shuffle source stays explicitly undefined', captured.counts['undefined_shuffle_sources'] == 1)
    captured = model(); captured.record(group(source=-1)); captured.finish(10, 'completed')
    check('nonvirtual data has no fabricated register transfers', captured.counts['destination_maps'] == 0)
    captured = model(); captured.record({'type': 'collective_wait', 'issue': 10}); captured.finish(10, 'completed')
    check('parked attempt completes without a dispatch or destination write', captured.counts['wait_attempts'] == 1 and not captured.groups)
    captured = model(mask=0); captured.finish(10, 'completed')
    check('fully predicated-off collective needs no dispatch', not captured.counts)
    captured = model(); captured.record(group()); captured.record(transfer(0, 1)); captured.finish(10, 'fault')
    check('fault preserves incomplete successful source-map prefix', captured.counts['destination_maps'] == 1)
    captured = model(defs=[2, 4]); captured.record(group()); captured.record(transfer(0, 1)); captured.record(transfer(1, 2)); captured.finish(10, 'completed')
    check('shuffle predicate output does not acquire data transfer dependency', captured.counts['destination_maps'] == 2)
    captured = model(); captured.record(group());
    reject('missing destination maps rejected', lambda: captured.finish(10, 'completed'))
    captured = model(); reject('missing collective dispatch rejected', lambda: captured.finish(10, 'completed'))
    captured = model(mask=15); captured.record(group()); captured.record(transfer(0, 1)); captured.record(transfer(1, 2))
    reject('missing participation subset rejected', lambda: captured.finish(10, 'completed'))
    for name, change in [
        ('empty dispatch', {'execution_mask': 0}), ('nonexecuting group lanes', {'execution_mask': 4}),
        ('boolean mask', {'execution_mask': True}), ('overflow mask', {'execution_mask': 1 << 32}),
        ('wrong kind', {'kind': 'vote'}), ('wrong data source', {'data_use': 6}),
        ('boolean data source', {'data_use': False}), ('negative source', {'data_use': -2}),
        ('invented destination', {'data_defs': [8]}), ('boolean destination', {'data_defs': [True]}),
        ('duplicate destination', {'data_defs': [2, 2]}), ('boolean group identity', {'sequence': True}),
        ('boolean parent', {'issue': True})]:
        event = group(); event.update(change)
        reject(name + ' rejected', lambda e=event: model().record(e))
    captured = model(); captured.record(group())
    reject('overlapping groups rejected', lambda: captured.record(group(sequence=14, mask=1)))
    reject('wait after dispatch rejected', lambda: captured.record({'type': 'collective_wait', 'issue': 10}))
    for name, change in [
        ('unknown group', {'group': 99}), ('wrong parent', {'issue': 99}),
        ('negative destination lane', {'destination_lane': -1}), ('nonexecuting destination', {'destination_lane': 3}),
        ('boolean destination lane', {'destination_lane': True}), ('source outside group', {'source_mask': 4}),
        ('multiple shuffle sources', {'source_mask': 3}), ('boolean source mask', {'source_mask': True})]:
        event = transfer(0, 1); event.update(change)
        reject(name + ' rejected', lambda e=event: captured.record(e))
    captured.record(transfer(0, 1)); reject('duplicate destination map rejected', lambda: captured.record(transfer(0, 2)))
    captured = model('redux'); captured.record(group(kind='redux'))
    reject('reduction partial source mask rejected', lambda: captured.record(transfer(0, 1)))
    captured = model(defs=[2, 4]); reject('shuffle predicate falsely depends on data rejected', lambda: captured.record(group(defs=[2, 4])))
    captured = model(); captured.record({'type': 'collective_wait', 'issue': 10})
    reject('dispatch after wait rejected', lambda: captured.record(group()))
    reject('duplicate wait rejected', lambda: captured.record({'type': 'collective_wait', 'issue': 10}))
    return checks

if __name__ == '__main__':
    print('PASS collective event components checks=' + str(len(verify_collective_components())))
