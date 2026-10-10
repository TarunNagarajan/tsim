"""Independent lane/resource hazard cases and malformed capture boundaries."""
from copy import deepcopy
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from register_dependencies import LaneDependencies, CompiledInstructions
from execution_events import validate_dependency_issue

def issue(sequence, defs=(), uses=(), mask=3, path=None, predicate=-1, frame=0,
          instruction='add.u32 r0,r1,r2', implicit_defs=(), implicit_uses=(), **extra):
    event = dict(sequence=sequence, launch=1, block=[0, 0, 0], warp=0, frame=frame,
                 function='kernel' if frame == 0 else 'helper', pc=sequence, ptx_line=sequence,
                 path_mask=mask if path is None else path, execution_mask=mask,
                 defs=list(defs), uses=list(uses), predicate_use=predicate,
                 virtual_registers32=8, virtual_registers64=4, declared_local_bytes=0,
                 instruction=instruction, implicit_defs=list(implicit_defs), implicit_uses=list(implicit_uses))
    event.update(extra)
    return event

def verify_dependency_components():
    checks = []
    def check(name, condition):
        if not condition: raise AssertionError(name)
        checks.append(name)
    def reject(name, action):
        try: action()
        except ValueError: checks.append(name)
        else: raise AssertionError(name)
    def hazards(node): return {(e['source'], e['kind'], tuple(e['resource'])): e['lane_mask'] for e in node['edges']}
    model = LaneDependencies()
    a = model.issue(issue(1, defs=[0])); b = model.issue(issue(2, defs=[2], uses=[0]))
    check('new definitions do not invent predecessors', not a['edges'] and not a['live_ins'])
    check('two-lane RAW combines exact masks', hazards(b) == {(1, 'RAW', ('register', 0, 0)): 3})
    c = model.issue(issue(3, defs=[0], mask=1))
    check('partial overwrite has separate WAW and WAR edges', hazards(c) == {
          (1, 'WAW', ('register', 0, 0)): 1, (2, 'WAR', ('register', 0, 0)): 1})
    d = model.issue(issue(4, uses=[0]))
    check('lanes retain different last writers', hazards(d) == {
          (3, 'RAW', ('register', 0, 0)): 1, (1, 'RAW', ('register', 0, 0)): 2})
    model = LaneDependencies(); model.issue(issue(1, defs=[0]))
    wide = model.issue(issue(2, uses=[1]))
    check('32-bit and 64-bit files do not alias same register index', not wide['edges'] and wide['live_ins'] == [{'resource': ['register', 0, 1], 'lane_mask': 3}])
    model.issue(issue(3, uses=[0, 0])); write = model.issue(issue(4, defs=[0], uses=[0]))
    check('read-modify-write has no self edge', all(e['source'] != 4 for e in write['edges']))
    check('duplicate operands do not duplicate hazard edges', len([e for e in write['edges'] if e['kind'] == 'RAW']) == 1)
    check('read-modify-write preserves preceding reader anti-dependency', hazards(write)[(3, 'WAR', ('register', 0, 0))] == 3)
    next_write = model.issue(issue(5, defs=[0]))
    check('overwritten version clears previous readers', hazards(next_write) == {(4, 'WAW', ('register', 0, 0)): 3})
    model = LaneDependencies(); model.issue(issue(1, defs=[0])); model.issue(issue(2, uses=[0])); model.issue(issue(3, uses=[0]))
    node = model.issue(issue(4, defs=[0]))
    check('all outstanding readers precede overwrite', {e['source'] for e in node['edges'] if e['kind'] == 'WAR'} == {2, 3})
    model = LaneDependencies(); model.issue(issue(1, defs=[0])); model.issue(issue(2, defs=[4]))
    guarded = model.issue(issue(3, defs=[2], uses=[4], mask=1, path=3, predicate=0, instruction='@%p0 add.u32 r1,r2,1'))
    check('guard predicate reads full path while operand reads execution lanes', hazards(guarded) == {
          (1, 'RAW', ('register', 0, 0)): 3, (2, 'RAW', ('register', 0, 4)): 1})
    skipped = model.issue(issue(4, defs=[2], uses=[4], mask=0, path=3, predicate=0, instruction='@!%p0 add.u32 r1,r2,1'))
    check('fully predicated-off issue still reads predicate', hazards(skipped) == {(1, 'RAW', ('register', 0, 0)): 3})
    check('predicated-off issue has no ordinary live-ins', not skipped['live_ins'])
    untouched = model.issue(issue(5, uses=[2], mask=2))
    check('predicated-off destinations do not become writers', not untouched['edges'] and bool(untouched['live_ins']))
    model = LaneDependencies()
    model.issue(issue(1, instruction='add.cc.u32 r0,r1,r2', implicit_defs=['CC.CF']))
    node = model.issue(issue(2, instruction='addc.cc.u32 r0,r1,r2', implicit_defs=['CC.CF'], implicit_uses=['CC.CF']))
    check('carry RAW and WAW dependencies are explicit', hazards(node) == {
          (1, 'RAW', ('implicit', 'CC.CF')): 3, (1, 'WAW', ('implicit', 'CC.CF')): 3})
    model.issue(issue(3, instruction='sub.cc.u32 r0,r1,r2', mask=1, implicit_defs=['CC.CF']))
    node = model.issue(issue(4, instruction='subc.u32 r0,r1,r2', implicit_uses=['CC.CF']))
    check('carry versions are lane-specific', hazards(node) == {
          (3, 'RAW', ('implicit', 'CC.CF')): 1, (2, 'RAW', ('implicit', 'CC.CF')): 2})
    model = LaneDependencies(); model.issue(issue(1, defs=[0])); model.issue(issue(2, defs=[0], frame=1))
    node = model.issue(issue(3, uses=[0]))
    check('restored caller frame retains original register version', hazards(node) == {(1, 'RAW', ('register', 0, 0)): 3})
    model.issue(issue(4, defs=[0], frame=2))
    node = model.issue(issue(5, uses=[0], frame=3))
    check('distinct call generations do not alias registers', not node['edges'] and bool(node['live_ins']))
    for name, context in [('warp', {'warp': 1}), ('block', {'block': [1, 0, 0]}), ('launch', {'launch': 2})]:
        model = LaneDependencies(); model.issue(issue(1, defs=[0]))
        node = model.issue(issue(2, uses=[0], **context))
        check(name + ' resource ownership is isolated', not node['edges'] and bool(node['live_ins']))
    for name, change in [
        ('boolean register', {'defs': [True]}), ('register beyond file', {'uses': [16]}),
        ('wide register beyond file', {'uses': [9]}), ('negative register', {'uses': [-1]}),
        ('floating register', {'defs': [0.0]}), ('boolean file capacity', {'virtual_registers32': True}),
        ('wide predicate', {'predicate_use': 1, 'instruction': '@%p add.u32 r0,r1,r2'}),
        ('missing guard predicate', {'instruction': '@%p add.u32 r0,r1,r2'}),
        ('unexpected guard predicate', {'predicate_use': 0}), ('negative frame', {'frame': -1}),
        ('overflow lane mask', {'path_mask': 1 << 32}), ('lanes outside path', {'execution_mask': 7}),
        ('missing carry', {'instruction': 'addc.u32 r0,r1,r2'}),
        ('invented carry', {'implicit_defs': ['CC.CF']}), ('unknown implicit state', {'implicit_uses': ['unknown']}),
        ('duplicate carry', {'instruction': 'addc.u32 r0,r1,r2', 'implicit_uses': ['CC.CF', 'CC.CF']})]:
        event = issue(1); event.update(change)
        reject(name + ' rejected', lambda e=event: validate_dependency_issue(e))
    missing = issue(1); del missing['defs']
    reject('missing operand metadata rejected', lambda: validate_dependency_issue(missing))
    model = LaneDependencies(); model.issue(issue(1, defs=[0]))
    reject('non-increasing issue identities rejected', lambda: model.issue(issue(1)))
    reject('frame aliasing function rejected', lambda: model.issue(issue(2, function='other')))
    model = LaneDependencies(); model.issue(issue(1))
    reject('frame layout changes rejected', lambda: model.issue(issue(2, virtual_registers32=9)))
    check('invalid frame layout does not consume sequence identity', model.last_sequence == 1 and not model.issue(issue(2))['edges'])
    compiled = CompiledInstructions('/* fake add.u32 r0,r1,r2; */\n{ mov.u32 r0,1; add.u32 r1,\n r0,2; }\n@p addc.u32 r0,r1,0; // comment\n')
    check('inline instructions match actual source line', compiled.matches(2, 'mov.u32 r0, 1') and compiled.matches(2, 'add.u32 r1,r0,2'))
    check('multiline instructions keep start line', not compiled.matches(3, 'add.u32 r1,r0,2'))
    check('predicated spelling retains guard identity', compiled.matches(4, '@p addc.u32 r0,r1,0') and not compiled.matches(4, '@q addc.u32 r0,r1,0'))
    check('commented fake instruction does not establish provenance', not compiled.matches(1, 'add.u32 r0,r1,r2'))
    return checks

if __name__ == '__main__':
    print('PASS register dependency components checks=' + str(len(verify_dependency_components())))
