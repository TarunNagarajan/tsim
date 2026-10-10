"""Register hazard references and actual predicated CUDA carry captures."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

def verify_dependencies(cache, out, replay_from=None):
    from tsim import ROOT, run_program
    from register_dependencies import analyze_dependencies
    from execution_events import summarize_events
    sys.path.insert(0, str(ROOT / 'tests'))
    from register_dependency_components import verify_dependency_components
    from collective_event_components import verify_collective_components
    from collective_dependency_components import verify_collective_dependencies
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=True)
    checks = verify_dependency_components()
    checks.extend(verify_collective_components())
    checks.extend(verify_collective_dependencies())
    report = {'kind': 'lane-aware observed PTX register/carry hazards; not hardware scoreboard validation',
              'passed': False, 'checks': checks, 'hardware_cycles': False, 'cases': [],
              'model_sha256': {name: hashlib.sha256((ROOT / 'scripts' / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                               for name in ['register_dependencies.py', 'execution_events.py', 'collective_events.py']},
              'reference_sha256': {name: hashlib.sha256((ROOT / 'tests' / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                                   for name in ['register_dependency_components.py', 'collective_event_components.py',
                                                'collective_dependency_components.py', 'conformance/collective_dependencies.cu']},
              'writer_sha256': hashlib.sha256((ROOT / 'include/tsim/execution_events.hpp').read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
              'patch_sha256': {p.name: hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                               for p in sorted((ROOT / 'patches').glob('*.patch'))}}
    def check(name, condition):
        if not condition: raise AssertionError(name)
        checks.append(name)
    def reject(name, action, message=None):
        try: action()
        except ValueError as error:
            if message is not None and message not in str(error): raise
            checks.append(name)
        else: raise AssertionError(name)
    compiled = {}
    if replay_from:
        for record in sorted(Path(replay_from).glob('*/run.json')):
            metadata = json.loads(record.read_text()); source = Path(metadata['source']); snapshot = record.parent / 'source.cu'
            if source.is_file() and snapshot.is_file() and snapshot.read_bytes() == source.read_bytes() and not metadata['nvcc_args']:
                compiled.setdefault(str(source), record.parent)
    def run(name, source, gpu='h100', scheduler='deterministic', limit=400000, negative=False, workers=1):
        args = argparse.Namespace(source=source, gpu=gpu, trace=False, trace_limit=200000, events=True,
            events_limit=limit, scheduler=scheduler, seed=73, threads=workers, race=True, strict=True,
            timeout=60, out=out / name, nvcc_arg=[])
        code, path = run_program(args, cache, compiled_from=compiled.get(str(source)))
        compiled.setdefault(str(source), path)
        check(name + ' numerical process', code != 0 if negative else code == 0)
        return path
    def save(path, model):
        (path / 'register-dependencies.json').write_text(json.dumps(model, indent=2))
        report['cases'].append({'name': path.name, 'nodes': len(model['nodes']), 'counts': model['counts'],
                                'event_sha256': model['event_sha256']})
    def carry_signature(events, model):
        ordinal, identities = {}, {}
        for event in events:
            if event['type'] != 'issue' or not (event['implicit_uses'] or event['implicit_defs']): continue
            key = (event['launch'], tuple(event['block']), event['warp'])
            ordinal[key] = ordinal.get(key, 0) + 1
            identities[event['sequence']] = (key, ordinal[key])
        result = []
        for node in model['nodes']:
            if node['sequence'] not in identities: continue
            key, index = identities[node['sequence']]
            edges = [e for e in node['edges'] if e['resource'] == ['implicit', 'CC.CF']]
            if any(identities[e['source']][0] != key for e in edges): raise AssertionError('Cross-warp carry hazard')
            result.append((key, index, sorted((identities[e['source']][1], e['kind'], e['lane_mask']) for e in edges)))
        return sorted(result)
    try:
        binary = out / 'register-dependency-extraction'
        subprocess.run(['clang++', '-std=c++20', '-O2', '-I' + str(ROOT / 'third_party/pantheonsim/include'),
                        str(ROOT / 'tests/register_dependency_extraction.cpp'), '-L' + str(Path(cache) / 'build'),
                        '-lvgpu', '-Wl,-rpath,' + str(Path(cache) / 'build'), '-o', str(binary)], check=True)
        result = subprocess.run([str(binary)], text=True, capture_output=True, check=True)
        (out / 'extraction.log').write_text(result.stdout + result.stderr)
        check('thirty-eight parser/AST operand role references passed', 'checks=38' in result.stdout)
        report['operand_role_references'] = 38
        for gpu in ['a100', 'h100']:
            path = run('vector-' + gpu, ROOT / 'examples/vector_add.cu', gpu)
            model = analyze_dependencies(path); save(path, model)
            check('vector-' + gpu + ' single-definition registers have RAW without artificial overwrite hazards',
                  model['counts'].get('RAW', 0) > 0 and model['counts'].get('WAR', 0) == model['counts'].get('WAW', 0) == 0)
            check('vector-' + gpu + ' no fabricated live-in register roots', model['counts']['live_in_lane_reads'] == 0)
        reference = None
        for gpu in ['a100', 'h100']:
            for scheduler in ['deterministic', 'random', 'adversarial']:
                name = 'carry-' + gpu + '-' + scheduler
                path = run(name, ROOT / 'tests/conformance/carry_dependencies.cu', gpu, scheduler)
                events = list(map(json.loads, (path / 'execution-events.jsonl').read_text().splitlines()))
                model = analyze_dependencies(path); save(path, model)
                parents = {e['sequence']: e for e in events if e['type'] == 'issue'}
                nodes = {n['sequence']: n for n in model['nodes']}
                guarded = [e for e in parents.values() if e['implicit_defs'] and e['instruction'].lstrip().startswith('@')]
                check(name + ' guarded carry writes use even executing lanes', guarded and all(
                      e['execution_mask'] == e['path_mask'] & 0x55555555 and e['predicate_use'] >= 0 for e in guarded))
                guards = [e for e in parents.values() if e['instruction'].lstrip().startswith('@') and e['implicit_uses']]
                check(name + ' guarded carry reads use odd executing lanes', guards and all(
                      e['execution_mask'] == e['path_mask'] & 0xaaaaaaaa for e in guards))
                correct = True; finals = 0
                for final in parents.values():
                    if re.sub(r'\s+', '', final['instruction']) != 'addc.u32final,30,0': continue
                    finals += 1
                    carry_edges = [e for e in nodes[final['sequence']]['edges'] if e['kind'] == 'RAW' and e['resource'] == ['implicit', 'CC.CF']]
                    expected_even = final['execution_mask'] & 0x55555555
                    expected_odd = final['execution_mask'] & 0xaaaaaaaa
                    expected = {}
                    if expected_even: expected['guarded'] = expected_even
                    if expected_odd: expected['seed'] = expected_odd
                    actual = {}
                    for edge in carry_edges:
                        producer = parents[edge['source']]
                        if producer['instruction'].lstrip().startswith('@'): kind = 'guarded'
                        elif re.sub(r'\s+', '', producer['instruction']) == 'add.cc.u32trash,0xffffffff,1': kind = 'seed'
                        else: kind = 'unexpected'
                        actual[kind] = actual.get(kind, 0) | edge['lane_mask']
                    correct &= actual == expected
                check(name + ' final carry sources preserve distinct even/odd writers', finals == 3 and correct)
                signature = carry_signature(events, model)
                if reference is None: reference = signature
                check(name + ' carry hazards invariant across profiles and schedules', signature == reference)
                check(name + ' graph explicitly lacks complete scoreboard claim', model['complete_scoreboard_dependencies'] is False)
        path = run('calls', ROOT / 'tests/conformance/device_calls.cu')
        reject('call ABI graph refuses incomplete capture', lambda: analyze_dependencies(path), 'coverage gaps')
        path = run('indirect', ROOT / 'tests/conformance/indirect_dependencies.cu')
        events = list(map(json.loads, (path / 'execution-events.jsonl').read_text().splitlines()))
        calls = [e for e in events if e['type'] == 'issue' and e['instruction'].startswith('call')]
        check('actual indirect calls capture one wide target register', calls and all(len(e['uses']) == 1 and e['uses'][0] & 1 for e in calls))
        reject('indirect target capture does not imply complete call ABI', lambda: analyze_dependencies(path), 'coverage gaps')
        path = run('collectives', ROOT / 'tests/conformance/warp_collectives.cu')
        collective_model = analyze_dependencies(path); save(path, collective_model)
        check('shuffle/vote/reduction/match graph consumes actual transfer records',
              collective_model['schema'] == 2 and any('destination_resource' in edge
              for node in collective_model['nodes'] for edge in node['edges']))
        events = list(map(json.loads, (path / 'execution-events.jsonl').read_text().splitlines()))
        parents = {e['sequence']: e for e in events if e['type'] == 'issue'}
        groups = [e for e in events if e['type'] == 'collective_group' and e['kind'] == 'shfl']
        transfers = {}
        for event in events:
            if event['type'] == 'collective_transfer': transfers.setdefault(event['group'], []).append(event)
        ordinals = {}; correct = len(groups) == 40
        for group in groups:
            warp = parents[group['issue']]['warp']; ordinal = ordinals.get(warp, 0); ordinals[warp] = ordinal + 1
            width = 2 << (ordinal // 4); mode = ordinal % 4
            expected = {}
            for lane in range(32):
                source = (lane // width * width + 1) if mode == 0 else (
                    lane - 1 if lane % width else lane) if mode == 1 else (
                    lane + 1 if lane % width + 1 < width else lane) if mode == 2 else lane ^ 1
                expected[lane] = 1 << source
            actual = {e['destination_lane']: e['source_mask'] for e in transfers.get(group['sequence'], [])}
            correct &= actual == expected
        check('actual shuffle source maps agree with twenty independent width/mode references per warp', correct)
        collective_reference = None
        collective_original = None
        for gpu in ['a100', 'h100']:
            for scheduler in ['deterministic', 'random', 'adversarial']:
                name = 'members-' + gpu + '-' + scheduler
                path = run(name, ROOT / 'tests/conformance/dynamic_members.cu', gpu, scheduler)
                events = list(map(json.loads, (path / 'execution-events.jsonl').read_text().splitlines()))
                selected = [e for e in events if e['type'] == 'issue' and
                            e['instruction'].split('.')[0] in ['vote', 'redux', 'match', 'shfl', 'fns']]
                # These compiled forms name only virtual registers and
                # immediates. Count every spelled source occurrence, including
                # repeated registers, independently of the AST field names.
                check(name + ' dynamic member and bit-search sources retained',
                      {e['instruction'].split('.')[0] for e in selected} == {'vote', 'redux', 'match', 'shfl', 'fns'} and
                      all(len(e['uses']) == len(re.findall(r'%[\w.$]+', e['instruction'])) - len(e['defs']) for e in selected))
                parents = {e['sequence']: e for e in events if e['type'] == 'issue'}
                groups = [e for e in events if e['type'] == 'collective_group']
                maps = {}
                for event in events:
                    if event['type'] == 'collective_transfer': maps.setdefault(event['group'], []).append(event)
                check(name + ' runtime vote/reduction/match dispatch disjoint sixteen-lane groups', groups and all(
                      e['execution_mask'] in [0xffff, 0xffff0000] for e in groups if e['kind'] != 'shfl'))
                check(name + ' actual XOR shuffle reads opposite neighbor', any(e['kind'] == 'shfl' for e in groups) and all(
                      event['source_mask'] == 1 << (event['destination_lane'] ^ 1)
                      for group in groups if group['kind'] == 'shfl' for event in maps.get(group['sequence'], [])))
                signature = sorted((parents[e['issue']]['warp'], parents[e['issue']]['pc'], e['kind'], e['execution_mask'],
                                    e['data_use'], e['data_defs'], sorted((t['destination_lane'], t['source_mask']) for t in maps.get(e['sequence'], [])))
                                   for e in groups)
                if collective_reference is None: collective_reference = signature; collective_original = path
                check(name + ' collective source maps invariant across profiles and schedules', signature == collective_reference)
                model = analyze_dependencies(path); save(path, model)
                check(name + ' collective graph preserves directed lane identities', any(
                      edge.get('destination_lane_mask', 0) != edge['lane_mask']
                      for node in model['nodes'] for edge in node['edges'] if 'destination_resource' in edge))
        captured = list(map(json.loads, (collective_original / 'execution-events.jsonl').read_text().splitlines()))
        negative_collective = out / 'negative-collective.jsonl'
        for name, change in [
            ('missing transfer', lambda rows: rows.pop(next(i for i, e in enumerate(rows) if e['type'] == 'collective_transfer'))),
            ('wrong source mask', lambda rows: next(e for e in rows if e['type'] == 'collective_transfer').update(source_mask=1 << 32)),
            ('wrong group owner', lambda rows: next(e for e in rows if e['type'] == 'collective_transfer').update(group=1)),
            ('missing dispatch participation', lambda rows: next(e for e in rows if e['type'] == 'collective_group').update(execution_mask=0))]:
            modified = deepcopy(captured); change(modified)
            # Keep the stream identities contiguous after deletion, while
            # remapping issue/group references, to reach conservation checks.
            identities = {e['sequence']: index + 1 for index, e in enumerate(modified)}
            for index, event in enumerate(modified):
                event['sequence'] = index + 1
                for key in ['issue', 'group']:
                    if key in event: event[key] = identities[event[key]]
            negative_collective.write_text(''.join(json.dumps(e) + '\n' for e in modified))
            reject('corrupted collective ' + name + ' rejected', lambda: summarize_events(negative_collective))
        for gpu in ['a100', 'h100']:
            for scheduler in ['deterministic', 'random', 'adversarial']:
                name = 'wait-' + gpu + '-' + scheduler
                path = run(name, ROOT / 'tests/conformance/collective_wait.cu', gpu, scheduler, workers=4)
                summary = summarize_events(path / 'execution-events.jsonl')
                check(name + ' actual parked collective attempts captured', summary['complete_supported_collectives'] and summary['collectives'].get('wait_attempts', 0) >= 4)
                events = list(map(json.loads, (path / 'execution-events.jsonl').read_text().splitlines()))
                parents = {e['sequence']: e for e in events if e['type'] == 'issue'}
                groups = [e for e in events if e['type'] == 'collective_group']
                check(name + ' four-worker groups retain exact block ownership', len(groups) == 12 and
                      {tuple(parents[e['issue']]['block']) for e in groups} == {(i, 0, 0) for i in range(4)} and
                      all(e['execution_mask'] == 0xffffffff for e in groups))
                by_id = {e['sequence']: e for e in groups}
                check(name + ' reconverged shuffle source is original lane zero', all(
                      e['source_mask'] == 1 for e in events if e['type'] == 'collective_transfer' and
                      by_id[e['group']]['kind'] == 'shfl'))
                model = analyze_dependencies(path); save(path, model)
                check(name + ' parked graph issues never write destinations', all(
                      node['actual_execution_mask'] == 0 and not any(edge['kind'] in ['WAW', 'WAR'] for edge in node['edges'])
                      for node in model['nodes'] if node.get('collective_wait')))
        for gpu in ['a100', 'h100']:
            for scheduler in ['deterministic', 'random', 'adversarial']:
                name = 'lane-writers-' + gpu + '-' + scheduler
                path = run(name, ROOT / 'tests/conformance/collective_dependencies.cu', gpu, scheduler, workers=4)
                events = list(map(json.loads, (path / 'execution-events.jsonl').read_text().splitlines()))
                parents = {e['sequence']: e for e in events if e['type'] == 'issue'}
                model = analyze_dependencies(path); save(path, model)
                shuffles = [node for node in model['nodes'] if node.get('collective_kind') == 'shfl' and not node['collective_wait']]
                check(name + ' actual four-warp in-place shuffles present', len(shuffles) == 4 and all(
                      parents[node['sequence']]['defs'] == parents[node['sequence']]['uses'] for node in shuffles))
                correct = True
                for node in shuffles:
                    parent = parents[node['sequence']]; key = parent['defs'][0]
                    producers = [e for e in parents.values() if e['sequence'] < parent['sequence'] and
                                 e['block'] == parent['block'] and e['warp'] == parent['warp'] and
                                 e['defs'] == [key] and 'add.u32' in e['instruction']]
                    even = [e for e in producers if e['execution_mask'] == 0x55555555]
                    odd = [e for e in producers if e['execution_mask'] == 0xaaaaaaaa]
                    routed = [e for e in node['edges'] if 'destination_resource' in e]
                    correct &= bool(even and odd) and len(routed) == 32
                    if even and odd:
                        expected = {lane: (odd[-1]['sequence'] if lane % 2 == 0 else even[-1]['sequence'], 1 << (lane ^ 1)) for lane in range(32)}
                        actual = {e['destination_lane_mask'].bit_length() - 1: (e['source'], e['lane_mask']) for e in routed}
                        correct &= actual == expected and all(e['resource'] == e['destination_resource'] for e in routed)
                check(name + ' every swapped lane depends on opposite parity original writer', correct)
        path = run('cap', ROOT / 'examples/vector_add.cu', limit=37)
        reject('capped capture refuses dependency graph', lambda: analyze_dependencies(path), 'complete capture')
        path = run('fault', ROOT / 'examples/out_of_bounds.cu', negative=True)
        reject('faulted capture refuses dependency graph', lambda: analyze_dependencies(path), 'complete capture')
        original = out / 'carry-h100-deterministic'; negative = out / 'negative-capture'; negative.mkdir(exist_ok=True)
        for name in ['run.json', 'program', 'kernel.ptx']: shutil.copy2(original / name, negative / name)
        data = list(map(json.loads, (original / 'execution-events.jsonl').read_text().splitlines()))
        def write(events): (negative / 'execution-events.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in events))
        for name, change in [
            ('unknown register', lambda e: e.update(uses=[999999])),
            ('missing carry', lambda e: e.update(implicit_defs=[])),
            ('missing predicate guard', lambda e: e.update(predicate_use=-1)),
            ('frame layout alias', lambda e: e.update(function='different')),
            ('wrong launch kernel', lambda e: e.update(kernel='different')),
            ('nonexistent warp', lambda e: e.update(warp=100)),
            ('nonexistent block', lambda e: e.update(block=[999, 0, 0]))]:
            modified = deepcopy(data)
            event = next(e for e in modified if e['type'] == 'issue' and e['implicit_defs'] and e['instruction'].lstrip().startswith('@'))
            change(event); write(modified)
            reject('modified ' + name + ' rejected', lambda: analyze_dependencies(negative))
        modified = deepcopy(data)
        next(e for e in modified if e['type'] == 'issue')['ptx_line'] += 1
        write(modified)
        reject('modified instruction line rejects provenance', lambda: analyze_dependencies(negative), 'compiled PTX')
        modified = deepcopy(data); modified[0].pop('dependency_schema')
        modified[0].pop('collective_schema', None); modified[0].pop('collective_coverage', None)
        for event in modified:
            if event['type'] == 'issue': event.pop('implicit_defs'); event.pop('implicit_uses')
        write(modified)
        check('older capture remains readable for memory conservation', summarize_events(negative / 'execution-events.jsonl')['complete_supported_memory'])
        reject('older capture cannot imply complete carry dependencies', lambda: analyze_dependencies(negative), 'schema-2')
        write(data); (negative / 'kernel.ptx').write_bytes((negative / 'kernel.ptx').read_bytes() + b'\n// changed\n')
        reject('modified compiled PTX rejects dependency report', lambda: analyze_dependencies(negative), 'artifact changed')
        report['passed'] = True
    finally:
        report['total_checks'] = len(checks)
        (out / 'results.json').write_text(json.dumps(report, indent=2))
    print('PASS register dependency conformance checks=' + str(len(checks)), flush=True)
    return 0
