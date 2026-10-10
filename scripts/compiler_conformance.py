"""CPU CUDA execution, real NVIDIA compiler resources and refusal checks."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import sys

def verify_compiler(cache, out, replay_from=None):
    from tsim import ROOT, run_program
    from compiler_resources import compile_resources, load_resources
    from sm_residency import analyze_residency
    sys.path.insert(0, str(ROOT / 'tests'))
    from compiler_resource_components import verify_compiler_components
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=True)
    checks = verify_compiler_components()
    report = {'kind': 'real NVIDIA compiler/cubin resource checks on CPU; no GPU hardware measurements',
              'passed': False, 'checks': checks, 'hardware_validated': False, 'cases': [],
              'model_sha256': {name: hashlib.sha256((ROOT / 'scripts' / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                               for name in ['compiler_resources.py', 'sm_residency.py', 'memory_model.py']}}
    def check(name, condition):
        if not condition: raise AssertionError(name)
        checks.append(name)
    def reject(name, action):
        try: action()
        except ValueError: checks.append(name)
        else: raise AssertionError(name)
    compiled = {}
    if replay_from:
        for record in sorted(Path(replay_from).glob('*/run.json')):
            metadata = json.loads(record.read_text()); source = Path(metadata['source'])
            snapshot = record.parent / 'source.cu'
            if source.is_file() and snapshot.is_file() and snapshot.read_bytes() == source.read_bytes() and not metadata['nvcc_args']:
                compiled.setdefault(str(source), record.parent)
    try:
        for name, source, gpu in [
                ('vector-a100', ROOT / 'examples/vector_add.cu', 'a100'),
                ('vector-h100', ROOT / 'examples/vector_add.cu', 'h100'),
                ('spaces', ROOT / 'tests/conformance/memory_spaces.cu', 'h100'),
                ('reduce', ROOT / 'examples/shared_reduce.cu', 'h100'),
                ('aligned-prefix', ROOT / 'tests/conformance/shared_prefix.cu', 'h100'),
                ('register-pressure', ROOT / 'tests/conformance/register_pressure.cu', 'h100'),
                ('nested-stack', ROOT / 'tests/conformance/compiler_stack.cu', 'h100'),
                ('device-calls', ROOT / 'tests/conformance/device_calls.cu', 'h100')]:
            args = argparse.Namespace(source=source, gpu=gpu, trace=False, trace_limit=200000, events=True,
                events_limit=400000, scheduler='deterministic', seed=43, threads=1, race=True, strict=True,
                timeout=60, out=out / name, nvcc_arg=[])
            code, path = run_program(args, cache, compiled_from=compiled.get(str(source)))
            compiled.setdefault(str(source), path)
            check(name + ' complete numerical CUDA reference', code == 0)
            directory = path / 'compiler-resources'
            resources = compile_resources(path, cache, directory)
            check(name + ' resource provenance reload', load_resources(directory, path) == resources)
            entries = resources['resources']['entries']
            check(name + ' positive allocated entry registers', all(0 < e['registers'] <= 256 for e in entries.values()))
            report['cases'].append({'name': name, 'gpu': gpu, 'resources': resources['resources'],
                                    'ptx_sha256': resources['ptx_sha256'], 'tools': resources['tools']})
            if name in ['device-calls', 'nested-stack']:
                check(name + ' device function stack properties remain separate', bool(resources['resources']['device_functions']))
                if name == 'nested-stack':
                    entry = next(iter(entries.values()))
                    check('real local array stack allocation differs from register spilling', entry['stack_bytes'] > 0 and
                          entry['spill_store_bytes'] == 0 and entry['spill_load_bytes'] == 0 and entry['cumulative_stack_bytes'] > 0)
                # Compiler introspection requires valid compilation, independently
                # of whether PTX call ABI events cover a full architecture replay.
                continue
            model = analyze_residency(path, directory)
            check(name + ' compiler allocation residency provenance', model['physical_register_allocation_verified'] and
                  all(r['register_provenance'] == 'NVIDIA ptxas allocation cross-checked with cuobjdump' for r in model['launches']))
            (path / 'compiler-residency.json').write_text(json.dumps(model, indent=2))
            if name == 'spaces':
                check('compiler resource count differs from PTX liveness', all(
                      r['ptx_estimated_registers_per_thread'] > r['compiler_allocation']['registers'] for r in model['launches']))
                check('physical allocation changes resource ceiling', all(r['resource_ceiling']['blocks_per_sm'] == 32 for r in model['launches']))
                check('three launch resource identities preserved', len(model['launches']) == 3)
            if name == 'aligned-prefix':
                record = model['launches'][0]; allocation = record['compiler_allocation']
                check('real compiler prefix includes alignment padding', allocation['shared_prefix_bytes'] == 16)
                check('real cubin shared reservation retained separately', allocation['cubin_raw_shared_bytes'] == 1040 and
                      allocation['encoded_system_reservation_bytes'] == 1024)
                check('dynamic shared plus prefix has exact rounded charge', record['footprint']['shared_user_bytes'] == 140 and
                      record['footprint']['shared_bytes'] == 1280)
                check('ordinary barrier count comes from compiler', record['footprint']['barriers'] == 1)
                reject('PTX-only prefix still refuses guessing', lambda: analyze_residency(path))
            if name == 'register-pressure':
                entry = next(iter(entries.values()))
                check('real launch-bound register allocation respects 32-slot budget', entry['registers'] == 32)
                check('real compiler spill load/store bytes recorded', entry['spill_store_bytes'] > 0 and entry['spill_load_bytes'] > 0)
                check('real stack and cumulative requirement retained', entry['stack_bytes'] > 0 and entry['cumulative_stack_bytes'] >= entry['stack_bytes'])
                check('compiler spills do not become invented PTX local events', not any(
                      e.get('space') == 'local' for e in map(json.loads, (path / 'execution-events.jsonl').read_text().splitlines()) if e['type'] == 'memory'))
        source = ROOT / 'examples/out_of_bounds.cu'
        args = argparse.Namespace(source=source, gpu='h100', trace=False, trace_limit=200000, events=True,
            events_limit=400000, scheduler='deterministic', seed=43, threads=1, race=True, strict=True,
            timeout=60, out=out / 'fault', nvcc_arg=[])
        code, path = run_program(args, cache, compiled_from=compiled.get(str(source)))
        check('deliberately faulted CUDA reference fails functional execution', code != 0)
        resources = compile_resources(path, cache)
        check('compiler introspection retains failed functional status', resources['functional_run_exit_code'] == code and
              load_resources(path / 'compiler-resources', path) == resources)
        reject('faulted capture cannot become compiler-backed residency', lambda: analyze_residency(path, path / 'compiler-resources'))
        # Owned temporary copies keep negative tests away from real run evidence.
        original = out / 'aligned-prefix'; negative = out / 'negative-provenance'
        negative.mkdir(exist_ok=True)
        for filename in ['run.json', 'program', 'kernel.ptx']:
            shutil.copy2(original / filename, negative / filename)
        directory = negative / 'compiler-resources'; directory.mkdir(exist_ok=True)
        original_directory = original / 'compiler-resources'
        for filename in ['resources.json', 'input.ptx', 'module.cubin', 'ptxas.log', 'cuobjdump.log']:
            shutil.copy2(original_directory / filename, directory / filename)
        saved = json.loads((directory / 'resources.json').read_text())
        for filename in ['input.ptx', 'module.cubin', 'ptxas.log', 'cuobjdump.log']:
            original_bytes = (directory / filename).read_bytes()
            (directory / filename).write_bytes(original_bytes + b'changed')
            reject('modified ' + filename + ' rejected', lambda: load_resources(directory, negative))
            (directory / filename).write_bytes(original_bytes)
        for field, replacement in [('architecture', 'sm_80'), ('gpu', 'nvidia/a100'), ('hardware_validated', True),
                                    ('functional_run_exit_code', 4), ('ptx_sha256', '0' * 64), ('schema', 2)]:
            modified = deepcopy(saved); modified[field] = replacement
            (directory / 'resources.json').write_text(json.dumps(modified))
            reject('modified report ' + field + ' rejected', lambda: load_resources(directory, negative))
        modified = deepcopy(saved); modified['resources']['entries'][next(iter(modified['resources']['entries']))]['registers'] += 1
        (directory / 'resources.json').write_text(json.dumps(modified))
        reject('report register count differs from logs rejected', lambda: load_resources(directory, negative))
        modified = deepcopy(saved); modified['tools']['ptxas']['release'] = 'V1.0'
        (directory / 'resources.json').write_text(json.dumps(modified))
        reject('unexpected compiler release rejected', lambda: load_resources(directory, negative))
        modified = deepcopy(saved); modified['command'][2] = '-arch=sm_80'
        (directory / 'resources.json').write_text(json.dumps(modified))
        reject('modified compiler command rejected', lambda: load_resources(directory, negative))
        modified = deepcopy(saved); del modified['tools']
        (directory / 'resources.json').write_text(json.dumps(modified))
        reject('malformed report missing fields rejected', lambda: load_resources(directory, negative))
        (directory / 'resources.json').write_text(json.dumps(saved))
        # Change only the simulated profile after the real compiler returns;
        # unchanged PTX/program hashes must not hide a changed run identity.
        import compiler_resources
        import subprocess
        run_metadata = (negative / 'run.json').read_bytes()
        actual_run = subprocess.run
        def change_profile_after_compilation(command, *args, **kwargs):
            result = actual_run(command, *args, **kwargs)
            if Path(command[0]).name == 'ptxas' and '-v' in command:
                changed = json.loads(run_metadata); changed['gpu'] = 'nvidia/a100'
                (negative / 'run.json').write_text(json.dumps(changed))
            return result
        try:
            compiler_resources.subprocess.run = change_profile_after_compilation
            reject('concurrent profile change cannot retain old compilation identity', lambda: compile_resources(negative, cache, directory))
            check('changed run identity cannot retain success marker', not (directory / 'resources.json').exists())
        finally:
            compiler_resources.subprocess.run = actual_run
            (negative / 'run.json').write_bytes(run_metadata)
            (directory / 'resources.json').write_text(json.dumps(saved))
        ptx = (negative / 'kernel.ptx').read_bytes()
        (negative / 'kernel.ptx').write_bytes(ptx + b'\n// changed\n')
        reject('run PTX changed since compilation rejected', lambda: load_resources(directory, negative))
        # With the changed input acknowledged by a new compilation identity,
        # malformed PTX must fail ptxas and remove the old success marker.
        malformed = b'not valid PTX\n'; (negative / 'kernel.ptx').write_bytes(malformed)
        metadata = json.loads((negative / 'run.json').read_text())
        metadata['compilation_sha256']['kernel.ptx'] = hashlib.sha256(malformed).hexdigest()
        (negative / 'run.json').write_text(json.dumps(metadata))
        reject('real failed ptxas compilation rejected', lambda: compile_resources(negative, cache, directory))
        check('failed compilation cannot reuse stale success report', not (directory / 'resources.json').exists())
        report['passed'] = True
    finally:
        report['total_checks'] = len(checks)
        (out / 'results.json').write_text(json.dumps(report, indent=2))
    print('PASS compiler resource conformance checks=' + str(len(checks)), flush=True)
    return 0
