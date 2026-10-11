"""Analytical SM references and numerical CUDA replay with independent audits."""
import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sys

def verify_execution(cache, out, replay_from=None):
    from tsim import ROOT, run_program
    from sm_execution import analyze_execution, default_config, Pipeline, classify_issue, load_config
    from register_dependencies import analyze_dependencies
    from compiler_resources import compile_resources
    from sector_memory_service import MemoryConfig, load_memory_config
    sys.path.insert(0, str(ROOT / 'tests'))
    from sm_execution_components import verify_execution_components
    from sm_schedule_reference import verify_schedule
    from sm_memory_components import verify_sm_memory
    from sector_memory_components import verify_sector_memory
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=True)
    checks = verify_execution_components()
    checks.extend(verify_sm_memory()); checks.extend(verify_sector_memory())
    def digest(path): return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    report = {'kind': 'hand-solved SM execution and independently audited CUDA schedule replay; uncalibrated',
              'passed': False, 'checks': checks, 'cases': [], 'hardware_cycles': False,
              'model_sha256': {n: digest(ROOT / 'scripts' / n) for n in ['sm_execution.py', 'sm_residency.py', 'register_dependencies.py', 'execution_events.py', 'collective_events.py', 'cta_barrier_events.py', 'memory_model.py', 'cache_hierarchy.py', 'sector_read_service.py', 'sector_memory_service.py', 'execution_conformance.py']},
              'reference_sha256': {n: digest(ROOT / 'tests' / n) for n in ['sm_execution_components.py', 'sm_schedule_reference.py', 'sm_memory_components.py', 'sector_memory_components.py', 'sector_service_reference.py', 'conformance/carry_dependencies.cu', 'conformance/collective_dependencies.cu', 'conformance/collective_discarded.cu', 'conformance/execution_launches.cu', 'conformance/collective_wait_reads.cu', 'conformance/device_calls.cu', 'conformance/cache_policies.cu', 'conformance/cache_hint.cu', 'conformance/memory_spaces.cu']},
              'example_sha256': {n: digest(ROOT / 'examples' / n) for n in ['vector_add.cu', 'shared_reduce.cu']},
              'writer_sha256': digest(ROOT / 'include/tsim/execution_events.hpp'),
              'patch_sha256': {p.name: digest(p) for p in sorted((ROOT / 'patches').glob('*.patch'))}}
    def check(name, condition):
        if not condition: raise AssertionError(name)
        checks.append(name)
    def reject(name, action, message=None):
        try: action()
        except ValueError as error:
            if message and message not in str(error): raise
            checks.append(name)
        else: raise AssertionError(name)
    compiled = {}
    if replay_from:
        for record in sorted(Path(replay_from).glob('*/run.json')):
            metadata = json.loads(record.read_text()); source = Path(metadata['source']); snapshot = record.parent / 'source.cu'
            if source.is_file() and snapshot.is_file() and snapshot.read_bytes() == source.read_bytes() and not metadata['nvcc_args']:
                compiled.setdefault(str(source), record.parent)
    def run(name, source, gpu='h100', schedule='deterministic', limit=400000, workers=1):
        args = argparse.Namespace(source=source, gpu=gpu, trace=False, trace_limit=200000, events=True,
            events_limit=limit, scheduler=schedule, seed=79, threads=workers, race=True, strict=True,
            timeout=60, out=out / name, nvcc_arg=[])
        code, path = run_program(args, cache, compiled_from=compiled.get(str(source)))
        compiled.setdefault(str(source), path); check(name + ' complete numerical reference', code == 0)
        return path
    def audit(name, path, config=None, compiler=None, memory=None):
        result = analyze_execution(path, config, compiler, memory); graph = analyze_dependencies(path)
        check(name + ' independent hazard/port/queue/admission/retirement interval audit', verify_schedule(result, graph))
        (path / (name + '-sm-execution.json')).write_text(json.dumps(result, indent=2))
        report['cases'].append({'name': name, 'event_sha256': result['event_sha256'], 'ptx_sha256': result['ptx_sha256'],
                                'artifact_directory': path.name, 'timed_memory': memory is not None,
                                'config_sha256': hashlib.sha256(json.dumps(result['config'], sort_keys=True).encode()).hexdigest(),
                                'memory_config_sha256': hashlib.sha256(json.dumps(result['memory_config'], sort_keys=True).encode()).hexdigest() if memory else None,
                                'model_cycles': result['model_cycles'], 'issues': len(graph['nodes']),
                                'compiler_artifact_sha256': result['compiler_artifact_sha256'],
                                'physical_register_allocation_verified': result['physical_register_allocation_verified']})
        return result, graph
    try:
        base = default_config(); memory = MemoryConfig(); vector = None; timed_reference = None
        for gpu in ['a100', 'h100']:
            for schedule in ['deterministic', 'random', 'adversarial']:
                for label, source in [('vector', ROOT / 'examples/vector_add.cu'),
                                      ('carry', ROOT / 'tests/conformance/carry_dependencies.cu'),
                                      ('inplace', ROOT / 'tests/conformance/collective_dependencies.cu'),
                                      ('discarded', ROOT / 'tests/conformance/collective_discarded.cu')]:
                    name = label + '-' + gpu + '-' + schedule
                    path = run(name, source, gpu, schedule); result, graph = audit(name, path)
                    staged, _ = audit(name + '-staged', path, memory=memory)
                    if label == 'vector' and gpu == 'h100' and schedule == 'deterministic': vector = (path, result, graph)
                    if label == 'vector' and gpu == 'h100' and schedule == 'deterministic': timed_reference = staged
        path, reference, graph = vector
        shared_pipes = {name: Pipeline(p.latency, 2, 1, 'sm') for name, p in base.pipelines.items()}
        for count in [1, 2]: audit('vector-shared-ports-' + str(count) + '-sm', path, replace(base, sm_count=count, pipelines=shared_pipes))
        audit('vector-control-resolution', path, replace(base, pipelines=dict(base.pipelines, control=Pipeline(7, 1, 4))))
        compile_resources(path, cache)
        physical, _ = audit('vector-physical-registers', path, compiler=path / 'compiler-resources')
        audit('vector-physical-staged', path, compiler=path / 'compiler-resources', memory=memory)
        audit('vector-finite-miss-queues', path, memory=replace(memory, mshrs=1, l2_slots=1, l2_queue=1, writeback_slots=4))
        audit('vector-two-sm-staged', path, config=replace(base, sm_count=2), memory=replace(memory, sm_count=2))
        check('physical allocation enters admission without replacing virtual dependency identities', physical['physical_register_allocation_verified'] and
              physical['launches'][0]['compiler_allocation']['registers'] != physical['launches'][0]['ptx_estimated_registers_per_thread'] and
              physical['launches'][0]['ptx_estimated_registers_per_thread'] == reference['launches'][0]['ptx_estimated_registers_per_thread'])
        for name, mutation in [('early hazard issue', lambda r: r['launches'][0]['issues'][-1].update(issue_cycle=0)),
                               ('premature block release', lambda r: r['launches'][0]['blocks'][0].update(release_cycle=0)),
                               ('invented resident integral', lambda r: r['launches'][0]['resident_warp_cycle_integral'].__setitem__(0, 0)),
                               ('invented hardware calibration', lambda r: r.update(calibrated=True))]:
            altered = deepcopy(reference); mutation(altered)
            try: verify_schedule(altered, graph)
            except AssertionError: checks.append('independent audit rejects ' + name)
            else: raise AssertionError('Independent audit accepted ' + name)
        multiple = run('multiple-launches', ROOT / 'tests/conformance/execution_launches.cu')
        result, _ = audit('multiple-launches', multiple)
        audit('multiple-launches-staged', multiple, memory=memory)
        audit('multiple-launches-slow-ports', multiple,
              config=replace(base, pipelines={n: replace(p, interval=100) for n, p in base.pipelines.items()}),
              memory=replace(memory, lsu_interval=100, l2_interval=100, dram_interval=100))
        check('separate launch lifetimes drain sequentially', len(result['launches']) > 1)
        parallel = run('inplace-four-workers', ROOT / 'tests/conformance/collective_dependencies.cu', workers=4)
        audit('inplace-four-workers', parallel)
        audit('inplace-four-workers-staged', parallel, memory=memory)
        policy_path = run('cache-policies', ROOT / 'tests/conformance/cache_policies.cu')
        policy_result, _ = audit('cache-policies', policy_path, memory=memory)
        check('actual independent ca/cg loads preserve L1 bypass and overlapping miss merges',
              any(r['policy']['cache_operator'] == 'ca' for r in policy_result['memory_policies']) and
              any(r['policy']['cache_operator'] == 'cg' for r in policy_result['memory_policies']) and
              policy_result['launches'][0]['memory_service']['stats']['merged_miss_sectors'] > 0)
        warm_result, _ = audit('cache-policies-one-lsu-slot', policy_path, memory=replace(memory, lsu_slots_per_sm=1))
        check('actual repeated ca load observes a completed fill when one LSU slot serializes requests', warm_result['launches'][0]['memory_service']['stats']['l1_hit_sectors'] > 0)
        for label, source, message in [('cache-hint', ROOT / 'tests/conformance/cache_hint.cu', 'Unsupported timed cache policy'),
                                      ('other-spaces', ROOT / 'tests/conformance/memory_spaces.cu', 'global load/store')]:
            unsupported = run('unsupported-' + label, source)
            reject(label + ' refuses undeclared timed memory behavior', lambda path=unsupported: analyze_execution(path, memory_config=memory), message)
        for label, mutation in [('invented memory response', lambda r: r['launches'][0]['memory_service']['requests'][0].update(complete_cycle=0)),
                                 ('missing reservation history', lambda r: r['launches'][0]['memory_service']['resource_records'].pop()),
                                 ('invented backing bandwidth', lambda r: r['launches'][0]['memory_service']['writebacks'][0].update(start_cycle=0))]:
            altered = deepcopy(timed_reference); mutation(altered)
            try: verify_schedule(altered, graph)
            except AssertionError: checks.append('independent audit rejects ' + label)
            else: raise AssertionError('Independent audit accepted ' + label)
        for label, source, message in [('volatile', ROOT / 'tests/conformance/collective_wait_reads.cu', 'Memory ordering'),
                                       ('barrier', ROOT / 'examples/shared_reduce.cu', 'execution pipeline'),
                                       ('device-call', ROOT / 'tests/conformance/device_calls.cu', 'call ABI')]:
            path = run('unsupported-' + label, source)
            reject(label + ' refuses undeclared readiness', lambda path=path: analyze_execution(path), message)
        capped = run('capped', ROOT / 'examples/vector_add.cu', limit=8)
        reject('capped capture refuses modeled duration', lambda: analyze_execution(capped))
        altered = out / 'changed-events'; shutil.copytree(vector[0], altered, dirs_exist_ok=True)
        with (altered / 'execution-events.jsonl').open('a') as stream: stream.write('{"type":"unknown"}\n')
        reject('changed event stream refuses modeled duration', lambda: analyze_execution(altered))
        node = {'live_ins': [], 'undefined_results': []}
        for label, instruction in [('missing pipeline', 'sqrt.rn.f32 %f1, %f2;'), ('additional floating type', 'add.f64 %d1, %d2, %d3;'),
                                   ('atomic readiness', 'atom.global.add.u32 %r1, [%rd1], %r2;')]:
            reject(label, lambda instruction=instruction: classify_issue({'instruction': instruction, 'execution_mask': 1}, node))
        reject('unknown input is never assumed ready', lambda: classify_issue({'instruction': 'mov.u32 %r1, %r2;', 'execution_mask': 1}, dict(node, live_ins=[1])))
        reject('legacy parked operand ambiguity refuses timing', lambda: classify_issue({'instruction': 'vote.sync.ballot.b32 %r1, %p1, %r2;', 'execution_mask': 0}, dict(node, conservative_selection_reads=True)))
        config_path = out / 'timing assumptions.json'; config_path.write_text(json.dumps(base.record(), indent=2))
        check('configuration round trip preserves all declared assumptions', load_config(config_path).record() == base.record())
        for label, mutation in [('extra configuration field', lambda d: d.update(guessed=True)),
                               ('unverified measurement claim', lambda d: d.update(provenance_kind='measured')),
                               ('boolean schema', lambda d: d.update(schema=True)),
                               ('missing pipeline field', lambda d: d['pipelines']['alu'].pop('latency'))]:
            data = deepcopy(base.record()); mutation(data); config_path.write_text(json.dumps(data))
            reject(label + ' refused', lambda: load_config(config_path))
        config_path.write_text(json.dumps(base.record(), indent=2))
        memory_path = out / 'memory assumptions.json'; memory_path.write_text(json.dumps(memory.record(), indent=2))
        check('memory configuration round trip preserves declared assumptions', load_memory_config(memory_path).record() == memory.record())
        for label, mutation in [('unknown memory field', lambda d: d.update(guessed=True)),
                               ('unverified memory calibration', lambda d: d.update(provenance_kind='measured')),
                               ('boolean memory schema', lambda d: d.update(schema=True))]:
            data = deepcopy(memory.record()); mutation(data); memory_path.write_text(json.dumps(data))
            reject(label + ' refused', lambda: load_memory_config(memory_path))
        memory_path.write_text(json.dumps(memory.record(), indent=2))
        report['passed'] = True
    finally:
        report['total_checks'] = len(checks); (out / 'results.json').write_text(json.dumps(report, indent=2))
    print('PASS SM execution conformance checks=' + str(len(checks)) + ' audited schedules=' + str(len(report['cases'])), flush=True)
    return 0
