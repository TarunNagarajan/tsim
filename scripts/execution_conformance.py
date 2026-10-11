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
    from cta_dependency_reference import verify_cta_dependencies
    from sm_memory_components import verify_sm_memory
    from sector_memory_components import verify_sector_memory
    from cta_timing_components import verify_cta_timing
    from shared_memory_components import verify_shared_components
    from sm_shared_components import verify_sm_shared
    from shared_memory_service import SharedConfig,load_shared_config
    from memory_system_reference import verify_captured_memory
    from global_atomic_service import AtomicConfig,load_atomic_config
    from global_atomic_components import verify_global_atomic_components
    from sm_atomic_components import verify_sm_atomics
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=True)
    checks = verify_execution_components()
    checks.extend(verify_sm_memory()); checks.extend(verify_sector_memory()); checks.extend(verify_cta_timing())
    checks.extend(verify_shared_components());checks.extend(verify_sm_shared())
    checks.extend(verify_global_atomic_components());checks.extend(verify_sm_atomics())
    def digest(path): return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
    report = {'kind': 'hand-solved SM execution and independently audited CUDA schedule replay; uncalibrated',
              'passed': False, 'checks': checks, 'cases': [], 'hardware_cycles': False,
              'model_sha256': {n: digest(ROOT / 'scripts' / n) for n in ['sm_execution.py', 'sm_residency.py', 'register_dependencies.py', 'execution_events.py', 'collective_events.py', 'cta_barrier_events.py', 'cta_barrier_report.py', 'cta_barrier_timing.py', 'memory_model.py', 'cache_hierarchy.py', 'sector_read_service.py', 'sector_memory_service.py', 'execution_conformance.py']},
              'reference_sha256': {n: digest(ROOT / 'tests' / n) for n in ['cta_timing_components.py', 'cta_timing_reference.py', 'cta_dependency_reference.py', 'register_dependency_components.py', 'conformance/cta_timing.cu', 'sm_execution_components.py', 'sm_schedule_reference.py', 'sm_memory_components.py', 'sector_memory_components.py', 'sector_service_reference.py', 'conformance/carry_dependencies.cu', 'conformance/collective_dependencies.cu', 'conformance/collective_discarded.cu', 'conformance/execution_launches.cu', 'conformance/collective_wait_reads.cu', 'conformance/device_calls.cu', 'conformance/cache_policies.cu', 'conformance/cache_hint.cu', 'conformance/memory_spaces.cu']},
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
    for n in ['shared_memory_service.py','memory_system.py','global_atomic_service.py']:report['model_sha256'][n]=digest(ROOT/'scripts'/n)
    for n in ['shared_memory_components.py','shared_service_reference.py','sm_shared_components.py','memory_system_reference.py','conformance/shared_bank_timing.cu',
        'global_atomic_components.py','global_atomic_reference.py','sm_atomic_components.py','conformance/global_atomic_timing.cu']:report['reference_sha256'][n]=digest(ROOT/'tests'/n)
    def audit(name, path, config=None, compiler=None, memory=None,shared=None,atomic=None):
        result = analyze_execution(path, config, compiler, memory,shared,atomic); graph = analyze_dependencies(path)
        check(name + ' independent hazard/port/queue/admission/retirement interval audit', verify_schedule(result, graph) and verify_cta_dependencies([json.loads(row) for row in (path/'execution-events.jsonl').read_text().splitlines()],graph))
        if shared or atomic:check(name+' independent original captured byte routing',verify_captured_memory([json.loads(row) for row in (path/'execution-events.jsonl').read_text().splitlines()],result))
        (path / (name + '-sm-execution.json')).write_text(json.dumps(result, indent=2))
        report['cases'].append({'name': name, 'event_sha256': result['event_sha256'], 'ptx_sha256': result['ptx_sha256'],
                                'artifact_directory': path.name, 'timed_memory': memory is not None,
                                'config_sha256': hashlib.sha256(json.dumps(result['config'], sort_keys=True).encode()).hexdigest(),
                                'memory_config_sha256': hashlib.sha256(json.dumps(result['memory_config'], sort_keys=True).encode()).hexdigest() if memory else None,
                                'shared_config_sha256':hashlib.sha256(json.dumps(result['shared_config'],sort_keys=True).encode()).hexdigest() if shared else None,
                                'atomic_config_sha256':hashlib.sha256(json.dumps(result['atomic_config'],sort_keys=True).encode()).hexdigest() if atomic else None,
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
                                       ('device-call', ROOT / 'tests/conformance/device_calls.cu', 'call ABI')]:
            path = run('unsupported-' + label, source)
            reject(label + ' refuses undeclared readiness', lambda path=path: analyze_execution(path), message)
        for gpu in ['a100','h100']:
            for schedule in ['deterministic','random','adversarial']:
                name='cta-'+gpu+'-'+schedule
                path=run(name,ROOT/'tests/conformance/cta_timing.cu',gpu,schedule,workers=4)
                compile_resources(path,cache)
                compiler=path/'compiler-resources'
                result,cta_graph=audit(name,path,compiler=compiler)
                audit(name+'-staged',path,compiler=compiler,memory=memory)
                audit(name+'-finite-release',path,config=replace(base,cta_barrier=Pipeline(7,3,1,'sm')),compiler=compiler,memory=replace(memory,lsu_slots_per_sm=1,mshrs=1,l2_slots=1,l2_queue=1))
                audit(name+'-two-sm',path,config=replace(base,sm_count=2),compiler=compiler,memory=replace(memory,sm_count=2))
                check(name+' actual phase and reduction data ownership',len(cta_graph['cta_phases'])==80 and cta_graph['counts']['CTA_REDUCTION']>0 and not cta_graph['counts'].get('live_in_lane_reads'))
                if gpu=='h100' and schedule=='random': cta_reference,cta_reference_graph=result,cta_graph
        shared=run('cta-shared-fixed',ROOT/'examples/shared_reduce.cu')
        audit('cta-shared-fixed',shared)
        reject('shared timed service is refused until bank completion is modeled',lambda:analyze_execution(shared,memory_config=memory),'global load/store')
        audit('cta-shared-banked',shared,memory=memory,shared=SharedConfig())
        for gpu in ['a100','h100']:
            for schedule in ['deterministic','random','adversarial']:
                name='shared-banks-'+gpu+'-'+schedule
                bank_path=run(name,ROOT/'tests/conformance/shared_bank_timing.cu',gpu,schedule,workers=4)
                compile_resources(bank_path,cache);bank_compiler=bank_path/'compiler-resources'
                banked,bank_graph=audit(name,bank_path,compiler=bank_compiler,memory=memory,shared=SharedConfig())
                audit(name+'-finite',bank_path,compiler=bank_compiler,memory=replace(memory,lsu_slots_per_sm=1,mshrs=1,l2_slots=1,l2_queue=1),
                    shared=SharedConfig(bank_slots=1,bank_queue=1,request_slots_per_sm=1))
                audit(name+'-two-sm',bank_path,config=replace(base,sm_count=2),compiler=bank_compiler,memory=replace(memory,sm_count=2),shared=SharedConfig(sm_count=2))
                check(name+' actual mixed/vector/subword/wide/atomic traffic and CTA handoffs',
                    any(len(r['routes'])==2 for l in banked['launches'] for r in l['memory_system']['requests']) and
                    {1,2,4,8}<={a['width'] for l in banked['launches'] for r in l['shared_service']['requests'] for a in r['request']['accesses']} and
                    any(a['element']==3 for l in banked['launches'] for r in l['shared_service']['requests'] for a in r['request']['accesses']) and
                    any(j['atomic'] and len(j['words'])==2 for l in banked['launches'] for j in l['shared_service']['jobs']) and
                    len(bank_graph['cta_phases'])==12)
        shared_path=out/'shared assumptions.json';shared_path.write_text(json.dumps(SharedConfig().record()))
        check('shared configuration round trip preserves all assumptions',load_shared_config(shared_path).record()==SharedConfig().record())
        for gpu in ['a100','h100']:
            for schedule in ['deterministic','random','adversarial']:
                atomic_name='global-atomics-'+gpu+'-'+schedule
                atomic_path=run(atomic_name,ROOT/'tests/conformance/global_atomic_timing.cu',gpu,schedule,workers=4)
                compile_resources(atomic_path,cache);atomic_compiler=atomic_path/'compiler-resources'
                atomics,atomic_graph=audit(atomic_name,atomic_path,compiler=atomic_compiler,memory=memory,shared=SharedConfig(),atomic=AtomicConfig())
                audit(atomic_name+'-finite',atomic_path,compiler=atomic_compiler,
                    memory=replace(memory,lsu_slots_per_sm=1,mshrs=1,l2_slots=1,l2_queue=1,cache_sets=1,cache_ways=1,writeback_slots=4),
                    shared=SharedConfig(bank_slots=1,bank_queue=1,request_slots_per_sm=1),atomic=AtomicConfig(queue=1,slots=1))
                audit(atomic_name+'-two-sm',atomic_path,config=replace(base,sm_count=2),compiler=atomic_compiler,
                    memory=replace(memory,sm_count=2),shared=SharedConfig(sm_count=2),atomic=AtomicConfig())
                check(atomic_name+' exact distinct lane updates, mixed routing and operator results',
                    sum(l['memory_service']['atomic_service']['stats']['committed_updates'] for l in atomics['launches'])==5952 and
                    any(len(r['routes'])==2 for l in atomics['launches'] for r in l['memory_system']['requests']) and
                    len(atomic_graph['cta_phases'])==12 and 'comparisons=6600 launches=4' in (atomic_path/'stdout.log').read_text())
        atomic_path=out/'atomic assumptions.json';atomic_path.write_text(json.dumps(AtomicConfig().record()))
        check('atomic configuration round trip preserves declared assumptions',load_atomic_config(atomic_path).record()==AtomicConfig().record())
        for label,mutation in [('unknown RMW field',lambda d:d.update(guessed=True)),('unverified atomic calibration',lambda d:d.update(provenance_kind='measured')),
            ('boolean RMW schema',lambda d:d.update(schema=True))]:
            data=AtomicConfig().record();mutation(data);atomic_path.write_text(json.dumps(data));reject(label,lambda:load_atomic_config(atomic_path))
        atomic_path.write_text(json.dumps(AtomicConfig().record()))
        for label,mutation in [
            ('early release',lambda r:r['launches'][0]['cta_barrier_service']['phases'][0].update(start_cycle=0)),
            ('missing memory prefix',lambda r:r['launches'][0]['cta_barrier_service']['phases'][0].update(memory_prior_issues=[])),
            ('borrowed phase reset',lambda r:r['launches'][0]['cta_barrier_service']['phases'][-1].update(previous_phase=None)),
            ('invented release capacity',lambda r:r['launches'][0]['cta_barrier_service']['peak_inflight_per_sm'].clear())]:
            altered=deepcopy(cta_reference); mutation(altered)
            try:verify_schedule(altered,cta_reference_graph)
            except AssertionError:checks.append('independent CTA audit rejects '+label)
            else:raise AssertionError('Independent CTA audit accepted '+label)
        dependency_events=[json.loads(row) for row in (out/'cta-h100-random'/'execution-events.jsonl').read_text().splitlines()]
        for label,mutation in [
            ('missing reduction contributor',lambda n:n['edges'].remove(next(e for e in n['edges'] if e['kind']=='CTA_REDUCTION'))),
            ('invented source lanes',lambda n:next(e for e in n['edges'] if e['kind']=='CTA_REDUCTION').update(lane_mask=1)),
            ('invented destination lanes',lambda n:next(e for e in n['edges'] if e['kind']=='CTA_REDUCTION').update(destination_lane_mask=1)),
            ('wrong saved phase answer',lambda n:n['cta_result'].update(result=999)),
            ('wrong operand action',lambda n:n.update(cta_action='cta_barrier_arrive'))]:
            altered=deepcopy(cta_reference_graph); node=next(n for n in altered['nodes'] if n.get('cta_result')); mutation(node)
            try:verify_cta_dependencies(dependency_events,altered)
            except AssertionError:checks.append('independent CTA dependency audit rejects '+label)
            else:raise AssertionError('Independent dependency audit accepted '+label)
        reject('missing release assumptions refuses CTA duration',lambda:analyze_execution(path,replace(base,cta_barrier=None),compiler_directory=compiler),'declared release service')
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
