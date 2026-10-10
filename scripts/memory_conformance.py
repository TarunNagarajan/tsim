"""Analytical components and end-to-end observed transaction references."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

def verify_memory(cache, out, replay_from=None):
    from tsim import ROOT, run_program
    from memory_model import analyze_transactions
    sys.path.insert(0, str(ROOT / 'tests'))
    from memory_model_components import verify_components
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=True)
    checks = verify_components()
    report = {'kind': 'analytical transaction/cache components; not NVIDIA hardware differential testing',
              'checks': checks, 'passed': False, 'hardware_cycles': False,
              'component_sha256': {name: hashlib.sha256((ROOT / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                                   for name in ['scripts/memory_model.py', 'scripts/execution_events.py']}}
    compiled = {}
    if replay_from:
        for record in sorted(Path(replay_from).glob('*/run.json')):
            metadata = json.loads(record.read_text())
            source = Path(metadata['source'])
            if source.is_file() and (record.parent / 'source.cu').read_bytes() == source.read_bytes() and not metadata['nvcc_args']:
                compiled.setdefault(str(source), record.parent)
    def check(name, condition):
        if not condition: raise AssertionError(name)
        checks.append(name)
    def run(name, source, gpu='h100', scheduler='deterministic', cap=400000, negative=False):
        args = argparse.Namespace(source=source, gpu=gpu, trace=False, trace_limit=200000,
            events=True, events_limit=cap, scheduler=scheduler, seed=23, threads=1,
            race=True, strict=True, timeout=60, out=out / name, nvcc_arg=[])
        code, path = run_program(args, cache, compiled_from=compiled.get(str(source)))
        compiled.setdefault(str(source), path)
        check(name + ' process', code != 0 if negative else code == 0)
        return path
    def analyze(path):
        model = analyze_transactions(path)
        (path / 'memory-transactions.json').write_text(json.dumps(model, indent=2))
        return model
    try:
        for gpu in ['a100', 'h100']:
            vector = run('vector-' + gpu, ROOT / 'examples/vector_add.cu', gpu)
            model = analyze(vector)
            expected = {'global_load_sectors': 66, 'global_store_sectors': 33,
                        'global_load_requested_bytes': 2072, 'global_store_requested_bytes': 1036}
            check(gpu + ' vector exact sector conservation', all(model['totals'][k] == n for k, n in expected.items()))
            index = run('index-' + gpu, ROOT / 'examples/index_3d.cu', gpu)
            model = analyze(index)
            check(gpu + ' 3D scattered stores', model['totals']['global_store_sectors'] == 32 and
                  model['totals']['global_store_requested_bytes'] == 384)
        publication = run('publication', ROOT / 'tests/conformance/atomic_protocol.cu')
        model = analyze(publication)
        check('generic shared publication has no global-load sectors', model['totals'].get('global_load_sectors', 0) == 0 and
              model['totals']['global_atomic_lane_accesses'] == 64 and model['totals']['shared_load_requested_bytes'] >= 256)
        reduce = run('reduce', ROOT / 'examples/shared_reduce.cu')
        model = analyze(reduce)
        check('synchronized reduction topology', model['totals']['shared_load_lane_accesses'] == 255 and
              all(r.get('bank_conflict_degree', 1) == 1 for r in model['requests']))
        spaces = run('spaces', ROOT / 'tests/conformance/memory_spaces.cu')
        model = analyze(spaces)
        check('private local stripes preserve bytes and owners', model['totals']['local_load_requested_bytes'] == 3 * 67 * 37 * 4 and
              model['totals']['local_store_requested_bytes'] == 3 * 67 * 37 * 4 and model['totals']['local_load_sectors'] == 999)
        check('constant address requests stay distinct', model['totals']['constant_load_distinct_constant_addresses'] > 0)
        constant = run('constant-distinct', ROOT / 'tests/conformance/constant_transactions.cu')
        model = analyze(constant)
        check('CUDA divergent constant addresses', model['totals']['constant_load_distinct_constant_addresses'] == 32 and
              model['totals']['constant_load_requested_bytes'] == 128)
        for name, source, cap, negative in [('cap', ROOT / 'examples/vector_add.cu', 37, False),
            ('fault', ROOT / 'examples/out_of_bounds.cu', 400000, True),
            ('gap', ROOT / 'tests/conformance/async_event_gap.cu', 400000, False)]:
            path = run(name, source, cap=cap, negative=negative)
            try: analyze_transactions(path)
            except ValueError: checks.append(name + ' refuses incomplete model input')
            else: raise AssertionError('Incomplete capture accepted: ' + name)
        # Changing the PTX after capture must invalidate provenance.
        path = vector / 'kernel.ptx'; original = path.read_bytes()
        try:
            path.write_bytes(original + b'\n// changed\n')
            try: analyze_transactions(vector)
            except ValueError as e: check('changed compilation rejected', 'artifact changed' in str(e))
            else: raise AssertionError('Changed PTX accepted')
        finally: path.write_bytes(original)
        report['passed'] = True
    finally:
        report['total_checks'] = len(checks)
        (out / 'results.json').write_text(json.dumps(report, indent=2))
    print('PASS memory model conformance checks=' + str(len(checks)), flush=True)
    return 0
