"""Analytical hierarchy references plus compiled full-opcode traffic checks."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

def verify_cache(cache, out, replay_from=None):
    from tsim import ROOT, run_program
    from cache_hierarchy import analyze_cache, DEFAULT_PROFILE
    sys.path.insert(0, str(ROOT / 'tests'))
    from cache_hierarchy_components import verify_cache_components
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=True)
    checks = verify_cache_components()
    report = {'kind': 'synthetic hierarchy conformance; no hardware calibration', 'passed': False,
              'checks': checks, 'hardware_cycles': False,
              'component_sha256': {name: hashlib.sha256((ROOT / name).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                                   for name in ['scripts/cache_hierarchy.py', 'scripts/memory_model.py']}}
    compiled = {}
    if replay_from:
        for record in Path(replay_from).glob('*/run.json'):
            metadata = json.loads(record.read_text()); source = Path(metadata['source'])
            if source.is_file() and (record.parent / 'source.cu').read_bytes() == source.read_bytes() and not metadata['nvcc_args']:
                compiled.setdefault(str(source), record.parent)
    def check(name, condition):
        if not condition: raise AssertionError(name)
        checks.append(name)
    def run(name, source, gpu='h100', workers=1):
        args = argparse.Namespace(source=source, gpu=gpu, trace=False, trace_limit=200000,
            events=True, events_limit=400000, scheduler='deterministic', seed=31, threads=workers,
            race=True, strict=True, timeout=60, out=out / name, nvcc_arg=[])
        code, path = run_program(args, cache, compiled_from=compiled.get(str(source)))
        compiled.setdefault(str(source), path)
        check(name + ' process', code == 0)
        return path
    def analyze(path, profile=None):
        model = analyze_cache(path, profile)
        (path / 'cache-hierarchy.json').write_text(json.dumps(model, indent=2))
        return model
    try:
        for gpu in ['a100', 'h100']:
            path = run('policies-' + gpu, ROOT / 'tests/conformance/cache_policies.cu', gpu)
            ptx = (path / 'kernel.ptx').read_text()
            check(gpu + ' compiler preserves ca/cg store and loads', ptx.count('ld.global.ca.u32') == 2 and
                  ptx.count('ld.global.cg.u32') == 2 and ptx.count('st.global.cg.u32') == 1)
            model = analyze(path)
            requests = model['request_records']; launch = model['launches']['1']
            check(gpu + ' exact CUDA cache traffic reference', len(requests) == 5 and
                  sum(r['l1_hits'] for r in requests) == 4 and sum(r['l1_misses'] for r in requests) == 4 and
                  sum(r['l1_bypass'] for r in requests) == 12 and sum(r['l2_hits'] for r in requests) == 8 and
                  sum(r['l2_misses'] for r in requests) == 8 and model['totals']['logical_request_sectors'] == 20 and
                  model['totals']['backing_read_sectors'] == 8 and launch['traffic']['backing_terminal_write_sectors'] == 4)
        path = run('hint', ROOT / 'tests/conformance/cache_hint.cu')
        check('compiled eviction hint survives functional parser in captured instruction', any(
            'ld.global.L1::evict_last.u32' in r.get('instruction', '')
            for r in map(json.loads, (path / 'execution-events.jsonl').read_text().splitlines())))
        try: analyze_cache(path)
        except ValueError as error: check('CUDA eviction hint refuses unsupported hierarchy', 'does not yet model hints' in str(error))
        else: raise AssertionError('Unsupported hint accepted')
        path = run('vector-workers', ROOT / 'examples/vector_add.cu', workers=4)
        config = deepcopy(DEFAULT_PROFILE); config['sm_count'] = 2
        model = analyze(path, config)
        check('multiworker vector sector and backing conservation', model['totals']['logical_request_sectors'] == 99 and
              model['totals']['backing_read_sectors'] == 99 and model['totals']['backing_terminal_write_sectors'] == 33)
        check('explicit synthetic placement distributes captured blocks', {r['sm'] for r in model['request_records']} == {0, 1})
        path = run('private-launches', ROOT / 'tests/conformance/memory_spaces.cu')
        model = analyze(path)
        check('three launches each start cold and drain independently', len(model['launches']) == 3 and all(
            summary['traffic']['backing_read_sectors'] > 0 for summary in model['launches'].values()))
        check('local cache ownership remains scoped to launch/block/warp', all(
            r['ownership'].startswith('local:' + str(r['launch']) + ':') for r in model['request_records'] if r['space'] == 'local'))
        # Retaining text is insufficient if modified text could hide a hint.
        captured = path / 'execution-events.jsonl'; original = captured.read_bytes()
        try:
            rows = list(map(json.loads, original.splitlines()))
            issue = next(r for r in rows if r['type'] == 'issue' and 'ld.local.' in r['instruction'])
            issue['instruction'] = issue['instruction'].replace('ld.local.', 'ld.local.cg.')
            captured.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            try: analyze_cache(path)
            except ValueError as error: check('changed captured policy rejected against compiled PTX', 'differs from compiled PTX' in str(error))
            else: raise AssertionError('Modified captured policy accepted')
        finally: captured.write_bytes(original)
        for name, source, cap, negative in [('cap', ROOT / 'examples/vector_add.cu', 37, False),
                ('fault', ROOT / 'examples/out_of_bounds.cu', 400000, True),
                ('gap', ROOT / 'tests/conformance/async_event_gap.cu', 400000, False)]:
            args = argparse.Namespace(source=source, gpu='h100', trace=False, trace_limit=200000, events=True,
                events_limit=cap, scheduler='deterministic', seed=31, threads=1, race=True, strict=True,
                timeout=60, out=out / name, nvcc_arg=[])
            code, path = run_program(args, cache, compiled_from=compiled.get(str(source)))
            check(name + ' process', code != 0 if negative else code == 0)
            try: analyze_cache(path)
            except ValueError: checks.append(name + ' refuses hierarchy input')
            else: raise AssertionError('Incomplete capture accepted by hierarchy')
        report['passed'] = True
    finally:
        report['total_checks'] = len(checks)
        (out / 'results.json').write_text(json.dumps(report, indent=2))
    print('PASS cache hierarchy conformance checks=' + str(len(checks)), flush=True)
    return 0
