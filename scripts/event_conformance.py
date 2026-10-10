"""Analytical memory totals, identities, fault coverage, and capture equivalence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

def verify_events(cache, out, replay_from=None):
    from tsim import ROOT, run_program
    from execution_events import summarize_events
    sys.path.insert(0, str(ROOT / 'tests'))
    from engine_patches_model import verify_patches
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=True)
    checks = []
    report = {'kind': 'interpreter event conformance; analytical references, not hardware differential',
              'checks': checks, 'passed': False,
              'patch_sha256': {p.name: hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                               for p in sorted((ROOT / 'patches').glob('*.patch'))}}
    def check(name, condition):
        if not condition: raise AssertionError(name)
        checks.append(name)
    def rows(path):
        return [json.loads(line) for line in (path / 'execution-events.jsonl').read_text().splitlines()]
    compiled = {}
    if replay_from:
        for record_path in sorted(Path(replay_from).glob('*/run.json')):
            record = json.loads(record_path.read_text())
            source = Path(record['source'])
            saved = record_path.parent
            if source.is_file() and (saved / 'source.cu').read_bytes() == source.read_bytes() and not record['nvcc_args']:
                compiled.setdefault(str(source), saved)
    def run(name, source, capture=True, workers=1, scheduler='deterministic', limit=400000, negative=False):
        args = argparse.Namespace(source=source, gpu='h100', trace=True, trace_limit=200000,
            events=capture, events_limit=limit, scheduler=scheduler, seed=19, threads=workers,
            race=True, strict=True, timeout=60, out=out / name, nvcc_arg=[])
        code, path = run_program(args, cache, compiled_from=compiled.get(str(source)))
        compiled.setdefault(str(source), path)
        check(name + ' process', code != 0 if negative else code == 0)
        return path
    try:
        check('managed patch replay and edit protection', verify_patches(out))
        unit = out / 'event-writer-model'
        subprocess.run(['clang++', '-std=c++20', '-O2', '-pthread', '-I' + str(ROOT / 'include'),
                        str(ROOT / 'tests/execution_events_model.cpp'), '-o', str(unit)], check=True)
        def writer(name, path=None, limit=200000):
            env = os.environ.copy(); env['TSIM_EVENTS'] = str(path or out / (name + '.jsonl'))
            env['TSIM_EVENTS_LIMIT'] = str(limit)
            return subprocess.run([str(unit)], env=env, text=True, capture_output=True)
        check('four-worker writer exits', writer('writer').returncode == 0)
        data = [json.loads(line) for line in (out / 'writer.jsonl').read_text().splitlines()]
        parents = {r['sequence']: r for r in data if r['type'] == 'issue'}
        memory = [r for r in data if r['type'] == 'memory']
        check('thread-local parents preserve high 64-bit addresses', len(memory) == 120 and all(
            int(r['address'], 16) == 0xfedcba9876540000 + parents[r['issue']]['block'][0] for r in memory))
        check('writer escapes instruction strings', all(r['instruction'] == 'quoted"\n\\instruction' for r in parents.values()))
        check('bounded writer exits', writer('writer-cap', limit=97).returncode == 0)
        capped = [json.loads(line) for line in (out / 'writer-cap.jsonl').read_text().splitlines()]
        check('cap emits one terminal marker', len(capped) == 98 and capped[-1] == {'type': 'truncated', 'limit': 97})
        check('writer rejects invalid caps', writer('bad-limit', limit=0).returncode == 2)
        check('writer reports file-open failure', writer('bad-path', out / 'missing' / 'events.jsonl').returncode == 2)
        check('writer reports disk-write failure', writer('full-device', Path('/dev/full')).returncode == 2)

        vector = ROOT / 'examples/vector_add.cu'
        captured = run('vector', vector)
        plain = run('vector-plain', vector, capture=False)
        check('capture preserves numerical result', (captured / 'stdout.log').read_text() == (plain / 'stdout.log').read_text())
        summary = summarize_events(captured / 'execution-events.jsonl')
        check('vector exact global traffic', summary['memory']['global_load_accesses'] == 518 and
              summary['memory']['global_load_bytes'] == 2072 and summary['memory']['global_store_accesses'] == 259 and
              summary['memory']['global_store_bytes'] == 1036 and summary['complete_supported_memory'])
        events = rows(captured)
        ptx = (captured / 'kernel.ptx').read_text().splitlines()
        normalize = lambda text: re.sub(r'\s+', '', text).rstrip(';')
        check('event instructions align with generated PTX', all(normalize(r['instruction']) == normalize(ptx[r['ptx_line'] - 1])
              for r in events if r['type'] == 'issue'))
        check('typed register/predicate metadata', all('defs' in r and 'uses' in r and 'predicate_use' in r
              for r in events if r['type'] == 'issue') and any(r['predicate_use'] >= 0 and
              r['execution_mask'] != r['path_mask'] for r in events if r['type'] == 'issue'))
        workers = run('vector-workers', vector, workers=4)
        check('four-worker actual memory totals', summarize_events(workers / 'execution-events.jsonl')['memory'] == summary['memory'])
        for name, source in [('numeric-edges', ROOT / 'tests/conformance/numeric_edges.cu'),
                             ('tiled-matmul', ROOT / 'examples/tiled_matmul.cu'),
                             ('shared-reduce', ROOT / 'examples/shared_reduce.cu')]:
            captured_result = run(name, source)
            plain_result = run(name + '-plain', source, capture=False)
            check(name + ' reference/fast capture equivalence',
                  (captured_result / 'stdout.log').read_text() == (plain_result / 'stdout.log').read_text())
        a = run('vector-random', vector, scheduler='random')
        b = run('vector-replay', vector, scheduler='random')
        check('seeded event replay', rows(a) == rows(b))
        cap = run('vector-cap', vector, limit=37)
        capped = summarize_events(cap / 'execution-events.jsonl')
        check('truncated stream refuses complete coverage', capped['truncated'] and not capped['complete_supported_memory'])
        index = run('index-3d', ROOT / 'examples/index_3d.cu', workers=4)
        traffic = summarize_events(index / 'execution-events.jsonl')['memory']
        check('3D exact stores', traffic['global_store_accesses'] == 96 and traffic['global_store_bytes'] == 384)

        mixed = run('generic-mixed', ROOT / 'tests/conformance/generic_memory_events.cu')
        summary = summarize_events(mixed / 'execution-events.jsonl')
        expected = {'global_load_accesses': 21, 'global_store_accesses': 86,
                    'shared_load_accesses': 23, 'shared_store_accesses': 23,
                    'local_load_accesses': 21, 'local_store_accesses': 21,
                    'global_atomic_accesses': 64, 'shared_atomic_accesses': 64}
        check('generic space routing and scalar atomic counts', all(summary['memory'].get(k) == n for k, n in expected.items())
              and summary['complete_supported_memory'])
        counters = re.findall(r'global\s+ld=(\d+) st=(\d+)\s+read=(\d+) B write=(\d+) B',
                              (mixed / 'stderr.log').read_text())
        check('generic shared/local never inflate global counters', counters == [('21', '85', '84', '340'), ('0', '1', '0', '4')])
        check('generic PTX probe retained', all(op in (mixed / 'kernel.ptx').read_text() for op in ['st.u32', 'ld.u32', 'atom.']))
        publication = run('generic-publication', ROOT / 'tests/conformance/atomic_protocol.cu')
        summary = summarize_events(publication / 'execution-events.jsonl')
        check('generic acquire publication never becomes global loads',
              summary['complete_supported_memory'] and summary['memory'].get('global_load_accesses', 0) == 0 and
              summary['memory']['shared_load_accesses'] >= 64 and summary['memory']['global_atomic_accesses'] == 64)
        check('generic publication counters agree', re.findall(r'global\s+ld=(\d+)',
              (publication / 'stderr.log').read_text()) == ['0'])
        spaces = run('memory-spaces', ROOT / 'tests/conformance/memory_spaces.cu')
        data = rows(spaces)
        local = [r for r in data if r['type'] == 'memory' and r['space'] == 'local']
        identities = {}
        for r in local: identities.setdefault(r['address'], set()).add(r['lane'])
        check('private local ownership remains per lane', local and any(len(v) > 1 for v in identities.values()))
        summary = summarize_events(spaces / 'execution-events.jsonl')
        check('constant-bank loads remain distinct', summary['memory'].get('constant_load_accesses', 0) > 0 and summary['launches'] == 3)
        calls = run('device-calls', ROOT / 'tests/conformance/device_calls.cu')
        data = [r for r in rows(calls) if r['type'] == 'issue']
        functions = {}
        for r in data:
            functions.setdefault((r['launch'], tuple(r['block']), r['warp'], r['frame']), set()).add(r['function'])
        check('call-frame generations do not alias functions', all(len(v) == 1 for v in functions.values()) and
              any(r['frame'] > 0 for r in data))
        check('call ABI coverage explicitly incomplete', not summarize_events(calls / 'execution-events.jsonl')['complete_supported_memory'])
        gap = run('async-gap', ROOT / 'tests/conformance/async_event_gap.cu')
        summary = summarize_events(gap / 'execution-events.jsonl')
        check('async paths refuse complete capture claim', summary['coverage_gaps'] > 0 and not summary['complete_supported_memory'])
        fault = run('fault', ROOT / 'examples/out_of_bounds.cu', negative=True)
        summary = summarize_events(fault / 'execution-events.jsonl')
        check('faulting attempts excluded from completed traffic', summary['memory'].get('global_store_accesses') == 4 and
              summary['fault_boundaries'] >= 2 and not summary['complete_supported_memory'])
        # Rejected malformed traces must never silently become timing inputs.
        data = rows(captured)
        for suffix, mutate in [('parent', lambda r: r.update(issue=9999999)),
                               ('mask', lambda r: r.update(lane=63)),
                               ('address', lambda r: r.update(address=1234)),
                               ('width', lambda r: r.update(bytes=3))]:
            altered = [dict(r) for r in data]
            mutate(next(r for r in altered if r['type'] == 'memory'))
            path = out / ('invalid-' + suffix + '.jsonl')
            path.write_text(''.join(json.dumps(r) + '\n' for r in altered))
            try: summarize_events(path)
            except (ValueError, TypeError): checks.append('reader rejects invalid ' + suffix)
            else: raise AssertionError('Malformed trace accepted: ' + suffix)
        altered = [dict(r) for r in data]
        altered.remove(next(r for r in altered if r['type'] == 'memory'))
        renumber = {r['sequence']: i + 1 for i, r in enumerate(altered)}
        for r in altered:
            r['sequence'] = renumber[r['sequence']]
            if 'issue' in r: r['issue'] = renumber[r['issue']]
        missing = out / 'missing-memory.jsonl'
        missing.write_text(''.join(json.dumps(r) + '\n' for r in altered))
        try: summarize_events(missing)
        except ValueError as e: check('reader rejects missing memory despite renumbering', 'conservation' in str(e))
        else: raise AssertionError('Missing actual memory access accepted')
        unfinished = out / 'unfinished.jsonl'
        unfinished.write_text(''.join(json.dumps(r) + '\n' for r in data[:-1]))
        check('missing stream footer refuses complete coverage', not summarize_events(unfinished)['complete_supported_memory'])
        report['passed'] = True
    finally:
        report['total_checks'] = len(checks)
        (out / 'results.json').write_text(json.dumps(report, indent=2))
    print('PASS execution event conformance checks=' + str(len(checks)), flush=True)
    return 0
