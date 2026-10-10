"""Numerical references, scheduling invariants, and intentional fault checks."""
import argparse
import json
from pathlib import Path
import re
import tempfile
from inspector import read_trace, create_report

def verify(cache, out=None):
    from tsim import ROOT, run_program
    out = Path(out or cache / 'verification').resolve()
    out.mkdir(parents=True, exist_ok=True)
    results = []
    def case(name, scheduler='deterministic', trace=True, limit=200000, negative=None, threads=1):
        args = argparse.Namespace(source=ROOT / 'examples' / (name + '.cu'), gpu='h100',
            trace=trace, trace_limit=limit, scheduler=scheduler, seed=7, threads=threads,
            race=True, strict=False, timeout=60, out=out / (name + '-' + scheduler + ('-trace' if trace else '-plain') + ('-cap' if limit < 100 else '') + ('-workers' + str(threads) if threads != 1 else '')),
            nvcc_arg=[])
        code, path = run_program(args, cache)
        stdout = (path / 'stdout.log').read_text()
        stderr = (path / 'stderr.log').read_text()
        if negative:
            assert code != 0, name + ': expected a rejected kernel'
            assert re.search(negative, stderr, re.I), name + ': missing diagnostic'
        else:
            assert code == 0 and 'PASS ' + name in stdout, name + ': numerical result failed'
        events = []
        if trace:
            events, truncated = read_trace(path / 'warp-trace.jsonl')
            assert events, name + ': trace is empty'
            for i, event in enumerate(events):
                assert event['sequence'] == i, 'Trace sequence is not contiguous'
                assert 0 <= event['execution_mask'] <= 0xffffffff
                assert event['execution_mask'] & ~event['path_mask'] == 0
                assert event['paths'] >= 1 and len(event['block']) == 3
            ptx_lines = (path / 'kernel.ptx').read_text().splitlines()
            normalize = lambda value: re.sub(r'\s+', '', value).rstrip(';')
            for event in events:
                assert 1 <= event['ptx_line'] <= len(ptx_lines), name + ': PTX line outside artifact'
                assert normalize(event['instruction']) == normalize(ptx_lines[event['ptx_line'] - 1]), name + ': PTX trace line misaligned'
            assert truncated == (limit < 100), name + ': unexpected trace truncation'
            html = (path / 'inspector.html').read_text()
            assert '__TSIM_DATA__' not in html and 'traceData' in html
        results.append({'case': name, 'scheduler': scheduler, 'trace': trace, 'exit_code': code,
                        'events': len(events), 'artifacts': str(path), 'passed': True})
        return path, events, stdout
    try:
        path, vector_events, output = case('vector_add')
        assert 'n=259 last=' + str(3 * 258 + 1) in output
        assert any(0 < e['execution_mask'] < 0xffffffff for e in vector_events), 'Bounds predicate absent'
        _, _, plain = case('vector_add', trace=False)
        assert output == plain, 'Tracing changed program results'
        first = (path / 'warp-trace.jsonl').read_bytes()
        again, _, _ = case('vector_add')
        assert first == (again / 'warp-trace.jsonl').read_bytes(), 'Deterministic trace did not replay'
        _, parallel, parallel_output = case('vector_add', threads=4)
        assert parallel_output == output and len(parallel) == len(vector_events), 'Parallel CPU blocks changed results or dropped trace events'
        _, diverged, output = case('divergence')
        assert 'even=14 odd=15' in output
        assert any(e['paths'] > 1 for e in diverged), 'No actual divergence recorded'
        _, shared, output = case('shared_reduce')
        assert 'total=' + str(sum(range(1, 129))) in output
        assert any('bar.sync' in e['instruction'] for e in shared)
        assert len({e['warp'] for e in shared}) == 4
        case('shared_reduce', scheduler='adversarial')
        case('shared_reduce', scheduler='random')
        _, warp, output = case('warp_atomic')
        assert 'total=' + str(4 * sum(range(1, 33))) in output and 'ballot=ffff' in output
        assert any('shfl.sync' in e['instruction'] for e in warp)
        assert any(re.search(r'(atom|red)\.', e['instruction']) for e in warp)
        case('warp_atomic', scheduler='adversarial')
        case('index_3d')
        _, multiple, output = case('multiple_launches')
        assert 'total=2' in output and len({e['launch'] for e in multiple}) == 2
        _, _, output = case('tiled_matmul', scheduler='adversarial')
        n = 19
        reference = lambda row, col: sum(((row * n + k) % 7 - 3) * ((k * n + col) % 5 - 2) for k in range(n))
        assert f'first={reference(0, 0)} last={reference(n - 1, n - 1)}' in output
        case('out_of_bounds', negative=r'out.of.bounds')
        case('shared_race', scheduler='adversarial', negative=r'race')
        case('vector_add', limit=5)
        # Source containing script markup remains inert JSON data in the offline viewer.
        with tempfile.TemporaryDirectory(dir=out) as temp:
            folder = Path(temp)
            (folder / 'source.cu').write_text('// </script><script>throw Error("unsafe")</script>')
            (folder / 'trace.jsonl').write_text(json.dumps(vector_events[0]) + '\n')
            create_report(folder / 'trace.jsonl', folder / 'report.html')
            html = (folder / 'report.html').read_text()
            assert '// </script>' not in html and '\\u003c/script\\u003e' in html
        (out / 'results.json').write_text(json.dumps({'passed': True, 'cases': results}, indent=2))
        print(f'PASS: {len(results)} CUDA execution checks, trace invariants, deterministic replay, and inspector data escaping')
        return 0
    except (AssertionError, RuntimeError) as exc:
        (out / 'results.json').write_text(json.dumps({'passed': False, 'cases': results, 'failure': str(exc)}, indent=2))
        print('FAIL: ' + str(exc))
        return 1
