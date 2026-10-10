"""CUDA source references and synchronization faults across profiles/schedules."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

def verify_source(cache, out, replay_from=None):
    from tsim import ROOT, run_program
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    compiler = 'clang++'
    binary = out / 'shared-race-model'
    subprocess.run([compiler, '-std=c++20', '-O2', '-I' + str(ROOT / 'include'),
                    str(ROOT / 'tests/shared_race_model.cpp'), '-o', str(binary)], check=True)
    result = subprocess.run([str(binary)], text=True, capture_output=True, check=True)
    (out / 'shared-race-model.log').write_text(result.stdout + result.stderr)
    cases = []
    report = {'kind': 'functional CPU CUDA conformance; not hardware differential testing',
              'engine_revision': '771bb8f96abc3721700219d88fa74e10d43dcfd5',
              'compiler_ptx_target': 'compute_80',
              'patch_sha256': {p.name: hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                               for p in sorted((ROOT / 'patches').glob('*.patch'))},
              'shared_race_model': 'passed', 'cases': cases, 'passed': False}
    probes = ['warp_collectives', 'device_calls', 'numeric_edges', 'memory_spaces',
              'atomic_protocol', 'transitive_publication', 'counted_barrier', 'rmw_publication', 'acquire_pattern']
    # The negative variants preserve the communication protocol and remove
    # exactly one ordering edge. Strict checking also catches equal-value stores.
    variants = [(name, name, [], False) for name in probes] + [
        ('missing_release', 'atomic_protocol', ['-DPUBLISH_ORDER=cuda::memory_order_relaxed'], True),
        ('missing_acquire', 'atomic_protocol', ['-DOBSERVE_ORDER=cuda::memory_order_relaxed'], True),
        ('post_arrival_write', 'counted_barrier', ['-DPOST_ARRIVAL=1'], True),
        ('plain_relay_store', 'rmw_publication', ['-DRELAY_STORE=1'], True)]
    manifest = json.loads((ROOT / 'docs/compatibility.json').read_text())
    covered = {name for feature in manifest['features'] for name in feature['cases']}
    if covered != {variant[0] for variant in variants}:
        raise RuntimeError('Compatibility inventory and executed cases differ')
    compiled = {}
    if replay_from:
        for name, source, options, _ in variants:
            prior = Path(replay_from) / (name + '-a100-deterministic')
            if (prior / 'run.json').is_file():
                record = json.loads((prior / 'run.json').read_text())
                if (record.get('nvcc_args') == options and (prior / 'source.cu').read_bytes() ==
                        (ROOT / 'tests/conformance' / (source + '.cu')).read_bytes()):
                    compiled[name] = prior
    try:
        for gpu in manifest['profile_matrix']:
            for scheduler in manifest['scheduler_matrix']:
                for name, source, options, negative in variants:
                    folder = out / (name + '-' + gpu + '-' + scheduler)
                    args = argparse.Namespace(source=ROOT / 'tests/conformance' / (source + '.cu'),
                        gpu=gpu, scheduler=scheduler, seed=17, threads=1, race=True, strict=True,
                        trace=False, trace_limit=200000, timeout=45, out=folder, nvcc_arg=options)
                    code, path = run_program(args, cache, compiled_from=compiled.get(name))
                    compiled.setdefault(name, path)
                    stdout = (path / 'stdout.log').read_text()
                    stderr = (path / 'stderr.log').read_text()
                    metadata = json.loads((path / 'run.json').read_text())
                    if source == 'counted_barrier':
                        ptx = (path / 'kernel.ptx').read_text()
                        if not all(op in ptx for op in ['st.shared.u32', 'ld.shared.u32', 'bar.arrive', 'bar.sync']):
                            raise RuntimeError('Counted barrier PTX lost a required memory operation')
                    if source in ['rmw_publication', 'acquire_pattern']:
                        ptx = (path / 'kernel.ptx').read_text()
                        required = ['ld.acquire.cta.b32', 'ld.relaxed.cta.b32', 'st.release.cta.b32', 'ld.shared.u32']
                        if name == 'rmw_publication': required.append('atom.add.relaxed.cta.u32')
                        else: required.append('st.relaxed.cta.b32')
                        if not all(op in ptx for op in required):
                            raise RuntimeError('Publication PTX lost a required memory operation')
                    success = (code != 0 and 'race' in stderr.lower() and not metadata['timed_out']) if negative else (
                        code == 0 and 'PASS ' + source in stdout)
                    cases.append({'case': name, 'source': source, 'profile': gpu, 'scheduler': scheduler,
                                  'seed': 17, 'expected_race': negative, 'exit_code': code, 'passed': success})
                    print(f'{name} {gpu} {scheduler}: ' + ('PASS' if success else 'FAIL'), flush=True)
                    if not success: raise RuntimeError('Source conformance failed: ' + str(folder))
        report['passed'] = True
    finally:
        report['total_checks'] = len(cases)
        (out / 'results.json').write_text(json.dumps(report, indent=2))
    return 0
