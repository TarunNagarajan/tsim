"""Build and execute pinned upstream functional PTX conformance tests."""
import json
import os
from pathlib import Path
import re
import subprocess

TARGETS = ['test_ptx', 'test_exec', 'test_exec2', 'test_exec3', 'test_fastpath',
           'test_barriers', 'test_memory', 'test_regalloc']

def verify_core(cache, out):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.pop('TSIM_TRACE', None)
    (out / 'tmp').mkdir(exist_ok=True)
    env['TMPDIR'] = str(out / 'tmp')
    cmake = cache / 'toolchain/cmake-wheel/cmake/data/bin/cmake'
    with (out / 'build.log').open('w') as log:
        result = subprocess.run([str(cmake), '--build', str(cache / 'build'), '--target',
                                 *TARGETS, '-j', '2'], env=env, stdout=log, stderr=subprocess.STDOUT)
    cases = []
    report = {'engine_revision': '771bb8f96abc3721700219d88fa74e10d43dcfd5',
              'kind': 'upstream functional conformance; not hardware differential testing',
              'build_exit_code': result.returncode, 'suites': cases}
    if result.returncode == 0:
        for target in TARGETS:
            try:
                result = subprocess.run([str(cache / 'build' / target)], env=env, text=True,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=240)
                output, code = result.stdout, result.returncode
            except subprocess.TimeoutExpired as exc:
                output = exc.stdout or b''
                if isinstance(output, bytes): output = output.decode(errors='replace')
                output += '\nSuite exceeded 240 seconds\n'
                code = 124
            (out / (target + '.log')).write_text(output)
            matches = re.findall(r'(\d+)/(\d+) tests passed', output)
            passed, total = map(int, matches[-1]) if matches else (0, 0)
            failures = re.findall(r'\[ FAIL \] (.+)', output)
            cases.append({'suite': target, 'exit_code': code, 'passed': passed, 'total': total,
                          'failures': failures, 'success': code == 0 and total > 0 and passed == total})
            print(f'{target}: {passed}/{total} passed, exit={code}', flush=True)
    report['passed'] = report['build_exit_code'] == 0 and len(cases) == len(TARGETS) and all(case['success'] for case in cases)
    report['total_checks'] = sum(case['total'] for case in cases)
    (out / 'results.json').write_text(json.dumps(report, indent=2))
    return 0 if report['passed'] else 1

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    raise SystemExit(verify_core(args.cache.resolve(), args.out))
