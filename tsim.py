#!/usr/bin/env python3
"""Compile real CUDA C++ and execute/inspect it on a CPU."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))

def linux_path(value):
    value = str(value)
    if re.match(r'^[A-Za-z]:[\\/]', value):
        return '/mnt/' + value[0].lower() + '/' + value[3:].replace('\\', '/')
    return value

def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', help='Local compiler/build cache (remembered after setup)')
    p.add_argument('--distro', default='Ubuntu', help='Windows WSL distribution')
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('setup', help='Download pinned compiler and build CPU runtime')
    s.add_argument('--jobs', type=int, default=2)
    sub.add_parser('doctor', help='Check local toolchain and CPU simulator')
    sub.add_parser('devices', help='List simulated GPU profiles')
    s = sub.add_parser('run', help='Compile and execute a .cu program on the CPU')
    s.add_argument('source', type=Path)
    s.add_argument('--gpu', default='h100')
    s.add_argument('--trace', action='store_true', help='Record real warp issues and create an HTML inspector')
    s.add_argument('--trace-limit', type=int, default=200000)
    s.add_argument('--events', action='store_true', help='Capture typed execution and actual memory events')
    s.add_argument('--events-limit', type=int, default=200000, help='Maximum execution event records')
    s.add_argument('--scheduler', choices=['deterministic', 'random', 'adversarial'], default='deterministic')
    s.add_argument('--seed', type=int, default=1)
    s.add_argument('--threads', type=int, default=1, help='CPU workers; one gives stable trace ordering')
    s.add_argument('--race', action='store_true')
    s.add_argument('--strict', action='store_true')
    s.add_argument('--timeout', type=float, default=60)
    s.add_argument('--out', type=Path, help='Run artifact directory')
    s.add_argument('--nvcc-arg', action='append', default=[])
    s = sub.add_parser('inspect', help='Build an offline interactive inspector from a saved trace')
    s.add_argument('trace', type=Path)
    s.add_argument('--out', type=Path)
    s = sub.add_parser('test', help='Run independently checked CUDA integration tests')
    s.add_argument('--out', type=Path)
    s = sub.add_parser('conformance', help='Build and run the pinned core PTX conformance suites')
    s.add_argument('--out', type=Path)
    s = sub.add_parser('source-conformance', help='Run CUDA feature and synchronization references')
    s.add_argument('--out', type=Path)
    s = sub.add_parser('event-conformance', help='Verify actual memory events and capture coverage')
    s.add_argument('--out', type=Path)
    s = sub.add_parser('memory-report', help='Analyze complete captured address transactions')
    s.add_argument('run_directory', type=Path)
    s.add_argument('--out', type=Path)
    s = sub.add_parser('memory-conformance', help='Verify transaction and sector-cache model components')
    s.add_argument('--out', type=Path)
    s = sub.add_parser('cache-report', help='Replay a configurable sector-cache hierarchy from complete events')
    s.add_argument('run_directory', type=Path)
    s.add_argument('--profile', type=Path)
    s.add_argument('--out', type=Path)
    s = sub.add_parser('cache-conformance', help='Verify cache policies and two-level traffic')
    s.add_argument('--out', type=Path)
    return p

def run_program(args, cache, program_args=(), compiled_from=None):
    from inspector import create_report
    source = args.source.resolve()
    if not source.is_file(): raise RuntimeError('CUDA source does not exist: ' + str(source))
    if source.suffix != '.cu': raise RuntimeError('Expected a CUDA C++ .cu source file')
    for name in ['threads', 'trace_limit']:
        if getattr(args, name) < 1: raise RuntimeError(name + ' must be positive')
    if getattr(args, 'events_limit', 200000) < 1: raise RuntimeError('events_limit must be positive')
    if args.timeout <= 0: raise RuntimeError('timeout must be positive')
    toolkit = cache / 'toolchain' / 'cuda'
    nvcc = toolkit / 'bin' / 'nvcc'
    simulator = cache / 'build' / 'vgpu'
    if not nvcc.exists() or not simulator.exists(): raise RuntimeError('Run tsim.py setup first')
    tag = datetime.datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    out = (args.out or cache / 'runs' / (source.stem + '-' + tag)).resolve()
    out.mkdir(parents=True, exist_ok=True)
    binary = out / 'program'
    ptx = out / 'kernel.ptx'
    env = os.environ.copy()
    env['PATH'] = str(toolkit / 'bin') + ':' + env.get('PATH', '')
    # Linux /tmp may live on a different/full disk; keep compiler scratch with artifacts.
    (out / 'tmp').mkdir(exist_ok=True)
    env['TMPDIR'] = str(out / 'tmp')
    # compute_80 PTX is interpreted; no NVIDIA driver or SASS execution is involved.
    host_compiler = shutil.which('g++-13') or shutil.which('g++')
    if not host_compiler: raise RuntimeError('A CUDA-compatible GNU host compiler is required (tested: GCC 13)')
    common = [str(nvcc), '-ccbin', host_compiler, '-std=c++17', '-O2', '-lineinfo', '-L' + str(toolkit / 'lib'),
              '-gencode', 'arch=compute_80,code=compute_80']
    common += args.nvcc_arg
    commands = [common + ['-cudart', 'shared', str(source), '-o', str(binary)],
                common + ['--ptx', str(source), '-o', str(ptx)]]
    if compiled_from is not None:
        # An explicit replay of one compilation, used by the schedule/profile
        # matrix. Verify the saved inputs and artifacts; this is not a build cache.
        previous = Path(compiled_from).resolve()
        record = json.loads((previous / 'run.json').read_text())
        if ((previous / 'source.cu').read_bytes() != source.read_bytes() or
                record['nvcc_args'] != args.nvcc_arg):
            raise RuntimeError('Compilation replay inputs differ')
        for name in ['program', 'kernel.ptx']:
            saved = previous / name
            if hashlib.sha256(saved.read_bytes()).hexdigest() != record['compilation_sha256'][name]:
                raise RuntimeError('Compilation replay artifact changed: ' + str(saved))
            if saved.resolve() != (out / name).resolve():
                shutil.copy2(saved, out / name)
        commands = record['compile_commands']
        if previous != out:
            shutil.copy2(previous / 'compile.log', out / 'compile.log')
    else:
        compile_log = []
        for command in commands:
            result = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, timeout=180)
            compile_log.append(result.stdout)
            if result.returncode:
                (out / 'compile.log').write_text(''.join(compile_log))
                print(result.stdout, end='')
                raise RuntimeError('CUDA compilation failed; see ' + str(out / 'compile.log'))
        (out / 'compile.log').write_text(''.join(compile_log))
    env.update(VGPU_SASS='0', VGPU_THREADS=str(args.threads), VGPU_SCHEDULER=args.scheduler,
               VGPU_SCHEDULER_SEED=str(args.seed), VGPU_MAX_STEPS='10000000')
    trace = out / 'warp-trace.jsonl'
    if args.trace:
        trace.write_text('')
        env['TSIM_TRACE'] = str(trace)
        env['TSIM_TRACE_LIMIT'] = str(args.trace_limit)
    else:
        env.pop('TSIM_TRACE', None)
        env.pop('TSIM_TRACE_LIMIT', None)
    events = out / 'execution-events.jsonl'
    if getattr(args, 'events', False):
        events.write_text('')
        env['TSIM_EVENTS'] = str(events)
        env['TSIM_EVENTS_LIMIT'] = str(getattr(args, 'events_limit', 200000))
    else:
        env.pop('TSIM_EVENTS', None)
        env.pop('TSIM_EVENTS_LIMIT', None)
    gpu = args.gpu if '/' in args.gpu else 'nvidia/' + args.gpu
    command = [str(simulator), 'run', '--gpu', gpu, '--counters']
    if args.race: command.append('--race')
    if args.strict: command.append('--strict')
    command += [str(binary)] + list(program_args)
    timed_out = False
    try:
        result = subprocess.run(command, env=env, text=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, timeout=args.timeout)
        stdout, stderr, code = result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or b'').decode() if isinstance(exc.stdout, bytes) else (exc.stdout or '')
        stderr = (exc.stderr or b'').decode() if isinstance(exc.stderr, bytes) else (exc.stderr or '')
        stderr += '\ntsim: execution exceeded timeout\n'
        code, timed_out = 124, True
    (out / 'stdout.log').write_text(stdout)
    (out / 'stderr.log').write_text(stderr)
    # Preserve the source used in this run, even after the original is edited.
    (out / 'source.cu').write_bytes(source.read_bytes())
    metadata = {'source': str(source), 'gpu': gpu, 'backend': 'CPU PTX interpreter',
                'scheduler': args.scheduler, 'seed': args.seed, 'threads': args.threads,
                'exit_code': code, 'timed_out': timed_out, 'compile_commands': commands,
                'nvcc_args': args.nvcc_arg, 'compiled_from': str(compiled_from) if compiled_from else None,
                'compilation_sha256': {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                                       for name in ['program', 'kernel.ptx']},
                'run_command': command, 'trace_limit': args.trace_limit,
                'events_limit': getattr(args, 'events_limit', 200000),
                'execution_environment': {key: env[key] for key in ['VGPU_SASS', 'VGPU_THREADS',
                    'VGPU_SCHEDULER', 'VGPU_SCHEDULER_SEED', 'VGPU_MAX_STEPS', 'TSIM_TRACE',
                    'TSIM_TRACE_LIMIT', 'TSIM_EVENTS', 'TSIM_EVENTS_LIMIT'] if key in env},
                'engine_revision': '771bb8f96abc3721700219d88fa74e10d43dcfd5'}
    (out / 'run.json').write_text(json.dumps(metadata, indent=2))
    if args.trace and trace.exists(): create_report(trace, out / 'inspector.html')
    if getattr(args, 'events', False) and events.exists():
        from execution_events import summarize_events
        summary = summarize_events(events)
        summary['process_exit_code'] = code
        summary['timed_out'] = timed_out
        summary['complete_supported_memory'] &= code == 0 and not timed_out
        (out / 'execution-summary.json').write_text(json.dumps(summary, indent=2))
    print(stdout, end='')
    print(stderr, end='', file=sys.stderr)
    print('Run artifacts: ' + str(out), flush=True)
    return code, out

def main():
    argv = sys.argv[1:]
    program_args = []
    if '--' in argv:
        split = argv.index('--')
        argv, program_args = argv[:split], argv[split + 1:]
    p = parser()
    args = p.parse_args(argv)
    config = ROOT / '.local.json'
    saved = json.loads(config.read_text()) if config.exists() else {}
    cache = args.cache or saved.get('cache') or str(ROOT / '.local')
    if os.name == 'nt':
        converted = ['--cache', linux_path(cache), args.command]
        if args.command == 'setup': converted += ['--jobs', str(args.jobs)]
        if args.command == 'run':
            converted += [linux_path(args.source.resolve()), '--gpu', args.gpu,
                          '--trace-limit', str(args.trace_limit), '--scheduler', args.scheduler,
                          '--seed', str(args.seed), '--threads', str(args.threads), '--timeout', str(args.timeout)]
            converted += ['--events-limit', str(args.events_limit)]
            for flag in ['trace', 'events', 'race', 'strict']:
                if getattr(args, flag): converted.append('--' + flag)
            for value in args.nvcc_arg: converted.append('--nvcc-arg=' + value)
        if args.command == 'inspect': converted.append(linux_path(args.trace.resolve()))
        if args.command in ['memory-report', 'cache-report']: converted.append(linux_path(args.run_directory.resolve()))
        if getattr(args, 'profile', None): converted += ['--profile', linux_path(args.profile.resolve())]
        if getattr(args, 'out', None): converted += ['--out', linux_path(args.out.resolve())]
        if program_args: converted += ['--'] + [linux_path(x) for x in program_args]
        distro = args.distro if '--distro' in argv else saved.get('distro', args.distro)
        if args.command == 'setup':
            config.write_text(json.dumps({'cache': str(cache), 'distro': distro}, indent=2))
        command = ['wsl', '-d', distro, '--cd', linux_path(Path.cwd()), '--', 'python3', linux_path(__file__)] + converted
        return subprocess.call(command)
    cache = Path(linux_path(cache)).resolve()
    if args.command == 'setup':
        subprocess.run([sys.executable, str(ROOT / 'scripts' / 'setup.py'), '--cache', str(cache),
                        '--jobs', str(args.jobs)], check=True)
        config.write_text(json.dumps({'cache': str(cache), 'distro': args.distro}, indent=2))
        return 0
    if args.command == 'run': return run_program(args, cache, program_args)[0]
    if args.command == 'inspect':
        from inspector import create_report
        target = args.out or args.trace.with_suffix('.html')
        create_report(args.trace, target)
        print('Inspector: ' + str(target.resolve()))
        return 0
    if args.command == 'test':
        from verify import verify
        return verify(cache, args.out)
    if args.command == 'conformance':
        from core_conformance import verify_core
        return verify_core(cache, args.out or cache / 'core-conformance')
    if args.command == 'source-conformance':
        from source_conformance import verify_source
        return verify_source(cache, args.out or cache / 'source-conformance')
    if args.command == 'event-conformance':
        from event_conformance import verify_events
        return verify_events(cache, args.out or cache / 'event-conformance')
    if args.command == 'memory-report':
        from memory_model import analyze_transactions
        report = analyze_transactions(args.run_directory)
        target = args.out or args.run_directory / 'memory-transactions.json'
        target.write_text(json.dumps(report, indent=2))
        print('Transaction report: ' + str(target.resolve()))
        return 0
    if args.command == 'memory-conformance':
        from memory_conformance import verify_memory
        return verify_memory(cache, args.out or cache / 'memory-conformance')
    if args.command == 'cache-report':
        from cache_hierarchy import analyze_cache
        profile = json.loads(args.profile.read_text()) if args.profile else None
        report = analyze_cache(args.run_directory, profile)
        target = args.out or args.run_directory / 'cache-hierarchy.json'
        target.write_text(json.dumps(report, indent=2))
        print('Cache hierarchy report: ' + str(target.resolve()))
        return 0
    if args.command == 'cache-conformance':
        from cache_conformance import verify_cache
        return verify_cache(cache, args.out or cache / 'cache-conformance')
    if args.command == 'devices':
        return subprocess.call([str(cache / 'build' / 'vgpu'), 'list-gpus'])
    if args.command == 'doctor':
        checks = [(['g++', '--version'], 'Host compiler'),
                  ([str(cache / 'toolchain/cuda/bin/nvcc'), '--version'], 'CUDA compiler'),
                  ([str(cache / 'build/vgpu'), 'info', '--gpu', 'nvidia/h100'], 'CPU simulator')]
        success = True
        for command, label in checks:
            print('\n' + label + ':', flush=True)
            try: success &= subprocess.call(command) == 0
            except FileNotFoundError:
                print('Missing: ' + command[0]); success = False
        return 0 if success else 1

if __name__ == '__main__':
    try: sys.exit(main())
    except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print('tsim: ' + str(exc), file=sys.stderr)
        sys.exit(1)
