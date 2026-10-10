"""Compile verified PTX and cross-check NVIDIA compiler/cubin resource tools."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

from memory_model import validate_compilation_artifacts

BINARY_UTILITIES = 'https://docs.nvidia.com/cuda/archive/12.6.3/cuda-binary-utilities/index.html'

def parse_ptxas_resources(text, architecture):
    functions, entries = {}, set()
    current = None
    for line in text.splitlines():
        compile_entry = re.fullmatch(r"ptxas info\s*: Compiling entry function '([^']+)' for '(sm_\d+)'", line.strip())
        if compile_entry:
            name, arch = compile_entry.groups()
            if arch != architecture or name in entries: raise ValueError('Unexpected or duplicate compiler entry architecture')
            entries.add(name)
            functions.setdefault(name, {})
            current = None
            continue
        properties = re.fullmatch(r'ptxas info\s*: Function properties for (\S+)', line.strip())
        if properties:
            current = properties[1]
            functions.setdefault(current, {})
            continue
        stack = re.fullmatch(r'\s*(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads\s*', line)
        if stack:
            if current is None or 'stack_bytes' in functions[current]: raise ValueError('Unparented or duplicate compiler stack metadata')
            functions[current].update(dict(zip(['stack_bytes', 'spill_store_bytes', 'spill_load_bytes'], map(int, stack.groups()))))
            continue
        used = re.fullmatch(r'ptxas info\s*: Used (\d+) registers(?:, (.*))?', line.strip())
        if used:
            if current is None or 'registers' in functions[current]: raise ValueError('Unparented or duplicate compiler resource metadata')
            record = functions[current]
            record.update(registers=int(used[1]), shared_prefix_bytes=0, local_bytes=0, constant_banks={})
            seen_sizes = set()
            for token in (used[2] or '').split(','):
                token = token.strip()
                if not token: continue
                barriers = re.fullmatch(r'used (\d+) barriers', token)
                size = re.fullmatch(r'(\d+) bytes (smem|lmem|cmem\[(\d+)\])', token)
                cumulative = re.fullmatch(r'(\d+) bytes cumulative stack size', token)
                if barriers:
                    if 'barriers' in record: raise ValueError('Duplicate compiler barrier count')
                    record['barriers'] = int(barriers[1])
                elif size:
                    count, kind, bank = size.groups()
                    if kind in seen_sizes: raise ValueError('Duplicate compiler allocation size')
                    seen_sizes.add(kind)
                    if bank is not None: record['constant_banks'][bank] = int(count)
                    else: record['shared_prefix_bytes' if kind == 'smem' else 'local_bytes'] = int(count)
                elif cumulative:
                    if 'cumulative_stack_bytes' in record: raise ValueError('Duplicate cumulative stack count')
                    record['cumulative_stack_bytes'] = int(cumulative[1])
                else: raise ValueError('Unknown compiler resource field: ' + token)
    required = {'registers', 'barriers', 'stack_bytes', 'spill_store_bytes', 'spill_load_bytes', 'shared_prefix_bytes'}
    if not entries or any(not required <= functions[name].keys() for name in entries):
        raise ValueError('Missing complete compiler entry resource metadata')
    return {'entries': {name: functions[name] for name in sorted(entries)},
            'device_functions': {name: value for name, value in functions.items() if name not in entries}}

def parse_cuobjdump_resources(text):
    functions, current = {}, None
    for line in text.splitlines():
        function = re.fullmatch(r'\s*Function (\S+):\s*', line)
        if function:
            current = function[1]
            if current in functions: raise ValueError('Duplicate cubin resource function')
            functions[current] = {}
        elif current is not None and line.strip():
            tokens = line.split()
            for token in tokens:
                field = re.fullmatch(r'([A-Z]+(?:\[\d+\])?):(\d+)', token)
                if not field or field[1] in functions[current]: raise ValueError('Malformed or duplicate cubin resource field')
                functions[current][field[1]] = int(field[2])
    required = {'REG', 'STACK', 'SHARED', 'LOCAL'}
    if not functions or any(not required <= value.keys() for value in functions.values()):
        raise ValueError('Missing cubin resource metadata')
    return functions

def crosscheck_resources(compiler, cubin):
    for name, compiled in compiler['entries'].items():
        if name not in cubin: raise ValueError('Compiled entry missing from cubin: ' + name)
        actual = cubin[name]
        if actual['REG'] != compiled['registers'] or actual['STACK'] != compiled.get('cumulative_stack_bytes', compiled['stack_bytes']):
            raise ValueError('Compiler and cubin register/stack counts disagree: ' + name)
        for bank, count in compiled['constant_banks'].items():
            if actual.get('CONSTANT[' + bank + ']') != count:
                raise ValueError('Compiler and cubin constant bank counts disagree: ' + name)
        # CUDA12.6 cubins with a shared section may include the 1-KiB system
        # reservation. Retain both producer values; never charge that twice.
        delta = actual['SHARED'] - compiled['shared_prefix_bytes']
        if delta not in [0, 1024]: raise ValueError('Unverified shared compiler/cubin encoding: ' + name)
        compiled['cubin_raw_shared_bytes'] = actual['SHARED']
        compiled['encoded_system_reservation_bytes'] = delta
        compiled['cubin_local_bytes'] = actual['LOCAL']
    return compiler

def compile_resources(run_directory, cache, out=None):
    run = Path(run_directory).resolve()
    metadata = validate_compilation_artifacts(run)
    out = Path(out or run / 'compiler-resources').resolve(); out.mkdir(parents=True, exist_ok=True)
    if out == run: raise ValueError('Compiler artifacts require a separate output directory')
    toolkit = Path(cache) / 'toolchain/cuda/bin'
    ptxas, cuobjdump = toolkit / 'ptxas', toolkit / 'cuobjdump'
    if not ptxas.is_file() or not cuobjdump.is_file(): raise ValueError('Run setup to install pinned compiler resource tools')
    tools = {}
    for tool, version in [(ptxas, 'V12.6.85'), (cuobjdump, 'V12.6.77')]:
        result = subprocess.run([str(tool), '--version'], text=True, capture_output=True, timeout=30, check=True)
        output = result.stdout + result.stderr
        if version not in output: raise ValueError('Unexpected compiler resource tool version: ' + tool.name)
        tools[tool.name] = {'release': version, 'version': output.strip(), 'sha256': hashlib.sha256(tool.read_bytes()).hexdigest()}
    architecture = 'sm_80' if metadata['gpu'] == 'nvidia/a100' else 'sm_90'
    cubin = out / 'module.cubin'
    # Remove only the owned prior result marker so a failed rebuild cannot
    # leave an apparently successful report for the new attempted inputs.
    (out / 'resources.json').unlink(missing_ok=True)
    snapshot = out / 'input.ptx'
    snapshot.write_bytes((run / 'kernel.ptx').read_bytes())
    if hashlib.sha256(snapshot.read_bytes()).hexdigest() != metadata['compilation_sha256']['kernel.ptx']:
        raise ValueError('PTX changed while taking compiler input snapshot')
    command = [str(ptxas), '-v', '-arch=' + architecture, str(snapshot), '-o', str(cubin)]
    result = subprocess.run(command, text=True, capture_output=True, timeout=180)
    log = result.stdout + result.stderr
    (out / 'ptxas.log').write_text(log)
    if result.returncode: raise ValueError('ptxas resource compilation failed; see ' + str(out / 'ptxas.log'))
    result = subprocess.run([str(cuobjdump), '--dump-resource-usage', str(cubin)], text=True, capture_output=True, timeout=30, check=True)
    dump = result.stdout
    (out / 'cuobjdump.log').write_text(dump)
    (out / 'cuobjdump-stderr.log').write_text(result.stderr)
    resources = crosscheck_resources(parse_ptxas_resources(log, architecture), parse_cuobjdump_resources(dump))
    # Refuse a concurrently changed source rather than associating output with
    # a different PTX hash than the one passed to the compiler.
    if validate_compilation_artifacts(run) != metadata:
        raise ValueError('Run identity changed during resource compilation')
    report = {'schema': 1, 'kind': 'NVIDIA compiler-reported allocations; not hardware measurements',
              'architecture': architecture, 'gpu': metadata['gpu'], 'ptx_sha256': metadata['compilation_sha256']['kernel.ptx'],
              'tools': tools, 'command': command, 'resources': resources,
              'artifact_sha256': {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                                  for name in ['input.ptx', 'module.cubin', 'ptxas.log', 'cuobjdump.log']},
              'functional_run_exit_code': metadata['exit_code'], 'hardware_validated': False,
              'source': BINARY_UTILITIES + '#resource-usage-information',
              'limits': ['compiler output is not measured residency or throughput',
                         'SASS lowering can remove or add traffic absent from the PTX event stream',
                         'spill/stack byte counts do not establish dynamic spill addresses or transactions',
                         'barrier count comes from ptxas; cubin raw shared bytes remain separately recorded']}
    (out / 'resources.json').write_text(json.dumps(report, indent=2))
    return report

def load_resources(directory, run_directory):
    try: return _load_resources(directory, run_directory)
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError('Malformed compiler resource metadata') from error

def _load_resources(directory, run_directory):
    directory = Path(directory).resolve()
    report = json.loads((directory / 'resources.json').read_text())
    metadata = validate_compilation_artifacts(run_directory)
    architecture = 'sm_80' if metadata['gpu'] == 'nvidia/a100' else 'sm_90'
    if (type(report['schema']) is not int or report['schema'] != 1 or report['ptx_sha256'] != metadata['compilation_sha256']['kernel.ptx'] or
            report['architecture'] != architecture or report['gpu'] != metadata['gpu'] or
            report['hardware_validated'] is not False or report['functional_run_exit_code'] != metadata['exit_code']):
        raise ValueError('Compiler resource input identity differs from run')
    command = report['command']
    if (not isinstance(command, list) or len(command) != 6 or not all(isinstance(item, str) for item in command) or
            Path(command[0]).name != 'ptxas' or command[1:3] != ['-v', '-arch=' + architecture] or
            Path(command[3]).name != 'input.ptx' or command[4] != '-o' or Path(command[5]).name != 'module.cubin'):
        raise ValueError('Unexpected compiler command provenance')
    if set(report['tools']) != {'ptxas', 'cuobjdump'}: raise ValueError('Incomplete compiler tool provenance')
    for name, release in [('ptxas', 'V12.6.85'), ('cuobjdump', 'V12.6.77')]:
        tool = report['tools'][name]
        if tool['release'] != release or release not in tool['version'] or not re.fullmatch('[0-9a-f]{64}', tool['sha256']):
            raise ValueError('Unexpected compiler tool provenance')
    if set(report['artifact_sha256']) != {'input.ptx', 'module.cubin', 'ptxas.log', 'cuobjdump.log'}: raise ValueError('Incomplete compiler resource provenance')
    if report['artifact_sha256']['input.ptx'] != report['ptx_sha256']: raise ValueError('Compiler input snapshot differs from run')
    for name, digest in report['artifact_sha256'].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != digest: raise ValueError('Compiler resource artifact changed: ' + name)
    parsed = crosscheck_resources(parse_ptxas_resources((directory / 'ptxas.log').read_text(), architecture),
                                  parse_cuobjdump_resources((directory / 'cuobjdump.log').read_text()))
    if parsed != report['resources']: raise ValueError('Resource report differs from compiler/cubin logs')
    return report
