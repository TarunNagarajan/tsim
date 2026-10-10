"""CPU-only differential against NVIDIA's installed occupancy SDK calculator."""
import argparse
import hashlib
from itertools import product
import json
from pathlib import Path
import subprocess
import sys

def verify_residency(cache, out, replay_from=None):
    from tsim import ROOT, run_program
    from sm_residency import profile_for, block_footprint, resource_ceiling, SMResidency, analyze_residency
    sys.path.insert(0, str(ROOT / 'tests'))
    from sm_residency_components import verify_residency_components
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=True)
    checks = verify_residency_components()
    header = Path(cache) / 'toolchain/cuda/include/cuda_occupancy.h'
    report = {'kind': 'resource model and NVIDIA standalone SDK calculator differential; no GPU hardware differential',
              'passed': False, 'checks': checks, 'hardware_cycles': False,
              'sdk_header_sha256': hashlib.sha256(header.read_bytes()).hexdigest(),
              'model_sha256': hashlib.sha256((ROOT / 'scripts/sm_residency.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest()}
    def check(name, condition):
        if not condition: raise AssertionError(name)
        checks.append(name)
    compiled = {}
    if replay_from:
        for record in sorted(Path(replay_from).glob('*/run.json')):
            metadata = json.loads(record.read_text()); source = Path(metadata['source'])
            if source.is_file() and (record.parent / 'source.cu').read_bytes() == source.read_bytes() and not metadata['nvcc_args']:
                compiled.setdefault(str(source), record.parent)
    try:
        binary = out / 'occupancy-sdk-reference'
        subprocess.run(['clang++', '-std=c++17', '-O2', '-I' + str(header.parent),
                        str(ROOT / 'tests/occupancy_sdk_reference.cpp'), '-o', str(binary)], check=True)
        cases = list(product([8, 9], [1, 31, 32, 33, 64, 96, 128, 192, 256, 512, 1024],
                             [0, 8, 17, 24, 56, 72, 88, 128, 200, 255, 256], [(0, 0), (12000, 9000), (0, 20001), (0, 60 * 1024)], [0, 1, 4, 16]))
        payload = ''.join(' '.join(map(str, [major, threads, registers, static, dynamic, barriers])) + '\n'
                          for major, threads, registers, (static, dynamic), barriers in cases)
        result = subprocess.run([str(binary)], input=payload, text=True, capture_output=True, check=True)
        lines = result.stdout.splitlines()
        check('SDK reference returns every requested case', len(lines) == len(cases))
        references = []
        for case, line in zip(cases, lines):
            major, threads, registers, (static, dynamic), barriers = case
            profile = profile_for('nvidia/' + ('a100' if major == 8 else 'h100'))
            footprint = block_footprint(profile, [threads, 1, 1], registers, static, dynamic, barriers)
            ceiling = resource_ceiling(profile, footprint)
            expected = list(map(int, line.split()))
            actual = [ceiling['blocks_per_sm'], footprint.registers, footprint.shared_bytes]
            if actual != expected: raise AssertionError('SDK mismatch ' + str(case) + ': ' + str(actual) + ' vs ' + str(expected))
            # Homogeneous allocator filling is independently checked against
            # the SDK's residency result, including uneven warp counts.
            sm = SMResidency(profile)
            for identity in range(expected[0]):
                if not sm.admit(identity, footprint): raise AssertionError('Allocator below SDK capacity: ' + str(case))
            if sm.admit(expected[0], footprint): raise AssertionError('Allocator exceeds SDK capacity: ' + str(case))
            references.append({'input': case, 'sdk': expected, 'model': actual})
        check('3872 resource ceilings and allocation charges match NVIDIA SDK', len(cases) == 3872)
        check('3872 homogeneous allocator fills match NVIDIA SDK', True)
        report['sdk_differential_cases'] = len(cases)
        (out / 'sdk-comparisons.json').write_text(json.dumps(references, indent=2))
        for name, source, gpu, expected_blocks, expected_warps in [
                ('vector-a100', ROOT / 'examples/vector_add.cu', 'a100', 32, 2),
                ('vector-h100', ROOT / 'examples/vector_add.cu', 'h100', 32, 2),
                ('reduce', ROOT / 'examples/shared_reduce.cu', 'h100', 16, 4),
                ('spaces', ROOT / 'tests/conformance/memory_spaces.cu', 'h100', 10, 2)]:
            args = argparse.Namespace(source=source, gpu=gpu, trace=False, trace_limit=200000, events=True,
                events_limit=400000, scheduler='deterministic', seed=43, threads=1, race=True, strict=True,
                timeout=60, out=out / name, nvcc_arg=[])
            code, path = run_program(args, cache, compiled_from=compiled.get(str(source)))
            compiled.setdefault(str(source), path)
            check(name + ' process', code == 0)
            model = analyze_residency(path)
            check(name + ' captured resource ceiling', all(r['resource_ceiling']['blocks_per_sm'] == expected_blocks and
                  r['footprint']['warps'] == expected_warps for r in model['launches']))
            (path / 'sm-residency.json').write_text(json.dumps(model, indent=2))
        source = ROOT / 'tests/conformance/shared_prefix.cu'
        args = argparse.Namespace(source=source, gpu='h100', trace=False, trace_limit=200000, events=True,
            events_limit=400000, scheduler='deterministic', seed=43, threads=1, race=True, strict=True,
            timeout=60, out=out / 'aligned-prefix', nvcc_arg=[])
        code, path = run_program(args, cache, compiled_from=compiled.get(str(source)))
        check('combined shared prefix CUDA process', code == 0)
        try: analyze_residency(path)
        except ValueError as error: check('uncaptured shared prefix refuses guessed resource charge', 'aligned storage prefix' in str(error))
        else: raise AssertionError('Uncaptured aligned shared prefix accepted')
        report['passed'] = True
    finally:
        report['total_checks'] = len(checks)
        (out / 'results.json').write_text(json.dumps(report, indent=2))
    print('PASS SM residency conformance checks=' + str(len(checks)) + ' SDK cases=' + str(report['sdk_differential_cases']), flush=True)
    return 0
