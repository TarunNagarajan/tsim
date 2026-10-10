"""Compiler metadata format and cross-producer consistency checks."""
from copy import deepcopy
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from compiler_resources import parse_ptxas_resources, parse_cuobjdump_resources, crosscheck_resources

def verify_compiler_components():
    checks = []
    def check(name, condition):
        if not condition: raise AssertionError(name)
        checks.append(name)
    def reject(name, action):
        try: action()
        except ValueError: checks.append(name)
        else: raise AssertionError(name)
    compiler = '''ptxas info    : Compiling entry function 'kernel' for 'sm_90'
ptxas info    : Function properties for kernel
    16 bytes stack frame, 8 bytes spill stores, 12 bytes spill loads
ptxas info    : Used 24 registers, used 1 barriers, 16 bytes smem, 8 bytes lmem, 512 bytes cmem[0]
'''
    dump = '''Resource usage:
 Common:
  GLOBAL:0
 Function kernel:
  REG:24 STACK:16 SHARED:1040 LOCAL:8 CONSTANT[0]:512 TEXTURE:0 SURFACE:0 SAMPLER:0
'''
    p = parse_ptxas_resources(compiler, 'sm_90'); record = p['entries']['kernel']
    check('compiler registers and barrier counts', record['registers'] == 24 and record['barriers'] == 1)
    check('stack and spill directions stay separate', record['stack_bytes'] == 16 and record['spill_store_bytes'] == 8 and record['spill_load_bytes'] == 12)
    check('user prefix local and constant bytes retained', record['shared_prefix_bytes'] == 16 and record['local_bytes'] == 8 and record['constant_banks'] == {'0': 512})
    c = parse_cuobjdump_resources(dump)
    result = crosscheck_resources(deepcopy(p), c)
    check('encoded reservation not confused with prefix', result['entries']['kernel']['encoded_system_reservation_bytes'] == 1024 and
          result['entries']['kernel']['shared_prefix_bytes'] == 16 and result['entries']['kernel']['cubin_raw_shared_bytes'] == 1040)
    reject('wrong architecture rejected', lambda: parse_ptxas_resources(compiler, 'sm_80'))
    reject('duplicate entry rejected', lambda: parse_ptxas_resources(compiler + compiler, 'sm_90'))
    reject('missing stack rejected', lambda: parse_ptxas_resources(compiler.replace('    16 bytes stack frame, 8 bytes spill stores, 12 bytes spill loads\n', ''), 'sm_90'))
    reject('missing barriers rejected', lambda: parse_ptxas_resources(compiler.replace('used 1 barriers, ', ''), 'sm_90'))
    reject('unknown field rejected', lambda: parse_ptxas_resources(compiler.replace('16 bytes smem', '16 bytes unknown'), 'sm_90'))
    reject('duplicate size rejected', lambda: parse_ptxas_resources(compiler.replace('16 bytes smem', '16 bytes smem, 16 bytes smem'), 'sm_90'))
    reject('duplicate constant bank rejected', lambda: parse_ptxas_resources(compiler.replace('512 bytes cmem[0]', '512 bytes cmem[0], 512 bytes cmem[0]'), 'sm_90'))
    reject('unparented Used record rejected', lambda: parse_ptxas_resources('ptxas info : Used 24 registers, used 1 barriers', 'sm_90'))
    reject('missing cubin field rejected', lambda: parse_cuobjdump_resources(dump.replace('REG:24 ', '')))
    reject('duplicate cubin field rejected', lambda: parse_cuobjdump_resources(dump.replace('REG:24 ', 'REG:24 REG:24 ')))
    reject('register producer disagreement rejected', lambda: crosscheck_resources(deepcopy(p), parse_cuobjdump_resources(dump.replace('REG:24', 'REG:25'))))
    reject('stack producer disagreement rejected', lambda: crosscheck_resources(deepcopy(p), parse_cuobjdump_resources(dump.replace('STACK:16', 'STACK:32'))))
    reject('constant producer disagreement rejected', lambda: crosscheck_resources(deepcopy(p), parse_cuobjdump_resources(dump.replace('CONSTANT[0]:512', 'CONSTANT[0]:516'))))
    reject('unknown shared encoding rejected', lambda: crosscheck_resources(deepcopy(p), parse_cuobjdump_resources(dump.replace('SHARED:1040', 'SHARED:4112'))))
    reject('missing entry from cubin rejected', lambda: crosscheck_resources(deepcopy(p), {'other': c['kernel']}))
    zero_shared = parse_ptxas_resources(compiler.replace(', 16 bytes smem', ''), 'sm_90')
    check('omitted zero shared prefix preserved', zero_shared['entries']['kernel']['shared_prefix_bytes'] == 0)
    device = compiler + 'ptxas info : Function properties for helper\n    8 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n'
    check('device stack kept apart from entry resources', parse_ptxas_resources(device, 'sm_90')['device_functions'] == {
          'helper': {'stack_bytes': 8, 'spill_store_bytes': 0, 'spill_load_bytes': 0}})
    cumulative = compiler.replace('used 1 barriers, ', 'used 1 barriers, 24 bytes cumulative stack size, ')
    parsed = parse_ptxas_resources(cumulative, 'sm_90')
    check('cumulative stack kept separate from own frame', parsed['entries']['kernel']['cumulative_stack_bytes'] == 24 and
          parsed['entries']['kernel']['stack_bytes'] == 16)
    check('cubin STACK checked against cumulative requirement when supplied', crosscheck_resources(
          parsed, parse_cuobjdump_resources(dump.replace('STACK:16', 'STACK:24')))['entries']['kernel']['cumulative_stack_bytes'] == 24)
    reject('duplicate cumulative stack rejected', lambda: parse_ptxas_resources(
          cumulative.replace('24 bytes cumulative stack size', '24 bytes cumulative stack size, 24 bytes cumulative stack size'), 'sm_90'))
    return checks

if __name__ == '__main__':
    print('PASS compiler resource components checks=' + str(len(verify_compiler_components())))
