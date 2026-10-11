"""Compare typed PTX fence parsing with the pinned NVIDIA assembler and execution."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'third_party/pantheonsim'
SOURCE = ROOT / 'tests/fence_semantics.cpp'
SPEC = 'https://docs.nvidia.com/cuda/archive/12.6.3/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar'

def digest(path):
    return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()

def cases():
    result = []
    def add(instruction, kind, sem, scope='none', space='all', version='8.5', target='sm_90',
            extra='', ranged=False, wide=False, offset=0, symbol=False, pred=False, assembler=True):
        expected = dict(kind=kind, sem=sem, scope=scope, space=space, range=int(ranged),
                        bytes=128 if ranged else 0, base='symbol' if symbol else 'reg',
                        wide=int(wide), offset=offset, defs=0,
                        uses=int(pred)+int(ranged and not symbol), address_used=int(ranged and not symbol))
        result.append(dict(instruction=instruction, expected=expected, version=version,
                           target=target, extra=extra, assembler=assembler))
    for scope in ['cta', 'cluster', 'gpu', 'sys']:
        add('fence.'+scope, 'thread', 'acq_rel', scope)
        for sem in ['acq_rel', 'sc']:
            add(f'fence.{sem}.{scope}', 'thread', sem, scope)
            add(f'fence.{scope}.{sem}', 'thread', sem, scope)
    for level, scope in [('cta','cta'),('gl','gpu'),('sys','sys')]:
        add('membar.'+level, 'membar', 'sc', scope)
    for mnemonic in ['fence', 'membar']:
        add(mnemonic+'.proxy.alias', 'alias', 'bidirectional')
    add('fence.proxy.async','async','bidirectional')
    for space in ['global','shared::cta','shared::cluster']:
        add('fence.proxy.async.'+space,'async','bidirectional',space=space)
    add('fence.mbarrier_init.release.cluster','mbarrier_init','release','cluster','shared::cta')
    for scope in ['cta','cluster','gpu','sys']:
        add('fence.proxy.tensormap::generic.release.'+scope, 'tensormap','release',scope)
        add('fence.proxy.tensormap::generic.acquire.'+scope+' [%rd1], 128',
            'tensormap','acquire',scope,ranged=True,wide=True)
    add('fence.proxy.tensormap::generic.acquire.gpu [%r1], 128',
        'tensormap','acquire','gpu',ranged=True)
    add('fence.proxy.tensormap::generic.acquire.gpu [%rd1+16], 128',
        'tensormap','acquire','gpu',ranged=True,wide=True,offset=16)
    add('fence.proxy.tensormap::generic.acquire.gpu [tmap], 128',
        'tensormap','acquire','gpu',ranged=True,symbol=True)
    add('@%p1 fence.proxy.tensormap::generic.acquire.gpu [%rd1], 128',
        'tensormap','acquire','gpu',ranged=True,wide=True,pred=True)
    add('@%p1 fence.cta','thread','acq_rel','cta',pred=True)
    add('fence.sc.gpu','thread','sc','gpu',version='6.0',target='sm_70')
    add('fence.sc.cluster','thread','sc','cluster',version='7.8')
    add('fence.proxy.alias','alias','bidirectional',version='7.5',target='sm_70')
    add('membar.proxy.alias','alias','bidirectional',version='7.5',target='sm_60')
    add('fence.proxy.async','async','bidirectional',version='8.0')
    add('fence.mbarrier_init.release.cluster','mbarrier_init','release','cluster','shared::cta',version='8.0')
    add('fence.proxy.tensormap::generic.acquire.gpu [%rd1], 128',
        'tensormap','acquire','gpu',ranged=True,wide=True,version='8.3')
    # NVIDIA 12.6 no longer assembles sm_10/sm_20. Preserve pre-Volta
    # membar as its own semantic category, without claiming an SC total order.
    add('membar.cta','membar','legacy_membar','cta',version='1.4',target='sm_10',assembler=False)
    add('membar.sys','membar','legacy_membar','sys',version='2.0',target='sm_20',assembler=False)
    invalid = [
        'fence', 'fence.sc', 'fence.acq_rel', 'fence.release.gpu', 'fence.acquire.gpu',
        'fence.relaxed.gpu', 'fence.cta.cta', 'fence.gpu.sys', 'fence.sc.sc.gpu',
        'fence.sc.acq_rel.gpu', 'fence.mystery.gpu', 'fence.sc.gpu.L2::cache_hint',
        'fence.sc.gpu %r1', 'fence..gpu', 'fence.gpu.',
        'membar', 'membar.gpu', 'membar.cluster', 'membar.gl.sys', 'membar.gl.sc',
        'membar.gl [%rd1], 128', 'membar.proxy', 'membar.proxy.async',
        'membar.proxy.tensormap::generic.release.gpu',
        'fence.proxy', 'fence.proxy.unknown', 'fence.proxy.alias.gpu',
        'fence.proxy.alias [%rd1], 128', 'fence.proxy.async.shared',
        'fence.proxy.async.local', 'fence.proxy.async.cta', 'fence.proxy.async.shared::cta.cta',
        'fence.proxy.async.acquire', 'fence.proxy.async.global [%rd1], 128',
        'fence.proxy.tensormap::generic.gpu', 'fence.proxy.tensormap::generic.acquire',
        'fence.proxy.tensormap::generic.sc.gpu', 'fence.proxy.tensormap::generic.acquire.gpu',
        'fence.proxy.tensormap::generic.acquire.gpu [%rd1], 64',
        'fence.proxy.tensormap::generic.acquire.gpu [%rd1], %r1',
        'fence.proxy.tensormap::generic.acquire.gpu [%rd1], 128, 1',
        'fence.proxy.tensormap::generic.release.gpu [%rd1], 128',
        'fence.proxy.tensormap::generic.acquire.acquire [%rd1], 128',
        'fence.proxy.tensormap::generic.gpu.sys [%rd1], 128',
        'fence.mbarrier_init.acquire.cluster', 'fence.mbarrier_init.release.gpu',
        'fence.mbarrier_init.release.cluster [%rd1], 128', 'fence.op_restrict.release.cluster',
    ]
    for instruction in invalid:
        result.append(dict(instruction=instruction,expected=None,version='8.5',target='sm_90',extra='',assembler=True))
    for instruction,version,target in [
        ('fence.sc.gpu','5.0','sm_70'), ('fence.sc.gpu','8.5','sm_60'),
        ('fence.sc.cluster','7.7','sm_90'), ('fence.sc.cluster','8.5','sm_80'),
        ('fence.proxy.alias','7.4','sm_70'), ('membar.proxy.alias','8.5','sm_50'),
        ('fence.proxy.async','7.8','sm_90'), ('fence.proxy.async','8.5','sm_80'),
        ('fence.mbarrier_init.release.cluster','7.8','sm_90'),
        ('fence.mbarrier_init.release.cluster','8.5','sm_80'),
        ('fence.proxy.tensormap::generic.acquire.gpu [%rd1], 128','8.2','sm_90'),
        ('fence.proxy.tensormap::generic.acquire.gpu [%rd1], 128','8.5','sm_80')]:
        result.append(dict(instruction=instruction,expected=None,version=version,target=target,extra='',assembler=True))
    return result

def verify_fences(cache, out):
    cache=Path(cache).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    ptxas=cache/'toolchain/cuda/bin/ptxas'
    version=subprocess.check_output([str(ptxas),'--version'],text=True)
    if 'V12.6.85' not in version: raise AssertionError('Expected pinned CUDA 12.6.85 assembler')
    inputs=[SOURCE,Path(__file__),ENGINE/'include/vgpu/ptx/ast.hpp',ENGINE/'src/ptx/parser.cpp',
            ENGINE/'src/ptx/regalloc.cpp',ENGINE/'src/exec/interpreter.cpp',*(sorted((ROOT/'patches').glob('*.patch')))]
    hashes={str(p.relative_to(ROOT)):digest(p) for p in inputs}
    env=os.environ.copy()
    for key in ['TSIM_TRACE','TSIM_EVENTS','TSIM_TRACE_LIMIT','TSIM_EVENTS_LIMIT']:
        env.pop(key,None)
    env.update(VGPU_THREADS='4',VGPU_SASS='0')
    report=dict(kind='Typed PTX fence syntax, operands and eager CPU execution; not hardware memory-model validation',
                passed=False,component_sha256=hashes,semantic_source=SPEC,assembler_version=version.strip(),
                assembler_sha256=hashlib.sha256(ptxas.read_bytes()).hexdigest(),builds=[],cases=[],
                limitations=['No global scoped causality integration','No asynchronous proxy completion model',
                             'No physical GPU execution','Pre-Volta membar syntax has no assembler comparison'])
    try:
        binaries=[]
        for label,names in [('gcc',['g++-13','g++']),('clang',['clang++-18','clang++'])]:
            compiler=next((shutil.which(n) for n in names if shutil.which(n)),None)
            if not compiler: raise RuntimeError('Missing compiler '+label)
            binary=out/('fence-'+label)
            command=[compiler,'-std=c++20','-O2','-Wall','-Wextra','-Werror',
                     '-I'+str(ENGINE/'include'),'-I'+str(ROOT/'include'),str(SOURCE),
                     str(cache/'build/libvgpu.a'),'-pthread','-o',str(binary)]
            built=subprocess.run(command,text=True,capture_output=True,timeout=180)
            (out/(label+'-build.log')).write_text(built.stdout+built.stderr)
            if built.returncode: raise RuntimeError('Fence compilation failed: '+label+'\n'+built.stderr)
            runtime=subprocess.run([str(binary),'runtime'],text=True,capture_output=True,timeout=60,env=env)
            (out/(label+'-runtime.log')).write_text(runtime.stdout+runtime.stderr)
            marker=re.fullmatch(r'PASS fence_runtime checks=(\d+)\n',runtime.stdout)
            if runtime.returncode or not marker or runtime.stderr: raise AssertionError(label+' runtime\n'+runtime.stdout+runtime.stderr)
            if int(marker[1]) != 2119: raise AssertionError('Fence execution coverage changed without updating the inventory')
            report['builds'].append(dict(name=label,compiler=subprocess.check_output([compiler,'--version'],text=True).splitlines()[0],
                                         command=command,runtime_checks=int(marker[1]),exit_code=0,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest()))
            binaries.append(binary);print(label+': runtime PASS',flush=True)
        for index,case in enumerate(cases()):
            folder=out/('case-'+str(index));folder.mkdir(exist_ok=True)
            source=folder/'input.ptx'
            source.write_text(f".version {case['version']}\n.target {case['target']}\n.address_size 64\n"
                              ".global .align 128 .b8 tmap[256];\n"
                              ".visible .entry k(){ .reg .b64 %rd<4>; .reg .b32 %r<4>; .reg .pred %p<2>; "
                              +case['extra']+case['instruction']+'; ret; }\n')
            entry=dict(case,ptx_sha256=digest(source),parser=[])
            report['cases'].append(entry)
            expected=case['expected']
            for binary in binaries:
                result=subprocess.run([str(binary),'inspect',str(source)],text=True,capture_output=True,env=env,timeout=15)
                (folder/(binary.name+'.log')).write_text(result.stdout+result.stderr)
                if expected is None:
                    if result.returncode!=2 or 'VirtualGPU error [' not in result.stderr:
                        raise AssertionError('Malformed fence accepted: '+case['instruction'])
                    entry['parser'].append(dict(build=binary.name,accepted=False,diagnostic=result.stderr.strip()))
                else:
                    if result.returncode: raise AssertionError(case['instruction']+'\n'+result.stderr)
                    actual=json.loads(result.stdout)
                    if actual!=expected: raise AssertionError('Fence metadata lost: '+case['instruction']+'\n'+repr(actual)+' != '+repr(expected))
                    entry['parser'].append(dict(build=binary.name,accepted=True,metadata=actual))
            if case['assembler']:
                assembled=subprocess.run([str(ptxas),'-arch='+case['target'],str(source),'-o',str(folder/'module.cubin')],
                                         text=True,capture_output=True,timeout=30)
                (folder/'ptxas.log').write_text(assembled.stdout+assembled.stderr)
                entry['nvidia_assembler']=dict(accepted=assembled.returncode==0,exit_code=assembled.returncode,
                                                diagnostic=assembled.stderr.strip())
                if (assembled.returncode==0)!=(expected is not None):
                    raise AssertionError('NVIDIA assembler disagreement: '+case['instruction']+'\n'+assembled.stderr)
                if assembled.returncode==0: entry['nvidia_assembler']['cubin_sha256']=hashlib.sha256((folder/'module.cubin').read_bytes()).hexdigest()
        if hashes!={str(p.relative_to(ROOT)):digest(p) for p in inputs}: raise AssertionError('Fence inputs changed during verification')
        report.update(passed=True,syntax_cases=len(report['cases']),
                      assembler_comparisons=sum(c['assembler'] for c in report['cases']),
                      accepted_cases=sum(c['expected'] is not None for c in report['cases']),
                      rejected_cases=sum(c['expected'] is None for c in report['cases']))
        if (report['syntax_cases'],report['assembler_comparisons'],report['accepted_cases'],report['rejected_cases']) != (112,110,52,60):
            report['passed']=False
            raise AssertionError('Fence syntax coverage changed without updating the inventory')
    finally:
        (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f"PASS typed fences cases={report['syntax_cases']} assembler_comparisons={report['assembler_comparisons']}",flush=True)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,default=ROOT/'.local')
    parser.add_argument('--out',type=Path,default=ROOT/'.local/fence-semantics-conformance')
    args=parser.parse_args();verify_fences(args.cache,args.out)
