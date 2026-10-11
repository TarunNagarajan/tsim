"""Build scoped causality references with independent compilers and mutations."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / 'include/tsim/scoped_memory_order.hpp'
SOURCE = ROOT / 'tests/scoped_memory_order_model.cpp'

def digest(path):
    return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()

def verify_scoped_memory(out):
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=True)
    gcc = shutil.which('g++-13') or shutil.which('g++')
    clang = shutil.which('clang++-18') or shutil.which('clang++')
    if not gcc or not clang: raise RuntimeError('Scoped memory references require GCC and Clang C++20 compilers')
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in [HEADER, SOURCE, Path(__file__)]}
    report = {'kind':'Observed per-thread scoped generic-proxy causality references; not integrated interpreter or hardware validation',
              'passed':False, 'runtime_integrated':False, 'hardware_validation':'unverified',
              'component_sha256':hashes, 'builds':[], 'mutations':[],
              'semantic_source':'https://docs.nvidia.com/cuda/archive/12.6.3/parallel-thread-execution/index.html#memory-consistency-model'}
    def compile_run(label, compiler, options, include=ROOT/'include', mutation=False):
        binary=out/label
        command=[compiler,'-std=c++20','-Wall','-Wextra','-Werror',*options,'-I'+str(include),str(SOURCE),'-o',str(binary)]
        built=subprocess.run(command,text=True,capture_output=True,timeout=180)
        (out/(label+'-build.log')).write_text(built.stdout+built.stderr)
        if built.returncode: raise RuntimeError('Reference compilation failed: '+label+'\n'+built.stderr)
        result=subprocess.run([str(binary)],text=True,capture_output=True,timeout=90)
        (out/(label+'.log')).write_text(result.stdout+result.stderr)
        if mutation:
            if result.returncode!=1 or not result.stderr.startswith('FAIL scoped_memory_order: '):
                raise AssertionError('Mutation was not rejected by behavioral reference: '+label)
            return {'name':label,'exit_code':result.returncode,'diagnostic':result.stderr.strip(),'behavioral_rejection':True}
        if result.returncode: raise AssertionError(label+'\n'+result.stderr)
        marker=re.fullmatch(r'PASS scoped_memory_order checks=(\d+) scenarios=(\d+) dag_relations=(\d+)\n',result.stdout)
        if not marker or result.stderr: raise AssertionError('Missing or altered reference output: '+label)
        checks,scenarios,relations=map(int,marker.groups())
        if (checks,scenarios,relations)!=(6888,3616,44544): raise AssertionError('Reference coverage changed without updating expected inventory')
        return {'name':label,'compiler':subprocess.check_output([compiler,'--version'],text=True).splitlines()[0],
                'command':command,'exit_code':0,'checks':checks,'scope_scenarios':scenarios,'dag_relations':relations,
                'binary_sha256':hashlib.sha256(binary.read_bytes()).hexdigest()}
    mutations = {
        'cta-scope-widened': [('if (scope == Scope::System) return true;', 'if (scope == Scope::System || scope == Scope::Cta) return true;')],
        'acqrel-treated-as-sc': [
            ('if (sequentially_consistent) for (const auto& p : sc_fences_)','for (const auto& p : sc_fences_)'),
            ('if (sequentially_consistent) retain(sc_fences_, endpoint(actor, scope));','if (sequentially_consistent || !sequentially_consistent) retain(sc_fences_, endpoint(actor, scope));')],
        'fence-snapshot-leak': [('for (const auto& p : state.fences) retain(publications, p);','for (const auto& p : state.fences) retain(publications, endpoint(actor, p.scope));')],
        'width-pattern-collapsed': [('const MemoryKey key{location, bytes};','const MemoryKey key{location, 4};')],
        'relaxed-rmw-acquires': [('if (acq) acquire(actor, scope, state.observations[key], key);','if (acq || kind == Kind::Rmw) acquire(actor, scope, state.observations[key], key);')],
        'failed-cas-writes': [('write = kind != Kind::Read && did_write;','write = kind != Kind::Read;')],
    }
    try:
        for label,compiler,flags in [('gcc-release',gcc,['-O2']),('clang-release',clang,['-O2']),
                                    ('gcc-address-undefined-sanitizers',gcc,['-O1','-g','-fsanitize=address,undefined','-fno-omit-frame-pointer'])]:
            report['builds'].append(compile_run(label,compiler,flags)); print(label+': PASS',flush=True)
        original=HEADER.read_text()
        for label,edits in mutations.items():
            content=original
            for before,after in edits:
                if content.count(before)!=1: raise AssertionError('Mutation anchor is no longer unique: '+label)
                content=content.replace(before,after)
            include=out/'mutations'/label/'include';target=include/'tsim/scoped_memory_order.hpp'
            target.parent.mkdir(parents=True,exist_ok=True);target.write_text(content)
            report['mutations'].append(compile_run(label,gcc,['-O0'],include,True));print(label+': rejected',flush=True)
        if hashes!={str(p.relative_to(ROOT)):digest(p) for p in [HEADER,SOURCE,Path(__file__)]}:
            raise AssertionError('Reference inputs changed during verification')
        report.update(passed=True,total_checks=6888,scope_scenarios=3616,independent_dag_relations=44544,behavioral_mutations=6)
    finally:
        (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS scoped memory conformance scenarios=3616 dag_relations=44544 behavioral_mutations=6',flush=True)
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=ROOT/'.local/scoped-memory-conformance')
    verify_scoped_memory(parser.parse_args().out)
