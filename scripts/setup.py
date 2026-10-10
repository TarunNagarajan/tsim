"""Install a workspace-local CUDA compiler and build the CPU interpreter."""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tarfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / 'third_party' / 'pantheonsim'
PIN = '771bb8f96abc3721700219d88fa74e10d43dcfd5'
BASE = 'https://developer.download.nvidia.com/compute/cuda/redist/'

def download(url, path, checksum=None):
    if path.exists() and (not checksum or hashlib.sha256(path.read_bytes()).hexdigest() == checksum):
        return
    print('Downloading ' + path.name, flush=True)
    part = path.with_suffix(path.suffix + '.part')
    with urllib.request.urlopen(url, timeout=120) as src, part.open('wb') as dst:
        shutil.copyfileobj(src, dst)
    if checksum and hashlib.sha256(part.read_bytes()).hexdigest() != checksum:
        raise RuntimeError('Checksum mismatch: ' + path.name)
    part.replace(path)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=ROOT / '.local')
    parser.add_argument('--jobs', type=int, default=2)
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('Run through tsim.py setup on Windows, or directly on Linux.')
    cache = args.cache.resolve()
    (cache / 'tmp').mkdir(parents=True, exist_ok=True)
    os.environ['TMPDIR'] = str(cache / 'tmp')
    downloads = cache / 'downloads'
    downloads.mkdir(parents=True, exist_ok=True)
    toolkit = cache / 'toolchain' / 'cuda'
    manifest_path = downloads / 'cuda-manifest.json'
    download(BASE + 'redistrib_12.6.3.json', manifest_path)
    manifest = json.loads(manifest_path.read_text())
    toolkit.mkdir(parents=True, exist_ok=True)
    def component(name):
        item = manifest[name]['linux-x86_64']
        archive = downloads / Path(item['relative_path']).name
        download(BASE + item['relative_path'], archive, item['sha256'])
        marker = toolkit / ('.' + name + '-ready')
        if marker.exists(): return
        with tarfile.open(archive) as tar:
            prefix = tar.getmembers()[0].name.split('/')[0]
            # NVIDIA archives are checksum verified before preserving their symlinks.
            for member in tar.getmembers():
                relative = Path(member.name).relative_to(prefix)
                if not relative.parts: continue
                if '..' in relative.parts: raise RuntimeError('Unsafe archive path')
                member.name = str(relative)
                tar.extract(member, toolkit)
        marker.touch()
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(component, ['cuda_nvcc', 'cuda_cudart', 'cuda_cccl']))
    cmake = cache / 'toolchain' / 'cmake-wheel' / 'cmake' / 'data' / 'bin' / 'cmake'
    if not cmake.exists():
        data = json.load(urllib.request.urlopen('https://pypi.org/pypi/cmake/3.31.6/json', timeout=120))
        item = next(x for x in data['urls'] if 'manylinux2014_x86_64' in x['filename'])
        wheel = downloads / item['filename']
        download(item['url'], wheel, item['digests']['sha256'])
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(cache / 'toolchain' / 'cmake-wheel')
    for binary in cmake.parent.iterdir():
        binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    for binary in (toolkit / 'bin').iterdir():
        if binary.is_file(): binary.chmod(binary.stat().st_mode | stat.S_IXUSR)
    subprocess.run(['git', '-C', str(ROOT), 'submodule', 'update', '--init', '--recursive'], check=True)
    actual = subprocess.check_output(['git', '-C', str(ENGINE), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != PIN: raise RuntimeError('Unexpected engine revision: ' + actual)
    patches = sorted((ROOT / 'patches').glob('*.patch'))
    # A Windows checkout and a Linux checkout may have different newlines.
    # Canonicalize only patch targets, keeping the patch application identical.
    for patch in patches:
        content = patch.read_bytes().replace(b'\r\n', b'\n')
        if content != patch.read_bytes(): patch.write_bytes(content)
        for line in content.decode().splitlines():
            if not line.startswith('+++ b/'): continue
            target = (ENGINE / line[6:]).resolve()
            if not target.is_relative_to(ENGINE.resolve()):
                raise RuntimeError('Patch target outside engine')
            if target.is_file():
                data = target.read_bytes()
                canonical = data.replace(b'\r\n', b'\n')
                if data != canonical: target.write_bytes(canonical)
    for patch in patches:
        check = subprocess.run(['git', '-C', str(ENGINE), 'apply', '--reverse', '--check', str(patch)], capture_output=True)
        if check.returncode:
            subprocess.run(['git', '-C', str(ENGINE), 'apply', '--check', str(patch)], check=True)
            subprocess.run(['git', '-C', str(ENGINE), 'apply', str(patch)], check=True)
    build = cache / 'build'
    subprocess.run([str(cmake), '-S', str(ENGINE), '-B', str(build), '-DCMAKE_BUILD_TYPE=Release',
                    '-DCMAKE_CXX_COMPILER=clang++', '-DCMAKE_C_COMPILER=clang',
                    '-DVGPU_ENABLE_LTO=OFF', '-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF', '-DCUDA_ABI_INCLUDE=' + str(toolkit / 'include'),
                    '-DCMAKE_CXX_FLAGS=', '-DTSIM_INCLUDE_DIR=' + str(ROOT / 'include'),
                    '-DVGPU_FETCH_CUDNN_HEADERS=OFF'], check=True)
    subprocess.run([str(cmake), '--build', str(build), '--target', 'vgpu_cli', 'vgpucudart',
                    'vgpucuda', '-j', str(args.jobs)], check=True)
    print('CPU CUDA simulator ready at ' + str(build), flush=True)

if __name__ == '__main__':
    main()
