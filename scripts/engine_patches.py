"""Replay an ordered engine patch stack, protecting unknown local edits."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

def apply_engine_patches(engine, patches, pin, cache):
    engine = Path(engine).resolve()
    patches = list(patches)
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    contents = {p: p.read_bytes().replace(b'\r\n', b'\n') for p in patches}
    targets = sorted({line[6:] for data in contents.values()
                      for line in data.decode().splitlines() if line.startswith('+++ b/')})
    for name in targets:
        if not (engine / name).resolve().is_relative_to(engine):
            raise RuntimeError('Patch target outside engine: ' + name)
    # Replay from the pinned Git objects before touching the checkout. Earlier
    # patches may overlap later ones, making isolated reverse checks invalid.
    # Known prefix states also allow safe upgrades from an older patch stack.
    known = {name: set() for name in targets}
    with tempfile.TemporaryDirectory(prefix='patch-replay-', dir=cache) as folder:
        replay = Path(folder)
        # A default cache lives inside the tsim checkout. Without an isolated
        # repository, git apply discovers that parent and silently skips Git
        # diff paths outside its subdirectory prefix, despite returning success.
        subprocess.run(['git', '-C', str(replay), 'init', '-q'], check=True, capture_output=True)
        for name in targets:
            data = subprocess.check_output(['git', '-C', str(engine), 'show', pin + ':' + name])
            data = data.replace(b'\r\n', b'\n')
            target = replay / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            known[name].add(data)
        for index, patch in enumerate(patches):
            before = {name: (replay / name).read_bytes() for name in targets}
            saved = replay / ('stack-' + str(index) + '.patch')
            saved.write_bytes(contents[patch])
            for suffix in [['--check'], []]:
                result = subprocess.run(['git', '-C', str(replay), 'apply', *suffix, str(saved)],
                                        text=True, capture_output=True)
                if result.returncode:
                    raise RuntimeError('Engine patch replay failed: ' + patch.name + '\n' + result.stderr)
            if all((replay / name).read_bytes() == before[name] for name in targets):
                raise RuntimeError('Engine patch replay changed no targets: ' + patch.name)
            for name in targets:
                known[name].add((replay / name).read_bytes())
        desired = {name: (replay / name).read_bytes() for name in targets}
        # Validate every target first. Never erase a user edit merely because
        # another target happened to have an accepted older managed version.
        for name in targets:
            current = (engine / name).read_bytes().replace(b'\r\n', b'\n')
            if current not in known[name]:
                raise RuntimeError('Unrecognized local engine changes; preserved: ' + name)
        changed = []
        for name in targets:
            target = engine / name
            if target.read_bytes() != desired[name]:
                target.write_bytes(desired[name])
                changed.append(name)
    report = {'passed': True, 'engine_revision': pin,
              'patch_sha256': {p.name: hashlib.sha256(contents[p]).hexdigest() for p in patches},
              'target_sha256': {name: hashlib.sha256(data).hexdigest() for name, data in desired.items()},
              'updated_targets': changed}
    (cache / 'engine-patch-state.json').write_text(json.dumps(report, indent=2))
    return report
