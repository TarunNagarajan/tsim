"""Overlapping patches, old-prefix upgrades, newline handling, and edit safety."""
import difflib
from pathlib import Path
import subprocess
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from engine_patches import apply_engine_patches

def verify_patches(cache):
    with tempfile.TemporaryDirectory(prefix='patch-model-', dir=cache) as folder:
        root = Path(folder)
        engine = root / 'engine'; engine.mkdir()
        subprocess.run(['git', '-C', str(engine), 'init', '-q'], check=True)
        first = 'alpha\nbeta\ngamma\n'; second = 'alpha\nmanaged\ngamma\n'; third = 'alpha\nmanaged-new\ngamma\n'
        (engine / 'one.txt').write_text(first)
        (engine / 'two.txt').write_text(first)
        subprocess.run(['git', '-C', str(engine), 'add', '.'], check=True)
        subprocess.run(['git', '-C', str(engine), '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                        'commit', '-qm', 'base'], check=True)
        pin = subprocess.check_output(['git', '-C', str(engine), 'rev-parse', 'HEAD'], text=True).strip()
        def patch(name, before, after, target):
            result = root / name
            result.write_text(''.join(difflib.unified_diff(before.splitlines(True), after.splitlines(True),
                                fromfile='a/' + target, tofile='b/' + target)), newline='\r\n')
            return result
        p1 = patch('one.patch', first, second, 'one.txt')
        p2 = patch('two.patch', second, third, 'one.txt')
        p3 = patch('three.patch', first, second, 'two.txt')
        apply_engine_patches(engine, [p1], pin, root / 'cache')
        assert (engine / 'one.txt').read_text() == second
        # A Windows newline checkout of the old managed prefix upgrades safely.
        (engine / 'one.txt').write_bytes(second.replace('\n', '\r\n').encode())
        apply_engine_patches(engine, [p1, p2, p3], pin, root / 'cache')
        assert (engine / 'one.txt').read_bytes() == third.encode()
        assert not apply_engine_patches(engine, [p1, p2, p3], pin, root / 'cache')['updated_targets']
        (engine / 'one.txt').write_text(first)
        (engine / 'two.txt').write_text('user edit\n')
        try: apply_engine_patches(engine, [p1, p2, p3], pin, root / 'cache')
        except RuntimeError as e: assert 'Unrecognized local engine changes' in str(e)
        else: raise AssertionError('Local edit was overwritten')
        assert (engine / 'one.txt').read_text() == first, 'Validation must precede every mutation'
        assert (engine / 'two.txt').read_text() == 'user edit\n'
        traversal = patch('escape.patch', first, second, '../escape.txt')
        try: apply_engine_patches(engine, [traversal], pin, root / 'cache')
        except RuntimeError as e: assert 'outside engine' in str(e)
        else: raise AssertionError('Escaped engine root')
    return True

if __name__ == '__main__':
    print(verify_patches(sys.argv[1]))
