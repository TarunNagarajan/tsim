"""Export an actual recorded run as a portable, offline demonstration."""
import argparse
import json
from pathlib import Path
from inspector import read_trace

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    events, truncated = read_trace(args.run / 'warp-trace.jsonl')
    metadata = json.loads((args.run / 'run.json').read_text())
    portable = {key: metadata[key] for key in ['gpu', 'backend', 'scheduler', 'seed', 'threads',
                                              'exit_code', 'engine_revision', 'trace_limit']}
    payload = {'events': events, 'truncated': truncated, 'metadata': portable,
               'source': (args.run / 'source.cu').read_text(), 'ptx': (args.run / 'kernel.ptx').read_text(),
               'stdout': (args.run / 'stdout.log').read_text(), 'stderr': (args.run / 'stderr.log').read_text()}
    # .file is a compiler debug-location record; remove the local directory only.
    source_path = metadata['source']
    payload['ptx'] = payload['ptx'].replace(source_path, 'examples/' + Path(source_path).name)
    encoded = json.dumps(payload, ensure_ascii=True).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    template = (Path(__file__).resolve().parents[1] / 'ui' / 'inspector.html').read_text()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(template.replace('__TSIM_DATA__', encoded), encoding='utf-8')
    print('Exported ' + str(len(events)) + ' actual warp issues to ' + str(args.out))

if __name__ == '__main__': main()
