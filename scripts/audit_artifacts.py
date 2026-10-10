"""Verify that saved interpreter events identify the exact emitted PTX lines."""
import argparse
import json
from pathlib import Path
import re
from inspector import read_trace

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    reports = []
    for trace in sorted(args.directory.glob('*/warp-trace.jsonl')):
        events, truncated = read_trace(trace)
        lines = trace.with_name('kernel.ptx').read_text().splitlines()
        normalize = lambda value: re.sub(r'\s+', '', value).rstrip(';')
        for event in events:
            line = event['ptx_line']
            assert 1 <= line <= len(lines), str(trace) + ': PTX line outside artifact'
            assert normalize(event['instruction']) == normalize(lines[line - 1]), str(trace) + ': wrong PTX instruction line'
        reports.append({'run': trace.parent.name, 'events': len(events), 'truncated': truncated})
    assert reports, 'No traces to verify'
    (args.directory / 'ptx-line-audit.json').write_text(json.dumps({'passed': True, 'runs': reports}, indent=2))
    print('PASS exact PTX line alignment for ' + str(sum(r['events'] for r in reports)) + ' actual issues across ' + str(len(reports)) + ' saved runs')

if __name__ == '__main__': main()
