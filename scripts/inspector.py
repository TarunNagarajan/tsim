"""Turn actual interpreter events into a self-contained offline HTML inspector."""
import json
from pathlib import Path

def read_trace(path):
    events, truncated = [], False
    with Path(path).open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            try: event = json.loads(line)
            except json.JSONDecodeError as exc: raise RuntimeError(f'Invalid trace JSON at line {number}') from exc
            if event.get('type') == 'truncated': truncated = True
            elif event.get('type') == 'issue': events.append(event)
            else: raise RuntimeError(f'Unknown trace event at line {number}')
    return events, truncated

def create_report(trace, target):
    trace, target = Path(trace), Path(target)
    events, truncated = read_trace(trace)
    def sibling(name):
        p = trace.parent / name
        return p.read_text(encoding='utf-8', errors='replace') if p.exists() else ''
    payload = {'events': events, 'truncated': truncated, 'source': sibling('source.cu'),
               'ptx': sibling('kernel.ptx'), 'stdout': sibling('stdout.log'), 'stderr': sibling('stderr.log')}
    if (trace.parent / 'run.json').exists(): payload['metadata'] = json.loads(sibling('run.json'))
    # JSON is data, even if a source comment contains a closing script tag.
    data = json.dumps(payload, ensure_ascii=True).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    template = (Path(__file__).resolve().parents[1] / 'ui' / 'inspector.html').read_text(encoding='utf-8')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(template.replace('__TSIM_DATA__', data), encoding='utf-8')
    return {'events': len(events), 'truncated': truncated}
