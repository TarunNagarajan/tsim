"""Optional real Chromium smoke check for a generated offline inspector."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

CHECK = r'''<script>
window.addEventListener('load', async () => {
  const checks=[];
  const assert=(ok,label)=>{if(!ok)throw Error(label);checks.push(label)};
  try {
    assert(document.querySelectorAll('#lanes .lane').length===32,'32 lanes rendered');
    assert(document.querySelectorAll('#kernel option').length>0,'kernel choices populated');
    assert(document.querySelectorAll('#warp option').length>0,'warp choices populated');
    assert(document.querySelectorAll('#rows tr').length>0,'actual instruction stream rendered');
    const initial=document.getElementById('position').textContent;
    document.getElementById('next').click();
    assert(document.getElementById('position').textContent!==initial,'next instruction works');
    document.getElementById('prev').click();
    assert(document.getElementById('position').textContent===initial,'previous instruction works');
    const slider=document.getElementById('step');
    slider.value=slider.max;slider.dispatchEvent(new Event('input'));
    assert(document.getElementById('next').disabled,'slider reaches final instruction');
    slider.value=0;slider.dispatchEvent(new Event('input'));
    document.getElementById('play').click();
    await new Promise(resolve=>setTimeout(resolve,400));
    assert(document.getElementById('position').textContent!==initial,'playback advances');
    document.getElementById('play').click();
    const warp=document.getElementById('warp');
    if(warp.options.length>1){warp.selectedIndex=warp.options.length-1;warp.dispatchEvent(new Event('change'));assert(document.getElementById('position').textContent.startsWith('1 /'),'warp selection resets position')}
    assert(document.getElementById('source').textContent.includes('__global__'),'source rendered as text');
    assert(document.querySelector('#ptxLines .line-current'),'PTX instruction highlighted');
    const recorded=JSON.parse(document.getElementById('traceData').textContent).events;
    const partial=recorded.find(event=>event.execution_mask>0&&event.execution_mask<0xffffffff);
    if(partial){
      const kernel=document.getElementById('kernel');kernel.value=partial.kernel+' / launch '+(partial.launch||1);kernel.dispatchEvent(new Event('change'));
      warp.value='('+partial.block.join(',')+') / warp '+partial.warp;warp.dispatchEvent(new Event('change'));
      const subset=recorded.filter(event=>event.kernel===partial.kernel&&event.launch===partial.launch&&event.warp===partial.warp&&event.block.join(',')===partial.block.join(','));
      slider.value=subset.findIndex(event=>event.sequence===partial.sequence);slider.dispatchEvent(new Event('input'));
      let mask=BigInt(partial.execution_mask),expected=0;while(mask){expected+=Number(mask&1n);mask>>=1n}
      assert(document.querySelectorAll('#lanes .active').length===expected,'partial mask renders correct active lanes');
    }
    assert(window.scrollY===0,'instruction navigation preserves page position');
    document.body.dataset.uiResult='PASS';document.body.dataset.uiChecks=JSON.stringify(checks);
    await fetch('/result',{method:'POST',body:JSON.stringify({passed:true,checks})});
  } catch(error) {document.body.dataset.uiResult='FAIL';document.body.dataset.uiFailure=error.message;await fetch('/result',{method:'POST',body:JSON.stringify({passed:false,failure:error.message})});}
});
</script>'''

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--browser', required=True, help='Installed Chromium/Chrome/Edge executable')
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = out / 'ui-smoke.html'
    report.write_text(args.report.read_text(encoding='utf-8').replace('</body>', CHECK + '</body>'), encoding='utf-8')
    received = threading.Event()
    browser_result = {}
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *values, **kwargs):
            super().__init__(*values, directory=str(out), **kwargs)
        def log_message(self, *values): pass
        def do_POST(self):
            if self.path != '/result': self.send_error(404); return
            browser_result.update(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            self.send_response(200); self.end_headers(); self.wfile.write(b'OK')
            received.set()
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    command = [args.browser, '--headless', '--disable-gpu', '--no-first-run', '--disable-extensions',
               '--no-default-browser-check', '--user-data-dir=' + str(out / 'profile'),
               '--disk-cache-dir=' + str(out / 'cache'), '--virtual-time-budget=2500',
               '--dump-dom', '--screenshot=' + str(out / 'inspector.png'), '--window-size=1440,1200',
               f'http://127.0.0.1:{server.server_port}/ui-smoke.html']
    env = os.environ.copy()
    (out / 'tmp').mkdir(exist_ok=True)
    env['TEMP'] = env['TMP'] = str(out / 'tmp')
    result = subprocess.run(command, env=env, text=True, encoding='utf-8', errors='replace',
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60)
    received.wait(25)
    server.shutdown()
    (out / 'dom.html').write_text(result.stdout, encoding='utf-8')
    (out / 'browser.log').write_text(result.stderr, encoding='utf-8')
    passed = browser_result.get('passed', False) and result.returncode == 0
    (out / 'result.json').write_text(json.dumps({'passed': passed, 'browser_exit_code': result.returncode,
                                               'report': str(args.report), 'command': command,
                                               'browser_checks': browser_result}, indent=2))
    if passed: print('PASS Chromium inspector rendering, navigation, slider, playback, warp selection, and PTX highlight')
    else:
        print('FAIL Chromium inspector: ' + browser_result.get('failure', 'see browser.log and dom.html'))
    return 0 if passed else 1

if __name__ == '__main__':
    raise SystemExit(main())
