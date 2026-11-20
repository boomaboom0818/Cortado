import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from cortado.run import DEFAULT
from cortado.checkpoints import load


def test_launcher_forwards_term(tmp_path):
    config=tmp_path/'config.json'
    config.write_text(json.dumps(dict(DEFAULT,game='kuhn',representation='exact-v1',iterations=10000000,checkpoint_every=100)))
    root=tmp_path/'run'
    with open(tmp_path/'output.log','w') as log:
        process=subprocess.Popen([sys.executable,'-m','cortado.parallel','--config',str(config),'--run-dir',str(root),'--seeds','42','43'],stdout=log,stderr=log)
        try:
            deadline=time.monotonic()+30
            while not all((root/f'seed-{s}'/'latest.json.gz').exists() for s in (42,43)):
                if process.poll() is not None or time.monotonic()>deadline:
                    raise AssertionError('Workers did not reach checkpoint')
                time.sleep(.05)
            os.kill(process.pid,signal.SIGTERM)
            assert process.wait(timeout=30)==0
            report=json.loads((root/'experiment-summary.json').read_text())
            assert report['stop_reason']=='SIGTERM'
            assert all(load(root/f'seed-{s}'/'latest.json.gz')['stop_reason']=='SIGTERM' for s in (42,43))
        finally:
            if process.poll() is None:
                process.kill();process.wait()
