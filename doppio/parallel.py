"""CPU parallelism via isolated processes: one independent CFR table per seed."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .checkpoints import save


def launch(config_path, directory, seeds, resume=False):
    if not seeds or len(seeds)!=len(set(seeds)) or any(s<0 for s in seeds):
        raise ValueError('Provide distinct nonnegative seeds')
    config=json.loads(Path(config_path).read_text())
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    manifest=dict(schema=1,config=config,seeds=seeds,parallelism='independent_processes')
    children=[]
    handles=[]
    handlers={}
    stopped=[]
    start=time.perf_counter()
    def stop(signum,frame):
        stopped.append(signal.Signals(signum).name)
        for child in children:
            if child.poll() is None:
                child.send_signal(signal.SIGTERM)
    with open(directory/'.lock','w') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Another launcher owns this experiment directory') from error
        manifest_path=directory/'experiment.json'
        if resume:
            if not manifest_path.exists() or json.loads(manifest_path.read_text())!=manifest:
                raise ValueError('Resume requires identical experiment manifest')
            if any(not (directory/f'seed-{seed}'/'latest.json.gz').exists() for seed in seeds):
                raise ValueError('All seeds need checkpoints before resume')
        elif manifest_path.exists():
            raise ValueError('Experiment exists; use --resume or a new directory')
        else:
            save(manifest_path,manifest)
        for sig in (signal.SIGINT,signal.SIGTERM):
            handlers[sig]=signal.signal(sig,stop)
        try:
            env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
            for seed in seeds:
                if stopped:
                    break
                worker_config=directory/f'config-{seed}.json'
                save(worker_config,dict(config,seed=seed))
                output=open(directory/f'seed-{seed}.log','a')
                handles.append(output)
                cmd=[sys.executable,'-m','doppio.run','--config',str(worker_config),
                     '--run-dir',str(directory/f'seed-{seed}')]
                if resume:
                    cmd.append('--resume')
                children.append(subprocess.Popen(cmd,env=env,stdout=output,stderr=subprocess.STDOUT))
            codes=[child.wait() for child in children]
            summaries={}
            for seed in seeds:
                p=directory/f'seed-{seed}'/'summary.json'
                if p.exists():
                    summaries[str(seed)]=json.loads(p.read_text())
            report=dict(schema=1,workers=len(children),seeds=seeds,exit_codes=codes,
                        wall_seconds=time.perf_counter()-start,stop_reason=stopped[-1] if stopped else 'completed',
                        summaries=summaries)
            save(directory/'experiment-summary.json',report)
            if any(codes):
                raise RuntimeError(f'Worker failure: {codes}; inspect seed logs in {directory}')
            return report
        finally:
            for child in children:
                if child.poll() is None:
                    child.terminate()
            for child in children:
                child.wait()
            for handle in handles:
                handle.close()
            for sig,handler in handlers.items():
                signal.signal(sig,handler)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/train-cpu.json')
    p.add_argument('--run-dir',required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43])
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    try:
        print(json.dumps(launch(args.config,args.run_dir,args.seeds,args.resume),indent=2))
    except (ValueError,RuntimeError) as error:
        p.error(str(error))

if __name__=='__main__': main()
