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


def launch(config_path, directory, seeds, resume=False, bootstrap_from=None):
    if not seeds or len(seeds)!=len(set(seeds)) or any(s<0 for s in seeds):
        raise ValueError('Provide distinct nonnegative seeds')
    if resume and bootstrap_from:
        raise ValueError('Bootstrap and resume are mutually exclusive')
    config=json.loads(Path(config_path).read_text())
    suffix = 'sqlite' if config.get('storage') == 'sqlite' else 'json.gz'
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
            if any(not (directory/f'seed-{seed}'/f'latest.{suffix}').exists() for seed in seeds):
                raise ValueError('All seeds need checkpoints before resume')
        elif manifest_path.exists():
            raise ValueError('Experiment exists; use --resume or a new directory')
        else:
            if bootstrap_from:
                import shutil
                source=Path(bootstrap_from)
                from .checkpoints import load
                for seed in seeds:
                    checkpoint=next((source/f'seed-{seed}').glob('latest.sqlite'),source/f'seed-{seed}'/'latest.json.gz')
                    data=load(checkpoint, config['game'])
                    if data['seed'] != seed or data.get('representation','exact-v1') != config['representation']:
                        raise ValueError('Bootstrap seed or representation mismatch')
                    if hasattr(data['table'],'close'):
                        data['table'].close()
                    del data
                for seed in seeds:
                    worker=directory/f'seed-{seed}'
                    worker.mkdir(exist_ok=True)
                    save(worker/'config.json',dict(config,seed=seed))
                    checkpoint=next((source/f'seed-{seed}').glob('latest.sqlite'),source/f'seed-{seed}'/'latest.json.gz')
                    if checkpoint.suffix == '.sqlite' and suffix == 'sqlite':
                        shutil.copy2(checkpoint,worker/f'latest.{suffix}')
                    else:
                        data=load(checkpoint)
                        save(worker/f'latest.{suffix}',data)
                        if hasattr(data['table'],'close'):
                            data['table'].close()
                        del data
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
                cmd=[sys.executable,'-m','cortado.run','--config',str(worker_config),
                     '--run-dir',str(directory/f'seed-{seed}')]
                if resume or bootstrap_from:
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
    p.add_argument('--bootstrap-from',help='Copy compatible existing checkpoints into a fresh experiment')
    args=p.parse_args()
    try:
        print(json.dumps(launch(args.config,args.run_dir,args.seeds,args.resume,args.bootstrap_from),indent=2))
    except (ValueError,RuntimeError) as error:
        p.error(str(error))

if __name__=='__main__': main()
