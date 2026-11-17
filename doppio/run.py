"""Isolated, locked experiment directory for sustained bot training."""
import argparse
import fcntl
import json
from importlib.metadata import version
from pathlib import Path
import shutil
from .checkpoints import save
from .training import train

DEFAULT = dict(schema=1, game='holdem', target='bot', representation='suits-v1',
               seed=42, iterations=100000, checkpoint_every=1000,
               max_seconds=3600, max_rss_mb=4096)

def run(config_path, directory, resume=False):
    with open(config_path) as stream:
        config=json.load(stream)
    if set(config)-{'storage','stack_bb'}!=set(DEFAULT) or config['schema']!=1 or config.get('storage','memory') not in ('memory','sqlite'):
        raise ValueError('Config must contain exactly the version-1 run fields')
    for package,expected in (('open-spiel','2.0.2'),('numpy','2.5.4')):
        if version(package)!=expected:
            raise ValueError(f'{package} must be pinned to {expected}')
    directory=Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    with open(directory/'.lock','w') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Another trainer owns this run directory') from error
        manifest=directory/'config.json'
        suffix = 'sqlite' if config.get('storage') == 'sqlite' else 'json.gz'
        checkpoint=directory/f'latest.{suffix}'
        if resume:
            if not checkpoint.exists() or not manifest.exists():
                raise ValueError('Resume requires an existing manifest and checkpoint')
            if json.loads(manifest.read_text())!=config:
                raise ValueError('Resume configuration differs from immutable run manifest')
        elif manifest.exists() or checkpoint.exists():
            raise ValueError('Run directory already initialized; use --resume or a new directory')
        if shutil.disk_usage(directory).free < 1024**3:
            raise ValueError('At least 1 GiB free disk required to start')
        if not resume:
            save(manifest,config)
        result=train(config['game'],config['iterations'],config['seed'],checkpoint,
                     resume=checkpoint if resume else None,target=config['target'],
                     representation=config['representation'],stack_bb=config.get('stack_bb',100),checkpoint_every=config['checkpoint_every'],
                     max_seconds=config['max_seconds'],max_rss_mb=config['max_rss_mb'],
                     metrics_path=directory/'metrics.jsonl')
        save(directory/'summary.json',result)
        # Keep a historical inference/resume artifact for each invocation.
        shutil.copy2(checkpoint,directory/f"iteration-{result['iterations']}.{suffix}")
        return result

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='configs/train-bot.json')
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    try:
        print(json.dumps(run(args.config,args.run_dir,args.resume),indent=2))
    except ValueError as error:
        parser.error(str(error))

if __name__=='__main__': main()
