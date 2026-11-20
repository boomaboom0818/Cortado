"""Bounded train/evaluate cycles. No restart after a resource or correctness stop."""
import argparse
import fcntl
import json
import hashlib
from pathlib import Path
from .checkpoints import save
from .parallel import launch
from .evaluation import evaluate
from .policies import Policy


def campaign(config, directory, seeds, historical, *, cycles=4, pairs=2000, bootstrap_from=None):
    if not 1<=cycles<=4 or pairs<2:
        raise ValueError('Use 1..4 cycles and at least two deal pairs')
    root=Path(directory)
    root.mkdir(parents=True,exist_ok=True)
    settings=json.loads(Path(config).read_text())
    if settings.get('storage')!='sqlite':
        raise ValueError('Sustained campaign requires bounded-memory SQLite storage')
    with open(root/'.campaign.lock','w') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Campaign already running') from error
        state_path=root/'campaign.json'
        identity=dict(config=settings,seeds=seeds,historical=str(Path(historical).resolve()),cycles=cycles,pairs=pairs)
        state=json.loads(state_path.read_text()) if state_path.exists() else dict(identity=identity,completed_cycles=0,status='starting')
        if state['identity']!=identity:
            raise ValueError('Campaign configuration changed')
        if state['status'] in ('resource_stop','failed','completed'):
            return state
        save(state_path,state)
        try:
            for index in range(state['completed_cycles'],cycles):
                resume=(root/'experiment.json').exists()
                result=launch(config,root,seeds,resume=resume,bootstrap_from=bootstrap_from if not resume else None)
                save(root/f'cycle-{index+1}-training.json',result)
                matches={}
                # Sequential evaluation bounds memory and never overlaps trainers.
                for seed in seeds:
                    candidate=root/f'seed-{seed}'/'latest.sqlite'
                    outcomes={}
                    for offset,kind in enumerate(('call','aggressive','uniform','checkpoint')):
                        a=Policy('checkpoint',candidate)
                        b=Policy(kind,historical if kind=='checkpoint' else None)
                        outcomes[kind]=evaluate(a,b,pairs,880000+offset*100000,'holdem')
                        if hasattr(a.data['table'],'close'):a.data['table'].close()
                        if b.data and hasattr(b.data['table'],'close'):b.data['table'].close()
                    with candidate.open('rb') as stream:
                        digest=hashlib.file_digest(stream,'sha256').hexdigest()
                    matches[str(seed)]=dict(checkpoint_sha256=digest,
                                            iterations=result['summaries'][str(seed)]['iterations'],opponents=outcomes)
                save(root/f'cycle-{index+1}-evaluation.json',dict(purpose='development',candidates=matches))
                reasons=[v['stop_reason'] for v in result['summaries'].values()]
                state.update(completed_cycles=index+1,last_stop_reasons=reasons,
                             status='ready' if all(r in ('completed','time_budget') for r in reasons) else 'resource_stop')
                save(state_path,state)
                if state['status']=='resource_stop':
                    return state
            state['status']='completed'
            save(state_path,state)
            return state
        except BaseException:
            state['status']='failed'
            save(state_path,state)
            raise


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',default='configs/train-disk.json')
    p.add_argument('--run-dir',required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43])
    p.add_argument('--historical',default='results/holdem-10k.json.gz')
    p.add_argument('--cycles',type=int,default=4)
    p.add_argument('--pairs',type=int,default=2000)
    p.add_argument('--bootstrap-from')
    args=p.parse_args()
    print(json.dumps(campaign(args.config,args.run_dir,args.seeds,args.historical,
                              cycles=args.cycles,pairs=args.pairs,bootstrap_from=args.bootstrap_from),indent=2))

if __name__=='__main__':main()
