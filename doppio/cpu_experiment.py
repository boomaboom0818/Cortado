"""Sequential concurrency sweep; training artifacts live outside git."""
import argparse
import json
from pathlib import Path
from .parallel import launch
from .checkpoints import save
from .evaluation import evaluate
from .policies import Policy
from .run import DEFAULT


def experiment(directory,iterations=1000):
    root=Path(directory)
    root.mkdir(parents=True,exist_ok=True)
    config=root/'pilot-config.json'
    save(config,dict(DEFAULT,iterations=iterations,checkpoint_every=iterations,max_seconds=300,max_rss_mb=1024))
    reports=[]
    for workers in (1,2,4):
        result=launch(config,root/f'workers-{workers}',list(range(42,42+workers)))
        result['iterations_per_second']=sum(s['iterations'] for s in result['summaries'].values())/result['wall_seconds']
        result['sum_peak_rss_mb']=sum(s['peak_rss_mb'] for s in result['summaries'].values())
        reports.append(result)
    evaluations={}
    for seed in (42,43):
        checkpoint=root/'workers-2'/f'seed-{seed}'/'latest.json.gz'
        evaluations[str(seed)]=evaluate(Policy('checkpoint',checkpoint),Policy('call'),500,880000,'holdem')
    report=dict(schema=1,config=json.loads(config.read_text()),sweep=reports,development_evaluation=evaluations,
                note='Independent seeds, not a shared strategy speedup. Peak RSS excludes final serialization overhead. Evaluation uses development seeds only.')
    save(root/'report.json',report)
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run-dir',required=True)
    p.add_argument('--iterations',type=int,default=1000)
    args=p.parse_args()
    experiment(args.run_dir,args.iterations)

if __name__=='__main__': main()
