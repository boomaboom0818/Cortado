"""Evaluate completed independent candidates using a frozen development pool."""
import argparse
from pathlib import Path
from .checkpoints import save
from .evaluation import evaluate
from .policies import Policy


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',required=True)
    parser.add_argument('--historical',required=True)
    parser.add_argument('--pairs',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=880000)
    args=parser.parse_args()
    root=Path(args.run_dir)
    checkpoints=sorted([*root.glob('seed-*/latest.json.gz'), *root.glob('seed-*/latest.sqlite')])
    if not checkpoints:
        parser.error('No worker checkpoints found')
    reports={}
    for checkpoint in checkpoints:
        matches={}
        for index,kind in enumerate(('call','aggressive','uniform','checkpoint')):
            matches[kind]=evaluate(Policy('checkpoint',checkpoint),
                Policy(kind,args.historical if kind=='checkpoint' else None),
                args.pairs,args.seed+index*100000,'holdem')
        reports[checkpoint.parent.name]=matches
    save(root/'development-evaluation.json',dict(schema=1,seed=args.seed,pairs=args.pairs,
         historical=str(Path(args.historical).resolve()),candidates=reports,
         purpose='Development only; these seeds/opponents are not untouched final tests'))

if __name__=='__main__': main()
