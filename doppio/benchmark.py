"""Frozen held-out controls plus a historical checkpoint, with separate seed blocks."""
import argparse
import json
from .evaluation import evaluate
from .policies import Policy

def main():
    p=argparse.ArgumentParser()
    p.add_argument('checkpoint')
    p.add_argument('--historical',required=True)
    p.add_argument('--pairs',type=int,default=10000)
    p.add_argument('--seed',type=int,default=100000)
    args=p.parse_args()
    reports={}
    for index,kind in enumerate(('call','aggressive','uniform','checkpoint')):
        reports[kind]=evaluate(Policy('checkpoint',args.checkpoint),
            Policy(kind,args.historical if kind=='checkpoint' else None),
            args.pairs,args.seed+index*100000,'holdem')
    print(json.dumps(reports,indent=2))
if __name__=='__main__': main()
