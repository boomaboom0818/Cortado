import argparse
import json
from .training import train
from .evaluation import evaluate
from .policies import Policy

def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['train','evaluate'])
    p.add_argument('--game',choices=['kuhn','holdem'],default='kuhn')
    p.add_argument('--iterations',type=int,default=1000)
    p.add_argument('--pairs',type=int,default=100)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--checkpoint',default='checkpoint.json')
    p.add_argument('--policy',choices=['uniform','call','aggressive','checkpoint'],default='uniform')
    p.add_argument('--opponent',choices=['uniform','call','aggressive','checkpoint'],default='call')
    p.add_argument('--opponent-checkpoint')
    p.add_argument('--mode',choices=['bot','human'],default='bot')
    a=p.parse_args()
    result = train(a.game,a.iterations,a.seed,a.checkpoint) if a.command=='train' else evaluate(
        Policy(a.policy,a.checkpoint if a.policy=='checkpoint' else None),
        Policy(a.opponent,a.opponent_checkpoint),a.pairs,a.seed,a.game)
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
