import argparse
import json
from .training import train
from .evaluation import evaluate
from .policies import Policy
from .profiles import active_profile

def main():
    p=argparse.ArgumentParser()
    p.add_argument('command',choices=['train','evaluate'])
    p.add_argument('--game',choices=['kuhn','leduc','holdem'],default='kuhn')
    p.add_argument('--iterations',type=int,default=1000)
    p.add_argument('--resume')
    p.add_argument('--representation',choices=['exact-v1','suits-v1','buckets-v1','buckets-v2'])
    p.add_argument('--checkpoint-every',type=int,default=1000)
    p.add_argument('--max-seconds',type=float)
    p.add_argument('--max-rss-mb',type=float)
    p.add_argument('--metrics')
    p.add_argument('--stack-bb',type=int,default=100)
    p.add_argument('--full-actions',action='store_true',help='Evaluation only; integer-chip raise sizes')
    p.add_argument('--pairs',type=int,default=100)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--checkpoint',default='checkpoint.json')
    p.add_argument('--policy',choices=['uniform','call','aggressive','checkpoint'],default='uniform')
    p.add_argument('--opponent',choices=['uniform','call','aggressive','checkpoint'],default='call')
    p.add_argument('--opponent-checkpoint')
    p.add_argument('--mode',choices=['bot','human'],default='bot')
    a=p.parse_args()
    try:
        profile = active_profile(a.mode)
    except ValueError as error:
        p.error(str(error))
    if a.full_actions and a.command == "train":
        p.error("Full-action training is not supported; branching is prohibitive")
    result = train(a.game,a.iterations,a.seed,a.checkpoint,a.resume,a.mode,representation=a.representation,stack_bb=a.stack_bb,checkpoint_every=a.checkpoint_every,max_seconds=a.max_seconds,max_rss_mb=a.max_rss_mb,metrics_path=a.metrics) if a.command=='train' else evaluate(
        Policy(a.policy,a.checkpoint if a.policy=='checkpoint' else None),
        Policy(a.opponent,a.opponent_checkpoint),a.pairs,a.seed,a.game,a.full_actions)
    result['profile'] = profile.metadata()
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
