"""Full-action benchmark: a candidate checkpoint (via the adapter) against the frozen pool and LBR.

Blocks of independent paired deals run in separate processes with disjoint seeds.
Each cluster is one deal played from both seats (paired); intervals are Student-t
over clusters. Opponents are reported individually, uncorrected for multiplicity.
"""
import argparse
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from scipy.stats import t
from .evaluation import evaluate
from .fullaction import FullActionAgent
from .lbr import LBR
from .policies import Policy
from .pool import STYLES, VERSION

OPPONENTS = tuple(STYLES) + ('lbr',)


def _task(args):
    checkpoint, opponent, pairs, seed, lbr_hands, fallback = args
    start = time.process_time()
    base = Policy('checkpoint', checkpoint)
    agent = FullActionAgent(base, fallback=fallback)
    if opponent == 'lbr':
        report = evaluate(agent, LBR(agent, hands=lbr_hands), pairs, seed, 'holdem', full_actions=True)
    else:
        from .pool import PoolPolicy
        report = evaluate(agent, PoolPolicy(opponent), pairs, seed, 'holdem', full_actions=True)
    report['source_counts'] = dict(agent.stats)
    report['candidate_latencies_ms'] = [x * 1000 for x in agent.latencies]
    report['cpu_seconds'] = time.process_time() - start
    return opponent, report


def _combine(reports):
    clusters = np.array([c for r in reports for c in r['clusters']])
    n = len(clusters)
    mean = float(clusters.mean() * 100)
    half = float(t.ppf(.975, n - 1) * clusters.std(ddof=1) / np.sqrt(n) * 100)
    sources = Counter()
    latencies = [x for r in reports for x in r['candidate_latencies_ms']]
    for r in reports:
        sources.update(r['source_counts'])
    decisions = sum(sources.values())
    return dict(independent_pairs=n, hands=2 * n, mean_bb_per_100=mean, ci95=[mean - half, mean + half],
                candidate_latency_ms_p50=float(np.quantile(latencies, .5)),
                candidate_latency_ms_p95=float(np.quantile(latencies, .95)),
                candidate_latency_ms_max=float(max(latencies)),
                cpu_seconds=sum(r['cpu_seconds'] for r in reports),
                candidate_decision_sources={k: v / decisions for k, v in sources.items()} if decisions else {},
                candidate_decisions=decisions)


def run(checkpoint, *, pairs, seed, workers, blocks, lbr_pairs, lbr_hands, fallback='heuristic', opponents=OPPONENTS):
    jobs = []
    for index, opponent in enumerate(opponents):
        total = lbr_pairs if opponent == 'lbr' else pairs
        per = max(2, total // blocks)
        for block in range(blocks):
            jobs.append((checkpoint, opponent, per, seed + index * 1_000_000 + block * 1000, lbr_hands, fallback))
    start = time.perf_counter()
    grouped = {}
    with ProcessPoolExecutor(workers) as pool:
        for opponent, report in pool.map(_task, jobs):
            grouped.setdefault(opponent, []).append(report)
    return dict(schema=1, pool_version=VERSION, checkpoint=str(checkpoint), seed=seed, fallback=fallback,
                lbr_hands_sampled=lbr_hands, wall_seconds=time.perf_counter() - start,
                units='bb/100, positive favours the candidate',
                opponents={o: _combine(r) for o, r in grouped.items()},
                note='Development evaluation unless the seed block was pre-registered as final; '
                     'intervals are per-opponent and not multiplicity-corrected.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint')
    p.add_argument('--pairs', type=int, default=1000)
    p.add_argument('--seed', type=int, default=700000)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--blocks', type=int, default=6)
    p.add_argument('--lbr-pairs', type=int, default=120)
    p.add_argument('--lbr-hands', type=int, default=60)
    p.add_argument('--fallback', choices=['heuristic', 'uniform'], default='heuristic')
    p.add_argument('--opponents', nargs='+', default=list(OPPONENTS))
    p.add_argument('--out')
    args = p.parse_args()
    report = run(args.checkpoint, pairs=args.pairs, seed=args.seed, workers=args.workers, blocks=args.blocks,
                 lbr_pairs=args.lbr_pairs, lbr_hands=args.lbr_hands, fallback=args.fallback, opponents=args.opponents)
    text = json.dumps(report, indent=2)
    if args.out:
        open(args.out, 'w').write(text + '\n')
    print(text)


if __name__ == '__main__':
    main()
