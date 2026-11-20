"""Head-to-head match between two checkpoints in the full no-limit game (paired deals, duplicate seats).

Each pair deals the same cards twice with seats swapped, so card luck largely cancels. Results are from the
first checkpoint's point of view (bb/100); AIVAT (weights 1,1,1,1) uses the first checkpoint's strategy.
"""
import argparse
import json
import time
from multiprocessing import get_context
import numpy as np
from scipy.stats import t
from .evaluation import evaluate
from .fullaction import FullActionAgent
from .policies import Policy


def _block(args):
    first, second, pairs, seed = args
    a, b = (FullActionAgent(Policy('checkpoint', c)) for c in (first, second))
    if a.stack_bb != b.stack_bb:
        raise ValueError('Checkpoints play different stack depths')
    report = evaluate(a, b, pairs, seed, 'holdem', full_actions=True, stack_bb=a.stack_bb, aivat=(1.0, 1.0, 1.0, 1.0))
    return report['clusters'], report['aivat_clusters'], dict(a.stats), dict(b.stats)


def _summary(clusters):
    x = np.asarray(clusters) * 100
    h = float(t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))
    return float(x.mean()), [float(x.mean() - h), float(x.mean() + h)]


def match(first, second, *, pairs, seed, workers, blocks):
    per = pairs // blocks
    jobs = [(first, second, per, seed + block * 1000) for block in range(blocks)]
    start = time.perf_counter()
    with get_context('fork').Pool(workers) as pool:
        results = pool.map(_block, jobs)
    raw = [c for r in results for c in r[0]]
    adjusted = [c for r in results for c in r[1]]
    sources = [{}, {}]
    for r in results:
        for side in (0, 1):
            for k, v in r[2 + side].items():
                sources[side][k] = sources[side].get(k, 0) + v
    mean, ci = _summary(raw)
    amean, aci = _summary(adjusted)
    return dict(first=first, second=second, seed=seed, independent_pairs=len(raw), hands=2 * len(raw),
                mean_bb_per_100=mean, ci95=ci, aivat_mean_bb_per_100=amean, aivat_ci95=aci,
                decision_sources=dict(first=sources[0], second=sources[1]),
                wall_seconds=time.perf_counter() - start, units='bb/100 for the first checkpoint',
                note='Development evaluation; not a pre-registered final test.',
                clusters=raw, aivat_clusters=adjusted)


def main():
    p = argparse.ArgumentParser(description='Head-to-head match between two checkpoints')
    p.add_argument('first')
    p.add_argument('second')
    p.add_argument('--pairs', type=int, default=10000)
    p.add_argument('--seed', type=int, default=850000)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--blocks', type=int, default=8)
    p.add_argument('--out')
    a = p.parse_args()
    report = match(a.first, a.second, pairs=a.pairs, seed=a.seed, workers=a.workers, blocks=a.blocks)
    text = json.dumps(report)
    if a.out:
        open(a.out, 'w').write(text + '\n')
    print(json.dumps({k: v for k, v in report.items() if 'clusters' not in k}, indent=1))


if __name__ == '__main__':
    main()
