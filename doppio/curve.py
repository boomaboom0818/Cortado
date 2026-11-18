"""Train-until-flat driver: repeated training cycles with a fixed paired benchmark after each.

Stopping rule (fixed before the run): the composite score is the mean AIVAT-corrected bb/100 over
STRATEGIC opponents on the same development deals. Stop when two consecutive cycles each improve the
composite by less than `flat_bb` bb/100 (paired point estimate), or after `max_cycles`, or on any
training resource stop. LBR is reported but not in the composite (its sample is small).
"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.stats import t
from .fullbench import run as bench
from .run import run as train_run

STRATEGIC = ('nit', 'tag', 'min-raiser', 'third-pot', 'random-size', 'overbet')


def composite(report):
    return np.mean([report['opponents'][o]['aivat_clusters'] for o in STRATEGIC], axis=0) * 100


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--invocations', type=int, default=3)
    p.add_argument('--max-cycles', type=int, default=12)
    p.add_argument('--flat-bb', type=float, default=10.0)
    p.add_argument('--pairs', type=int, default=3000)
    p.add_argument('--lbr-pairs', type=int, default=300)
    p.add_argument('--workers', type=int, default=5)
    p.add_argument('--seed', type=int, default=700000)
    a = p.parse_args()
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    curve_path = out / 'curve.jsonl'
    curve = [json.loads(l) for l in curve_path.read_text().splitlines()] if curve_path.exists() else []

    def evaluate(iterations):
        path = Path(a.run_dir) / f'iteration-{iterations}.json.gz'
        report_path = out / f'bench-{iterations}.json'
        if not report_path.exists():
            report = bench(str(path), pairs=a.pairs, seed=a.seed, workers=a.workers, blocks=a.workers * 2,
                           lbr_pairs=a.lbr_pairs, lbr_hands=60, opponents=STRATEGIC + ('lbr',), aivat=(1.0, 1.0, 1.0, 1.0))
            report_path.write_text(json.dumps(report) + '\n')
        return json.loads(report_path.read_text())

    def record(iterations, stop_reason):
        report = evaluate(iterations)
        score = composite(report)
        entry = dict(iterations=iterations, stop_reason=stop_reason, composite=float(score.mean()),
                     per_opponent={o: v['aivat_mean_bb_per_100'] for o, v in report['opponents'].items()},
                     lbr_ci95=report['opponents']['lbr']['aivat_ci95'])
        if curve:
            prev = composite(evaluate(curve[-1]['iterations']))
            diff = score - prev
            h = float(t.ppf(.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff)))
            entry.update(improvement=float(diff.mean()), improvement_ci95=[float(diff.mean() - h), float(diff.mean() + h)])
        curve.append(entry)
        with curve_path.open('a') as stream:
            stream.write(json.dumps(entry) + '\n')
        print(json.dumps(entry), flush=True)

    if not curve:
        summary = json.loads((Path(a.run_dir) / 'summary.json').read_text())
        record(summary['iterations'], 'baseline')
    while len(curve) <= a.max_cycles:
        recent = [c.get('improvement') for c in curve[-2:]]
        if len(curve) >= 3 and all(x is not None and x < a.flat_bb for x in recent):
            print(json.dumps(dict(stop='flat', rule=f'two consecutive improvements < {a.flat_bb} bb/100')))
            return
        reason = None
        for _ in range(a.invocations):
            result = train_run(a.config, a.run_dir, resume=True)
            reason = result['stop_reason']
            if reason not in ('time_budget', 'completed'):
                break
        record(result['iterations'], reason)
        if reason not in ('time_budget', 'completed'):
            print(json.dumps(dict(stop='resource', reason=reason)))
            return
    print(json.dumps(dict(stop='max_cycles')))


if __name__ == '__main__':
    main()
