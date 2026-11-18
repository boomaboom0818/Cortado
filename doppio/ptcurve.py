"""Learning-curve step for a running parallel-training service (run repeatedly, e.g. hourly).

If the run's latest checkpoint has advanced by at least `--step` iterations since the last curve
point, copy it to an immutable `iteration-N.json.gz`, benchmark it on the fixed development deals
(seed 700000, six strategic pool bots plus LBR, AIVAT) and append a curve entry with the paired
change in composite versus the previous point. Stopping rule, fixed before the runs: stop the
training service after two consecutive improvements below `--flat` bb/100.
"""
import argparse
import json
import shutil
import subprocess
from pathlib import Path
import numpy as np
from scipy.stats import t
from .checkpoints import load
from .curve import STRATEGIC, composite
from .fullbench import run as bench


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-dir', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--service', required=True)
    p.add_argument('--step', type=int, default=2_000_000)
    p.add_argument('--flat', type=float, default=10.0)
    p.add_argument('--pairs', type=int, default=3000)
    p.add_argument('--workers', type=int, default=2)
    a = p.parse_args()
    run, out = Path(a.run_dir), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    curve_path = out / 'curve.jsonl'
    curve = [json.loads(l) for l in curve_path.read_text().splitlines()] if curve_path.exists() else []
    if curve and curve[-1].get('stop'):
        print('already stopped'); return
    metrics = [json.loads(l) for l in (run / 'metrics.jsonl').read_text().splitlines()] if (run / 'metrics.jsonl').exists() else []
    if not metrics:
        print('no checkpoint yet'); return
    iterations = metrics[-1]['iterations']
    if curve and iterations - curve[-1]['iterations'] < a.step:
        print(f'{iterations} iterations; next point at {curve[-1]["iterations"] + a.step}'); return
    snapshot = run / f'iteration-{iterations}.json.gz'
    if not snapshot.exists():
        shutil.copy2(run / 'latest.json.gz', snapshot)
    if load(snapshot, 'holdem')['iterations'] != iterations:
        print('latest checkpoint changed while copying; retry next time'); snapshot.unlink(); return
    report = bench(str(snapshot), pairs=a.pairs, seed=700000, workers=a.workers, blocks=a.workers * 2,
                   lbr_pairs=300, lbr_hands=60, opponents=STRATEGIC + ('lbr',), aivat=(1.0, 1.0, 1.0, 1.0))
    (out / f'bench-{iterations}.json').write_text(json.dumps(report) + '\n')
    score = composite(report)
    entry = dict(iterations=iterations, composite=float(score.mean()),
                 per_opponent={o: v['aivat_mean_bb_per_100'] for o, v in report['opponents'].items()},
                 lbr_ci95=report['opponents']['lbr']['aivat_ci95'],
                 table_fraction={o: v['candidate_decision_sources'].get('table', 0) / max(1, sum(v['candidate_decision_sources'].values()))
                                 for o, v in report['opponents'].items()})
    if curve:
        previous = composite(json.loads((out / f'bench-{curve[-1]["iterations"]}.json').read_text()))
        diff = score - previous
        h = float(t.ppf(.975, len(diff) - 1) * diff.std(ddof=1) / np.sqrt(len(diff)))
        entry.update(improvement=float(diff.mean()), improvement_ci95=[float(diff.mean() - h), float(diff.mean() + h)])
    recent = [c.get('improvement') for c in curve[-1:]] + [entry.get('improvement')]
    if len(curve) >= 2 and all(x is not None and x < a.flat for x in recent):
        entry['stop'] = f'flat: two consecutive improvements < {a.flat} bb/100'
        subprocess.run(['systemctl', '--user', 'kill', '--signal=TERM', a.service], check=False)  # trainer saves, then exits
    with curve_path.open('a') as stream:
        stream.write(json.dumps(entry) + '\n')
    print(json.dumps(entry))


if __name__ == '__main__':
    main()
