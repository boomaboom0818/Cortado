"""Paired comparison of two fullbench reports run on identical deals (same seed, pairs, blocks)."""
import argparse
import json
import numpy as np
from scipy.stats import t


def main():
    p = argparse.ArgumentParser()
    p.add_argument('baseline')
    p.add_argument('candidate')
    p.add_argument('--field', default='aivat_clusters')
    a = p.parse_args()
    base, cand = (json.load(open(x)) for x in (a.baseline, a.candidate))
    if base['seed'] != cand['seed']:
        raise SystemExit('Reports must use the same seed block')
    out = {}
    for opponent in cand['opponents']:
        x = np.array(cand['opponents'][opponent][a.field]); y = np.array(base['opponents'][opponent][a.field])
        if len(x) != len(y):
            raise SystemExit(f'{opponent}: different pair counts')
        d = (x - y) * 100
        h = float(t.ppf(.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d)))
        out[opponent] = dict(pairs=len(d), improvement_bb_per_100=float(d.mean()), ci95=[float(d.mean() - h), float(d.mean() + h)])
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
