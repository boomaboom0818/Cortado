"""Regenerate versioned equity-bucket edges from seeded random deals (deterministic)."""
import json
import numpy as np
from .cards import DECK
from .buckets import EDGE_FILE, _strength

def main(buckets=30, samples=8000, seed=20260101):
    rng = np.random.default_rng(seed)
    edges = {}
    for n in (3, 4, 5):
        values = []
        for _ in range(samples):
            d = [DECK[j] for j in rng.permutation(52)[:2 + n]]
            values.append(_strength(tuple(sorted(d[:2])), tuple(sorted(d[2:]))))
        edges[str(n)] = [float(x) for x in np.quantile(values, np.arange(1, buckets) / buckets)]
    EDGE_FILE.write_text(json.dumps(dict(version='buckets-v1', buckets=buckets, seed=seed,
        samples_per_street=samples, edges=edges), indent=1) + '\n')

if __name__ == '__main__':
    main()
