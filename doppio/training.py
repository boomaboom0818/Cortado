"""Reproducible external sampling CFR; trusted solver owns simulator states."""
import json
import os
from pathlib import Path
import tempfile
import time
import numpy as np
from open_spiel.python.algorithms import external_sampling_mccfr, exploitability
from .game import game, SPEC
from .checkpoints import load

def train(name, iterations, seed, path, resume=None):
    if iterations < 1:
        raise ValueError('Positive iterations required')
    np.random.seed(seed)
    g = game(name)
    solver = external_sampling_mccfr.ExternalSamplingSolver(g)
    previous = load(resume, name) if resume else None
    if previous:
        if previous['schema'] != 2:
            raise ValueError('Legacy snapshot cannot resume exactly')
        solver._infostates = {k:[np.array(x,dtype=float) for x in row] for k,row in previous['table'].items()}
        r = previous['rng']
        np.random.set_state((r[0], np.array(r[1],dtype=np.uint32), r[2], r[3], r[4]))
        seed = previous['seed']
    start = time.perf_counter()
    for _ in range(iterations):
        solver.iteration()
    report = dict(schema=2, algorithm='external_sampling_mccfr', game=name,
                  engine='open_spiel==2.0.2', specification=SPEC if name=='holdem' else {},
                  iterations=iterations + (previous['iterations'] if previous else 0), seed=seed, seconds=time.perf_counter()-start,
                  table={k: [v[0].tolist(), v[1].tolist()] for k,v in solver._infostates.items()})
    rng = np.random.get_state()
    report['rng'] = [rng[0],rng[1].tolist(),rng[2],rng[3],rng[4]]
    report['seconds'] += previous['seconds'] if previous else 0
    if name == 'kuhn':
        report['exact_exploitability'] = exploitability.exploitability(g, solver.average_policy())
    target = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', dir=target.parent, delete=False) as f:
        temporary = f.name
        json.dump(report, f, sort_keys=True)
    os.replace(temporary, target)
    return {k:v for k,v in report.items() if k not in ('table', 'rng')} | {'information_sets':len(report['table'])}
