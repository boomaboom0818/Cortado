"""Reproducible external sampling CFR; trusted solver owns simulator states."""
import json
import time
import numpy as np
from open_spiel.python.algorithms import external_sampling_mccfr, exploitability
from .game import game, SPEC

def train(name, iterations, seed, path):
    if iterations < 1:
        raise ValueError('Positive iterations required')
    np.random.seed(seed)
    g = game(name)
    solver = external_sampling_mccfr.ExternalSamplingSolver(g)
    start = time.perf_counter()
    for _ in range(iterations):
        solver.iteration()
    report = dict(schema=1, algorithm='external_sampling_mccfr', game=name,
                  engine='open_spiel==2.0.2', specification=SPEC if name=='holdem' else {},
                  iterations=iterations, seed=seed, seconds=time.perf_counter()-start,
                  table={k: [v[0].tolist(), v[1].tolist()] for k,v in solver._infostates.items()})
    if name == 'kuhn':
        report['exact_exploitability'] = exploitability.exploitability(g, solver.average_policy())
    with open(path, 'w') as f:
        json.dump(report, f, sort_keys=True)
    return {k:v for k,v in report.items() if k!='table'} | {'information_sets':len(report['table'])}
