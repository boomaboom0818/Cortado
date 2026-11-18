"""Parallel external-sampling MCCFR: forked workers, additive merges of regret and strategy sums.

Each round, `workers` processes fork from the master table (copy-on-write) and run `batch`
iterations each with independent, deterministic seeds (base seed, round, worker). A worker
records every row it touches in a private overlay and returns only the change in regret and
average-strategy sums. External-sampling updates are additive, so the master adds all workers'
changes. Within a round workers do not see each other's updates (bounded staleness, as in other
parallel CFR implementations); with workers=1 this is exactly sequential MCCFR.

Checkpoints use the ordinary schema-3 format (`algorithm` external_sampling_mccfr) so `Policy`,
`FullActionAgent` and all evaluation tools load them unchanged; a `parallel` field records the
scheme. Resume continues from the last completed round and is deterministic given the config.
"""
import argparse
import json
import os
import platform
import signal
import sys
import time
from multiprocessing import get_context
from pathlib import Path
import numpy as np
from open_spiel.python.algorithms import external_sampling_mccfr
from .abstraction import AbstractGame
from .checkpoints import ENGINE, load, save
from .game import game, spec
from .profiles import active_profile
from .training import _digest

_TABLE = None      # master table inherited by forked workers
_GAME = None


class Overlay(dict):
    """Solver table view: reads fall through to the master table; first access copies the row."""

    def __init__(self, base):
        super().__init__()
        self.base = base

    def get(self, key, default=None):
        row = dict.get(self, key)
        if row is None:
            original = self.base.get(key)
            if original is None:
                return default
            row = [original[0].copy(), original[1].copy()]
            dict.__setitem__(self, key, row)
        return row

    def __getitem__(self, key):
        row = self.get(key)
        if row is None:
            raise KeyError(key)
        return row


def _worker(args):
    seed, batch = args
    np.random.seed(seed)
    solver = external_sampling_mccfr.ExternalSamplingSolver(_GAME)
    overlay = Overlay(_TABLE)
    solver._infostates = overlay
    for _ in range(batch):
        solver.iteration()
    delta = {}
    for key, row in dict.items(overlay):
        original = _TABLE.get(key)
        if original is None:
            delta[key] = (row[0], row[1])
        else:
            delta[key] = (row[0] - original[0], row[1] - original[1])
    return delta


def _seed(base, round_index, worker):
    return int(np.random.SeedSequence([base, round_index, worker]).generate_state(1)[0])


def train(config, directory, resume=False):
    global _TABLE, _GAME
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest, checkpoint = directory / 'config.json', directory / 'latest.json.gz'
    if resume:
        if json.loads(manifest.read_text()) != config:
            raise ValueError('Resume configuration differs from the run manifest')
    elif manifest.exists():
        raise ValueError('Run directory already initialized; use --resume')
    else:
        manifest.write_text(json.dumps(config, indent=1))
    representation, betting, stack_bb = config['representation'], config['betting'], config['stack_bb']
    workers, batch = config['workers'], config['batch']
    _GAME = AbstractGame(game('holdem', stack_bb=stack_bb, betting=betting), representation)
    profile = active_profile('bot')
    digest = _digest(representation)
    if resume:
        previous = load(checkpoint, 'holdem')
        if (previous.get('representation'), previous.get('betting'), previous.get('stack_bb')) != (representation, betting, stack_bb):
            raise ValueError('Checkpoint abstraction differs from config')
        if previous.get('abstraction_digest') != digest:
            raise ValueError('Bucket tables changed since the checkpoint was written')
        _TABLE = {k: [np.array(r[0], dtype=float), np.array(r[1], dtype=float)] for k, r in previous.pop('table').items()}
        round_index, iterations, seconds = previous['parallel']['rounds'], previous['iterations'], previous['seconds']
    else:
        _TABLE, round_index, iterations, seconds = {}, 0, 0, 0.0
    stop = []
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, lambda s, f: stop.append(signal.Signals(s).name))
    start, last_save = time.perf_counter(), time.perf_counter()
    context = get_context('fork')

    def snapshot(reason):
        report = dict(schema=3, algorithm='external_sampling_mccfr', game='holdem', engine=ENGINE,
                      specification=spec(stack_bb, betting), stack_bb=stack_bb, betting=betting,
                      representation=representation, abstraction_digest=digest, profile=profile.metadata(),
                      storage='memory', iterations=iterations, seed=config['seed'],
                      seconds=seconds + time.perf_counter() - start, stop_reason=reason,
                      parallel=dict(workers=workers, batch=batch, rounds=round_index),
                      runtime={'python': platform.python_version(), 'numpy': np.__version__},
                      table=_TABLE)
        save(checkpoint, report)
        summary = {k: v for k, v in report.items() if k != 'table'}
        summary['information_sets'] = len(_TABLE)
        with open(directory / 'metrics.jsonl', 'a') as stream:
            stream.write(json.dumps(summary) + '\n')
        print(f'{time.strftime("%H:%M:%S")} saved {iterations} iterations, {len(_TABLE)} infosets ({reason})', flush=True)
        return summary

    while True:
        jobs = [(_seed(config['seed'], round_index, w), batch) for w in range(workers)]
        with context.Pool(workers) as pool:
            deltas = pool.map(_worker, jobs)
        for delta in deltas:
            for key, (regret, average) in delta.items():
                row = _TABLE.get(key)
                if row is None:
                    _TABLE[key] = [regret, average]
                else:
                    row[0] += regret
                    row[1] += average
        del deltas
        round_index += 1
        iterations += workers * batch
        elapsed = time.perf_counter() - start
        if round_index % 10 == 0 or round_index <= 3:
            print(f'{time.strftime("%H:%M:%S")} round {round_index}: {iterations} iterations, {len(_TABLE)} infosets, '
                  f'{iterations / max(elapsed + seconds, 1e-9) if not resume else (workers * batch * round_index) / max(elapsed + seconds, 1e-9):.0f} it/s', flush=True)
        reason = stop[-1] if stop else None
        if config.get('max_seconds') and elapsed >= config['max_seconds']:
            reason = reason or 'time_budget'
        if iterations >= config['iterations']:
            reason = reason or 'completed'
        if reason:
            return snapshot(reason)
        if time.perf_counter() - last_save >= config.get('save_seconds', 1800):
            snapshot('periodic')
            last_save = time.perf_counter()


def main():
    p = argparse.ArgumentParser(description='Parallel MCCFR training run')
    p.add_argument('--config', required=True)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args()
    config = json.loads(Path(a.config).read_text())
    required = {'representation', 'betting', 'stack_bb', 'workers', 'batch', 'seed', 'iterations'}
    if not required <= set(config):
        p.error(f'config needs {sorted(required)}')
    print(json.dumps(train(config, a.run_dir, a.resume), indent=1))


if __name__ == '__main__':
    main()
