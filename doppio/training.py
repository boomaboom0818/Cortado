"""Single-process reproducible MCCFR with boundary-safe interruption and budgets."""
import json
import platform
import resource
import signal
import sys
import time
from pathlib import Path
import numpy as np
from open_spiel.python.algorithms import external_sampling_mccfr, exploitability
from .game import game, SPEC
from .checkpoints import load, save, ENGINE
from .profiles import active_profile
from .abstraction import AbstractGame, REPRESENTATIONS


def train(name, iterations, seed, path, resume=None, target='bot', *,
          representation=None, checkpoint_every=1000, max_seconds=None,
          max_rss_mb=None, metrics_path=None):
    profile = active_profile(target)
    if iterations < 1 or checkpoint_every < 1:
        raise ValueError('Iterations and checkpoint interval must be positive')
    if any(v is not None and v <= 0 for v in (max_seconds, max_rss_mb)):
        raise ValueError('Resource budgets must be positive')
    previous = load(resume, name) if resume else None
    inherited = previous.get('representation', 'exact-v1') if previous else 'exact-v1'
    representation = representation or inherited
    if representation not in REPRESENTATIONS or (name in ('kuhn','leduc') and representation!='exact-v1'):
        raise ValueError('Invalid representation for game')
    if previous and representation != inherited:
        raise ValueError('Cannot change representation on resume')
    if previous and (previous['schema'] not in (2,3) or previous.get('profile',{}).get('target','bot') != target):
        raise ValueError('Incompatible resume checkpoint')
    if previous and previous['schema'] == 3:
        runtime = previous.get('runtime', {})
        if runtime.get('numpy') != np.__version__ or runtime.get('python') != platform.python_version():
            raise ValueError('Exact resume requires the checkpoint Python and NumPy versions')
    raw_game = game(name)
    g = AbstractGame(raw_game,representation) if representation!='exact-v1' else raw_game
    solver = external_sampling_mccfr.ExternalSamplingSolver(g)
    original_rng = np.random.get_state()
    np.random.seed(seed)
    if previous:
        solver._infostates = {k:[np.array(x,dtype=float) for x in row] for k,row in previous['table'].items()}
        r = previous['rng']
        np.random.set_state((r[0],np.array(r[1],dtype=np.uint32),r[2],r[3],r[4]))
        seed = previous['seed']
    completed = previous['iterations'] if previous else 0
    base_seconds = previous['seconds'] if previous else 0
    start = time.perf_counter()
    stop = []
    handlers = {}
    def request_stop(signum, frame):
        stop.append(signal.Signals(signum).name)
    # Signals only set a flag: finish both traversal passes before saving.
    import threading
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGINT, signal.SIGTERM):
            handlers[signum] = signal.signal(signum, request_stop)
    def snapshot(reason):
        rng = np.random.get_state()
        report = dict(schema=3, algorithm='external_sampling_mccfr',game=name,
                      engine=ENGINE,specification=SPEC if name=='holdem' else {},
                      representation=representation,profile=profile.metadata(),
                      iterations=completed,seed=seed,seconds=base_seconds+time.perf_counter()-start,
                      stop_reason=reason, peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                      runtime={'python':platform.python_version(),'numpy':np.__version__,'platform':platform.platform()},
                      rng=[rng[0],rng[1].tolist(),rng[2],rng[3],rng[4]],
                      table={k:[x.tolist() for x in v] for k,v in solver._infostates.items()})
        if name in ('kuhn','leduc'):
            report['exact_exploitability']=exploitability.exploitability(raw_game,solver.average_policy())
        save(path,report)
        summary={k:v for k,v in report.items() if k not in ('table','rng')}
        summary['information_sets']=len(report['table'])
        summary['checkpoint_bytes']=Path(path).stat().st_size
        if metrics_path:
            Path(metrics_path).parent.mkdir(parents=True,exist_ok=True)
            with open(metrics_path,'a') as stream:
                stream.write(json.dumps(summary)+'\n')
        print(f"Saved {completed} iterations ({reason}) to {path}",file=sys.stderr)
        return summary
    try:
        for index in range(iterations):
            solver.iteration()
            completed += 1
            reason = stop[-1] if stop else None
            if max_seconds and time.perf_counter()-start >= max_seconds:
                reason = reason or 'time_budget'
            if max_rss_mb and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024 >= max_rss_mb:
                reason = reason or 'memory_budget'
            if reason or index == iterations-1:
                return snapshot(reason or 'completed')
            if completed % checkpoint_every == 0:
                snapshot('periodic')
    finally:
        np.random.set_state(original_rng)
        for signum,handler in handlers.items():
            signal.signal(signum,handler)
