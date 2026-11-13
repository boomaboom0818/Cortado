"""Independent duplicate-deal clusters, seats swapped inside each cluster."""
import time
import numpy as np
from scipy.stats import t
from .game import game, observe

def evaluate(a, b, pairs=100, seed=100000, name='holdem', full_actions=False):
    if pairs < 2:
        raise ValueError('At least two independent pairs required')
    for policy in (a,b):
        if policy.data and policy.data['game'] != name:
            raise ValueError('Checkpoint game mismatch')
    if full_actions and any(p.data for p in (a,b)):
        raise ValueError("FCPA checkpoints cannot play full-action games")
    g = game(name, full_actions=full_actions)
    clusters, latencies = [], []
    start = time.perf_counter()
    for pair in range(pairs):
        scores = []
        for seat in (0,1):
            chance = np.random.default_rng(np.random.SeedSequence([seed,pair,0]))
            action_rng = np.random.default_rng(np.random.SeedSequence([seed,pair,seat,1]))
            s = g.new_initial_state()
            while not s.is_terminal():
                if s.is_chance_node():
                    choices, probs = zip(*s.chance_outcomes())
                    s.apply_action(int(chance.choice(choices,p=probs)))
                else:
                    policy = a if s.current_player()==seat else b
                    obs = observe(s)
                    tick = time.perf_counter()
                    action = policy.act(obs, action_rng)
                    latencies.append(time.perf_counter()-tick)
                    if action not in obs.legal_actions:
                        raise ValueError('Illegal policy action')
                    s.apply_action(action)
            scores.append(s.returns()[seat]/(2 if name=='holdem' else 1))
        clusters.append(np.mean(scores))
    mean = float(np.mean(clusters)*100)
    half = float(t.ppf(.975,pairs-1)*np.std(clusters,ddof=1)/np.sqrt(pairs)*100)
    return dict(game=name, seed=seed, hands=2*pairs, independent_pairs=pairs,
                units='bb/100' if name=='holdem' else 'chips/100', mean=mean,
                ci95=[mean-half,mean+half], latency_ms_p50=float(np.median(latencies)*1000),
                latency_ms_p95=float(np.quantile(latencies,.95)*1000),
                wall_seconds=time.perf_counter()-start, decisions=[a.decisions,b.decisions], checkpoint_misses=[a.misses,b.misses],
                cost='Local CPU; monetary cost not metered')
