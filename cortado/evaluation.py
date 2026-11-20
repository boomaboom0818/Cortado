"""Independent duplicate-deal clusters, seats swapped inside each cluster."""
import time
import numpy as np
from scipy.stats import t
from .game import game, observe

def evaluate(a, b, pairs=100, seed=100000, name='holdem', full_actions=False, stack_bb=100, aivat=None):
    """`aivat`: None, or a tuple of street weights to also report an AIVAT-corrected estimate (policy `a`)."""
    if pairs < 2:
        raise ValueError('At least two independent pairs required')
    for policy in (a,b):
        if policy.data and policy.data['game'] != name:
            raise ValueError('Checkpoint game mismatch')
    if full_actions and any(p.data for p in (a,b)):
        raise ValueError("FCPA checkpoints cannot play full-action games")
    g = game(name, full_actions=full_actions, stack_bb=stack_bb) if name == 'holdem' else game(name)
    clusters, latencies, adjusted = [], [], []
    if aivat is not None:
        from .aivat import Recorder, correction
    start = time.perf_counter()
    for pair in range(pairs):
        scores, corrected = [], []
        for seat in (0,1):
            chance = np.random.default_rng(np.random.SeedSequence([seed,pair,0]))
            action_rng = np.random.default_rng(np.random.SeedSequence([seed,pair,seat,1]))
            s = g.new_initial_state()
            recorder = Recorder() if aivat is not None else None
            while not s.is_terminal():
                if s.is_chance_node():
                    choices, probs = zip(*s.chance_outcomes())
                    if recorder: recorder.before_chance(s)
                    s.apply_action(int(chance.choice(choices,p=probs)))
                    if recorder: recorder.after_step(s)
                else:
                    policy = a if s.current_player()==seat else b
                    obs = observe(s)
                    tick = time.perf_counter()
                    action = policy.act(obs, action_rng)
                    latencies.append(time.perf_counter()-tick)
                    if action not in obs.legal_actions:
                        raise ValueError('Illegal policy action')
                    if recorder and policy is a:
                        recorder.our_action(s, obs, action)
                    s.apply_action(action)
            scores.append(s.returns()[seat]/(2 if name=='holdem' else 1))
            if recorder:
                rng = np.random.default_rng(np.random.SeedSequence([seed,pair,seat,2]))
                corrected.append(scores[-1] - correction(recorder, seat, a, rng, aivat)/2)
        clusters.append(np.mean(scores))
        if corrected:
            adjusted.append(np.mean(corrected))
    mean = float(np.mean(clusters)*100)
    half = float(t.ppf(.975,pairs-1)*np.std(clusters,ddof=1)/np.sqrt(pairs)*100)
    extra = {}
    if adjusted:
        m = float(np.mean(adjusted)*100)
        h = float(t.ppf(.975,pairs-1)*np.std(adjusted,ddof=1)/np.sqrt(pairs)*100)
        extra = dict(aivat_mean=m, aivat_ci95=[m-h,m+h], aivat_weights=list(aivat),
                     aivat_variance_ratio=float(np.var(adjusted,ddof=1)/np.var(clusters,ddof=1)),
                     aivat_clusters=[float(c) for c in adjusted])
    return dict(**extra, game=name, seed=seed, hands=2*pairs, independent_pairs=pairs,
                units='bb/100' if name=='holdem' else 'chips/100', mean=mean,
                ci95=[mean-half,mean+half], latency_ms_p50=float(np.median(latencies)*1000),
                latency_ms_p95=float(np.quantile(latencies,.95)*1000),
                wall_seconds=time.perf_counter()-start, decisions=[a.decisions,b.decisions], checkpoint_misses=[a.misses,b.misses],
                cost='Local CPU; monetary cost not metered', clusters=[float(c) for c in clusters])
