import json
import numpy as np
from open_spiel.python.algorithms import external_sampling_mccfr
from cortado.abstraction import AbstractGame
from cortado.betting import BettingGame, raise_targets, SCHEMES
from cortado.checkpoints import load
from cortado.evaluation import evaluate
from cortado.fullaction import FullActionAgent, pot_fraction
from cortado.game import game
from cortado.policies import Policy
from cortado.pool import PoolPolicy
from cortado import ptrain


def test_targets_are_legal_and_include_allin():
    legal = [0, 1] + list(range(4, 201))
    for street in range(4):
        for raises in range(4):
            targets = raise_targets('sizes-v1', street, raises, (2, 4), 0, legal)
            assert set(targets) <= set(legal) and max(legal) in targets
            if raises >= len(SCHEMES['sizes-v1'][street]):
                assert targets == [max(legal)]


def test_target_fractions_match_scheme():
    # Postflop first bet with 10 chips each in: pot 20, sizes are pot fractions.
    legal = [0, 1] + list(range(12, 201))
    targets = raise_targets('sizes-v1', 1, 0, (10, 10), 1, legal)
    fractions = [round(pot_fraction(t, [10, 10], 1), 2) for t in targets[:-1]]
    assert fractions == [0.35, 0.75, 2.0]  # 0.33 * 20 rounds to 7 chips


def test_betting_state_tracks_streets_and_commitments():
    g = game('holdem', stack_bb=100, betting='sizes-v1')
    s = g.new_initial_state()
    for card in range(4):
        s.apply_action(card)
    assert (s.street, s.commits) == (0, (1, 2))
    s.apply_action(1)         # small blind calls
    s.apply_action(1)         # big blind checks
    for card in (10, 20, 30):
        s.apply_action(card)
    assert (s.street, s.raises, s.commits) == (1, 0, (2, 2))
    bet = s.legal_actions()[2]
    s.apply_action(bet)
    assert s.raises == 1 and s.commits[1] == bet


def test_parallel_one_worker_equals_sequential(tmp_path):
    config = dict(representation='buckets-v1', betting='sizes-v1', stack_bb=100, workers=1, batch=20, seed=5, iterations=20)
    ptrain.train(config, tmp_path)
    parallel = load(tmp_path / 'latest.json.gz', 'holdem')['table']
    np.random.seed(ptrain._seed(5, 0, 0))
    solver = external_sampling_mccfr.ExternalSamplingSolver(AbstractGame(game('holdem', stack_bb=100, betting='sizes-v1'), 'buckets-v1'))
    for _ in range(20):
        solver.iteration()
    assert set(parallel) == set(solver._infostates)
    for key, row in solver._infostates.items():
        assert np.allclose(parallel[key][0], row[0]) and np.allclose(parallel[key][1], row[1])


def test_scheme_agent_is_legal_against_arbitrary_sizes(tmp_path):
    config = dict(representation='buckets-v1', betting='sizes-v1', stack_bb=100, workers=2, batch=30, seed=7, iterations=60)
    ptrain.train(config, tmp_path)
    agent = FullActionAgent(Policy('checkpoint', str(tmp_path / 'latest.json.gz')))
    for style in ('random-size', 'overbet', 'min-raiser'):
        evaluate(agent, PoolPolicy(style), pairs=15, seed=11, name='holdem', full_actions=True)
    assert agent.decisions > 0 and sum(agent.stats.values()) == agent.decisions


def test_history_resolver_plays_legally_and_resolves_off_tree(tmp_path):
    from cortado.rhresolve import HistoryResolvingAgent
    config = dict(representation='buckets-v1', betting='sizes-v1', stack_bb=100, workers=2, batch=30, seed=3, iterations=60)
    ptrain.train(config, tmp_path)
    agent = HistoryResolvingAgent(Policy('checkpoint', str(tmp_path / 'latest.json.gz')), iterations=30)
    evaluate(agent, PoolPolicy('random-size'), pairs=12, seed=21, name='holdem', full_actions=True)
    assert agent.resolve_stats['resolved'] > 0
    assert agent.stats['resolve'] == agent.resolve_stats['resolved']
