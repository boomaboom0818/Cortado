from itertools import combinations
import numpy as np
from doppio.cards import DECK
from doppio.evaluation import evaluate
from doppio.policies import Policy
from doppio.pool import PoolPolicy
from doppio.resolve import ResolvingAgent, RiverSolver, build_tree
from doppio.training import train

BOARD = ('Ah', 'Kd', '7c', '4s', '2h')
HANDS = [h for h in combinations(DECK, 2) if not set(h) & set(BOARD)]


def test_fast_showdown_matches_dense_with_card_removal():
    rng = np.random.default_rng(1)
    s = RiverSolver(BOARD, [HANDS[:250], HANDS[150:400]], [rng.random(250), rng.random(250)], (10, 10), 200, iterations=1)
    q = rng.random(250)
    d0, m0 = s._fast[0].evaluate(q)
    assert np.allclose(d0, (s.result * s.valid) @ q) and np.allclose(m0, s.valid @ q)
    d1, m1 = s._fast[1].evaluate(q)
    assert np.allclose(d1, (-s.result * s.valid).T @ q) and np.allclose(m1, s.valid.T @ q)


def test_cfr_plus_converges():
    rng = np.random.default_rng(2)
    args = (BOARD, [HANDS[:120], HANDS[60:180]], [rng.random(120), rng.random(120)], (20, 20), 200)
    early = RiverSolver(*args, iterations=20).exploitability()
    late = RiverSolver(*args, iterations=400).exploitability()
    assert late < early / 5 and late < 0.01 * 40   # under 1% of the pot


def test_tree_contains_off_tree_forced_action():
    tree = build_tree((20, 20), 200, 1, forced=(1, 57))
    child = tree.children[tree.actions.index(1)]
    assert 57 in child.actions


def test_resolving_agent_plays_legally(tmp_path):
    path = tmp_path / 'tiny.json.gz'
    train('holdem', 150, 5, path, representation='buckets-v1')
    agent = ResolvingAgent(Policy('checkpoint', path), iterations=30)
    for style in ('overbet', 'station'):
        evaluate(agent, PoolPolicy(style, samples=16), 3, 11, 'holdem', full_actions=True)
    assert agent.stats['resolve'] > 0
