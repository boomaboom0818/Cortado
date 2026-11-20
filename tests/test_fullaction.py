import re
import numpy as np
import pytest
from cortado.fullaction import (FullActionAgent, commitments, harmonic_probability, parse_sequence,
                               pot_fraction, to_real)
from cortado.game import game, observe
from cortado.policies import Policy
from cortado.pool import PoolPolicy, STYLES
from cortado.evaluation import evaluate
from cortado.training import train


class Uniform(Policy):
    """Checkpoint-shaped stub that never hits the table (exercises the fallback path)."""
    def __init__(self):
        self.kind, self.data, self.misses, self.decisions = 'checkpoint', None, 0, 0

    def strategy(self, observation):
        n = len(observation.legal_actions)
        return np.ones(n) / n, False


def random_hands(count, seed, chooser):
    g = game('holdem', full_actions=True)
    rng = np.random.default_rng(seed)
    for _ in range(count):
        s = g.new_initial_state()
        while not s.is_terminal():
            if s.is_chance_node():
                s.apply_action(int(rng.choice([a for a, _ in s.chance_outcomes()])))
            else:
                s.apply_action(chooser(s, observe(s), rng))
                yield s


def test_commitments_and_sequence_match_engine():
    def chooser(s, o, rng):
        return int(rng.choice(o.legal_actions))
    checked = 0
    for s in random_hands(150, 1, chooser):
        if s.is_terminal() or s.is_chance_node():
            continue
        o = observe(s)
        spent = [int(x) for x in re.search(r'Spent: \[P0: (\d+)\s+P1: (\d+)', str(s)).groups()]
        assert commitments(o.information) == spent
        parse_sequence(o.information)
        checked += 1
    assert checked > 300


def test_pseudo_harmonic_probabilities():
    assert harmonic_probability(0, 1, .5) == pytest.approx(1 / 3)
    assert harmonic_probability(0, 1, 0) == 1 and harmonic_probability(0, 1, 1) == 0
    assert harmonic_probability(1, 3, 2) == pytest.approx((3 - 2) * 2 / ((3 - 1) * 3))
    with pytest.raises(ValueError):
        harmonic_probability(1, 3, 4)


def test_real_action_mapping_and_pot_fraction():
    legal = (0, 1) + tuple(range(18, 201))
    # facing a bet of 10 (we have 2): pot raise goes to 30, bounded by legal raises.
    assert to_real(2, legal, [2, 10], 0) == 30
    assert to_real(3, legal, [2, 10], 0) == 200 and to_real(1, legal, [2, 10], 0) == 1
    assert to_real(0, (1, 4, 5), [2, 2], 0) == 1  # fold becomes check when checking is free
    assert to_real(2, (0, 1), [2, 200], 0) == 1   # no raise legal after an all-in
    assert pot_fraction(30, [2, 10], 0) == pytest.approx(1.0)
    assert pot_fraction(6, [2, 2], 0) == pytest.approx(1.0)


def test_agent_always_legal_and_distribution_is_valid():
    agent = FullActionAgent(Uniform())
    def chooser(s, o, rng):
        if rng.random() < .5:  # arbitrary off-abstraction sizes in the history
            return int(rng.choice(o.legal_actions))
        actions, probabilities, _ = agent.distribution(o)
        assert probabilities.sum() == pytest.approx(1) and set(actions) <= set(o.legal_actions)
        return agent.act(o, rng)
    assert sum(1 for _ in random_hands(120, 2, chooser)) > 200
    assert agent.stats['desync_heuristic'] + agent.stats['heuristic_miss'] == agent.decisions


def test_agent_ignores_opponent_hole_cards_and_is_stateless():
    agent = FullActionAgent(Uniform())
    g = game('holdem', full_actions=True)
    outputs = []
    for opponent in ((4, 5), (30, 31)):
        s = g.new_initial_state()
        for card in (0, 9, *opponent):
            s.apply_action(card)
        s.apply_action(14)   # off-abstraction raise to 14 by seat 0... seat 1 may act
        o = observe(s)
        first = agent.distribution(o)
        second = agent.distribution(o)
        assert first[0] == second[0] and np.array_equal(first[1], second[1])
        outputs.append((o.player, first[0], tuple(first[1])))
    assert outputs[0] == outputs[1]


def test_agent_uses_table_and_all_pool_styles_are_legal(tmp_path):
    path = tmp_path / 'tiny.json.gz'
    train('holdem', 120, 3, path, representation='buckets-v1')
    agent = FullActionAgent(Policy('checkpoint', path))
    for style in STYLES:
        report = evaluate(agent, PoolPolicy(style, samples=16), 6, 7, 'holdem', full_actions=True)
        assert report['hands'] == 12
    assert agent.stats['table'] > 0


def test_full_action_evaluation_rejects_fcpa_checkpoint(tmp_path):
    path = tmp_path / 'tiny.json.gz'
    train('holdem', 5, 1, path, representation='buckets-v1')
    with pytest.raises(ValueError):
        evaluate(Policy('checkpoint', path), Policy('call'), 2, 1, 'holdem', full_actions=True)


def test_configurable_stack_trains_and_plays_full_actions(tmp_path):
    path = tmp_path / 'deep.json.gz'
    train('holdem', 60, 4, path, representation='buckets-v1', stack_bb=200)
    agent = FullActionAgent(Policy('checkpoint', path))
    assert agent.stack == 400
    for style in ('tag', 'overbet'):
        report = evaluate(agent, PoolPolicy(style, samples=16), 5, 9, 'holdem', full_actions=True, stack_bb=200)
        assert report['hands'] == 10
    with pytest.raises(ValueError):   # a 100 BB game must reject a 200 BB checkpoint
        evaluate(agent, PoolPolicy('tag', samples=16), 2, 9, 'holdem', full_actions=True, stack_bb=100)
