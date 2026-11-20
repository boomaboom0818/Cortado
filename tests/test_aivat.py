import numpy as np
import pytest
from cortado.aivat import equity
from cortado.cards import DECK
from cortado.evaluation import evaluate
from cortado.fullaction import FullActionAgent
from cortado.game import game
from cortado.pool import PoolPolicy
from cortado.policies import Policy


class Uniform(Policy):
    def __init__(self):
        self.kind, self.data, self.misses, self.decisions = 'checkpoint', None, 0, 0

    def strategy(self, observation):
        n = len(observation.legal_actions)
        return np.ones(n) / n, False


def dealt(cards, actions=(1, 1)):
    s = game('holdem', full_actions=True).new_initial_state()
    for c in cards[:4]:
        s.apply_action(DECK.index(c))
    for a in actions:
        s.apply_action(a)
    for c in cards[4:]:
        while not s.is_chance_node():
            s.apply_action(1)
        s.apply_action(DECK.index(c))
    return s


def test_chance_term_is_exactly_zero_mean_on_the_river():
    turn = dealt(('Ah', 'Kh', '9c', '9d', '2h', '7h', 'Qs', '3c'))
    while not turn.is_chance_node():
        turn.apply_action(1)
    before = equity(turn, 0, np.random.default_rng(0))
    after = []
    for card, p in turn.chance_outcomes():
        child = turn.child(card)
        after.append(p * equity(child, 0, np.random.default_rng(0)))
    assert sum(after) == pytest.approx(before)


def test_aivat_reports_corrected_estimate_and_keeps_raw():
    agent = FullActionAgent(Uniform())
    r = evaluate(agent, PoolPolicy('tag', samples=16), 6, 3, 'holdem', full_actions=True, aivat=(1, 1, 1, 1))
    plain = evaluate(FullActionAgent(Uniform()), PoolPolicy('tag', samples=16), 6, 3, 'holdem', full_actions=True)
    assert r['mean'] == plain['mean'] and len(r['aivat_clusters']) == 6 and 'aivat_ci95' in r
