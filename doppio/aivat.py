"""AIVAT-style variance reduction (Burch et al. 2018) for local evaluation.

Unbiasedness only requires each correction term to have zero conditional mean:

* Chance events (hole deal, flop, turn, river; consecutive chance nodes form one event):
  term = w * 2 * (eq_after - eq_before), where eq is our exact showdown equity given *both*
  players' cards and the board (known to the trusted harness after the hand). eq_before is the
  expectation of eq_after over the deal, so the term has mean zero. eq is exact by enumeration
  from the flop on; preflop equity uses Monte Carlo with independent randomness, which keeps
  the term zero-mean.
* Our own decisions: term = u(after chosen) - sum_a pi(a) u(after a), with pi the candidate's
  exact strategy (`distribution`) and u a heuristic value (exact payoff after a fold, otherwise
  (2*eq - 1) * matched chips). Opponent decisions get no term (their strategy is unknown).

The heuristic value function and the weights `betas` only change the variance, never the mean,
provided they are fixed before the evaluation data are seen (tune them on separate seeds).
Corrections never reach a policy: they are computed after the hand from cloned states.
"""
import re
from itertools import combinations
import numpy as np
from .cards import DECK, showdown_value

_CARDS = re.compile(r'P0 Cards: (\S*)\s+P1 Cards: (\S*)\s+BoardCards (\S*)')
_SPENT = re.compile(r'Spent: \[P0: (\d+)\s+P1: (\d+)')
DEFAULT_BETAS = (1.0, 1.0, 1.0, 1.0)  # chosen on seeds 600000-range, validated on 910000-range


def _split(text):
    return tuple(text[i:i + 2] for i in range(0, len(text), 2))


def cards(state):
    p0, p1, board = _CARDS.search(str(state)).groups()
    return (_split(p0), _split(p1)), _split(board)


def spent(state):
    return [int(x) for x in _SPENT.search(str(state)).groups()]


def equity(state, seat, rng, preflop_samples=400):
    holes, board = cards(state)
    mine, theirs = holes[seat], holes[1 - seat]
    if len(mine) < 2 or len(theirs) < 2:
        return 0.5
    rest = [c for c in DECK if c not in set(mine) | set(theirs) | set(board)]
    need = 5 - len(board)
    if need == 0:
        return showdown_value(mine, theirs, board)
    if need <= 2:
        outs = list(combinations(rest, need))
        return sum(showdown_value(mine, theirs, board + o) for o in outs) / len(outs)
    total = 0.0
    for _ in range(preflop_samples):
        draw = tuple(rest[i] for i in rng.choice(len(rest), size=need, replace=False))
        total += showdown_value(mine, theirs, board + draw)
    return total / preflop_samples


class Recorder:
    """Collects the events of one hand from the evaluation loop."""
    def __init__(self):
        self.events, self._open = [], None

    def before_chance(self, state):
        if self._open is None:
            self._open = state.clone()

    def after_step(self, state):
        if self._open is not None and not state.is_chance_node():
            self.events.append(('chance', self._open, state.clone()))
            self._open = None

    def our_action(self, state, observation, action):
        self.events.append(('action', state.clone(), observation, action))


def _value(state, seat, eq):
    if state.is_terminal():
        return state.returns()[seat]
    return (2 * eq - 1) * min(spent(state))


def correction(recorder, seat, candidate, rng, betas=DEFAULT_BETAS):
    """Sum of zero-mean control-variate terms for one hand, in engine chips, from `seat`'s view."""
    total, cache = 0.0, {}

    def eq(state):
        key = str(state)
        if key not in cache:
            cache[key] = equity(state, seat, rng)
        return cache[key]

    for event in recorder.events:
        if event[0] == 'chance':
            _, before, after = event
            board = len(cards(before)[1])
            street = {0: 0, 3: 1, 4: 2, 5: 3}[board]
            weight = betas[street] * min(spent(before))
            total += weight * 2 * (eq(after) - eq(before))
        elif candidate is not None and hasattr(candidate, 'distribution'):
            _, state, observation, chosen = event
            actions, probabilities, _ = candidate.distribution(observation)
            current = eq(state)
            values = {a: _value(state.child(a), seat, current) for a in actions}
            total += values[chosen] - float(np.dot(probabilities, [values[a] for a in actions]))
    return total
