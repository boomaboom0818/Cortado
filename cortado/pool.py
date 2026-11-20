"""Frozen, versioned scripted opponent pool for the unrestricted-size game.

Every opponent is written against the public observation only, evaluates hands
with `phevaluator`, and sizes bets in real chips, so none is limited to the
trained FCPA action set. These are *diagnostic* opponents: they expose sizing,
passivity and over/under-folding weaknesses, but are not claimed to be strong.
Do not change a style's parameters in place; add a new version instead.
"""
from dataclasses import dataclass
import numpy as np
from .cards import equity, parse_information
from .fullaction import commitments
from .game import stack_of

VERSION = 'pool-v1'


@dataclass(frozen=True)
class Style:
    bet_above: float        # equity needed to bet/raise for value
    call_margin: float      # extra equity over pot odds needed to continue
    size: float             # raise increment as a fraction of the pot after calling
    bluff: float = 0.0      # probability of betting/raising with a non-value hand
    raise_above: float = 0.9  # equity needed to raise when facing a bet
    random_size: bool = False
    open_above: float = None  # preflop override for first-in raising


STYLES = {
    'nit': Style(.72, .12, .75),
    'tag': Style(.62, .04, .66, .08, .78),
    'lag': Style(.52, -.02, .75, .25, .66),
    'station': Style(2., -.5, 0.),             # never raises, calls nearly everything
    'maniac': Style(0., -.5, 1.0, 1., 0.),      # always raises pot
    'min-raiser': Style(.50, .02, 0.0, .10, .60),
    'third-pot': Style(.50, .02, .33, .20, .75),
    'overbet': Style(.60, .02, 2.5, .30, .80),
    'random-size': Style(.50, .0, .75, .20, .70, True),
}


class PoolPolicy:
    data = None

    def __init__(self, name, samples=64):
        if name not in STYLES:
            raise ValueError(name)
        self.name, self.style, self.samples = name, STYLES[name], samples
        self.decisions = self.misses = 0

    def act(self, observation, rng):
        self.decisions += 1
        st, legal = self.style, observation.legal_actions
        hole, board = parse_information(observation.information)
        me = observation.player
        commits = commitments(observation.information, stack_of(observation.game_id))
        facing = commits[1 - me] - commits[me]
        pot = commits[0] + commits[1]
        e = equity(hole, board, self.samples, np.random.default_rng(int(rng.integers(2**32))))
        raises = [a for a in legal if a >= 2]
        strong = e >= (st.raise_above if facing else st.bet_above)
        if raises and (strong or rng.random() < st.bluff):
            frac = float(np.exp(rng.uniform(np.log(.2), np.log(4)))) if st.random_size else st.size
            target = commits[1 - me] + frac * (pot + facing)
            return int(min(max(round(target), min(raises)), max(raises)))
        if facing and 0 in legal and e < facing / (pot + facing) + st.call_margin:
            return 0
        return 1
