"""Local Best Response (Lisy & Bowling 2017) adversary against a white-box victim.

LBR tracks the victim's hand distribution Bayesianly from the victim's own action
probabilities, then greedily picks fold / call / a few raise sizes by one-step
lookahead, assuming it checks and calls down afterward. It needs the victim's
`distribution(observation)`. A positive LBR result demonstrates exploitability;
a weak LBR result does NOT bound exploitability (it is a myopic lower bound).
The adversary is a harness-side tool: it never touches engine state for the
victim, only constructs the victim's legitimate observations for hypothetical hands.
"""
from itertools import combinations
import numpy as np
from .cards import DECK, parse_information, showdown_value
from .fullaction import commitments, parse_sequence
from .game import game, observe

FULL = game('holdem', full_actions=True)


def action_ids(information):
    """Chronological (actor, action id) pairs from the public betting string."""
    pairs = []
    for index, actions in enumerate(parse_sequence(information)):
        actor = 0 if index == 0 else 1
        for kind, amount, _ in actions:
            pairs.append((actor, 0 if kind == 'f' else 1 if kind == 'c' else amount))
            actor ^= 1
    return pairs


def build(seat_holes, board, ids):
    """Trusted harness-side reconstruction of the real state for hypothetical private cards."""
    state = FULL.new_initial_state()
    for hole in seat_holes:
        for card in hole:
            state.apply_action(DECK.index(card))
    dealt = 0
    for action in ids:
        while state.is_chance_node():
            state.apply_action(DECK.index(board[dealt])); dealt += 1
        state.apply_action(action)
    while state.is_chance_node() and dealt < len(board):
        state.apply_action(DECK.index(board[dealt])); dealt += 1
    return state


class LBR:
    data = None

    def __init__(self, victim, hands=60, runouts=2, sizes=(.5, 1., 2.)):
        self.victim, self.hands, self.runouts, self.sizes = victim, hands, runouts, sizes
        self.decisions = self.misses = 0

    def _holes(self, me, mine, v_hole):
        return (mine, v_hole) if me == 0 else (v_hole, mine)

    def act(self, observation, rng):
        self.decisions += 1
        me, v = observation.player, 1 - observation.player
        legal = observation.legal_actions
        mine, board = parse_information(observation.information)
        history = action_ids(observation.information)
        commits = commitments(observation.information)
        known = set(mine) | set(board)
        pool = [c for c in DECK if c not in known]
        combos = list(combinations(pool, 2))
        picks = rng.choice(len(combos), size=min(self.hands, len(combos)), replace=False)
        ids = [a for _, a in history]
        memo = {}

        def probabilities(v_hole, prefix_ids):
            key = (v_hole, tuple(prefix_ids))
            if key not in memo:
                state = build(self._holes(me, mine, v_hole), board, prefix_ids)
                memo[key] = self.victim.distribution(observe(state))[:2]
            return memo[key]

        hands, weights = [], []
        for i in picks:
            h = combos[int(i)]
            w = 1.0
            for step, (actor, action) in enumerate(history):
                if actor == v and w > 0:
                    actions, probs = probabilities(h, ids[:step])
                    w *= float(probs[actions.index(action)]) if action in actions else 0.0
            hands.append(h); weights.append(w)
        weights = np.array(weights)
        if weights.sum() <= 0:
            weights = np.ones(len(hands))   # sampled support collapsed: fall back to uniform
        equities = np.array([self._equity(mine, h, board, rng) for h in hands])
        c_me, c_opp = commits[me], commits[v]
        pot, facing = c_me + c_opp, c_opp - c_me
        total = weights.sum()
        options = {}
        if 0 in legal:
            options[0] = -c_me
        options[1] = float((weights * c_opp * (2 * equities - 1)).sum() / total)
        raises = [a for a in legal if a >= 2]
        if raises:
            targets = {int(min(max(round(c_opp + f * (pot + facing)), min(raises)), max(raises))) for f in self.sizes}
            targets.add(max(raises))
            for target in sorted(targets):
                value = 0.0
                for h, w, e in zip(hands, weights, equities):
                    if w <= 0:
                        continue
                    actions, probs = probabilities(h, ids + [target])
                    p = dict(zip(actions, probs))
                    pf, pc = p.get(0, 0.0), p.get(1, 0.0)
                    value += w * (pf * c_opp + pc * target * (2 * e - 1) - (1 - pf - pc) * target)
                options[target] = value / total
        return max(options, key=lambda a: (options[a], a == 1))

    def _equity(self, mine, theirs, board, rng):
        need = 5 - len(board)
        if need == 0:
            return showdown_value(mine, theirs, board)
        rest = [c for c in DECK if c not in set(mine) | set(theirs) | set(board)]
        total = 0.0
        for _ in range(self.runouts):
            draw = tuple(rest[i] for i in rng.choice(len(rest), size=need, replace=False))
            total += showdown_value(mine, theirs, tuple(board) + draw)
        return total / self.runouts
