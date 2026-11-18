"""Real-history river re-solving for bet-size-scheme blueprints (`--agent rhresolve`).

The blueprint plays every decision whose public history stayed on the trained tree. On the river,
when the history left the tree (an opponent size that had to be translated, or a replay desync,
which previously fell back to the equity heuristic), the agent instead:

1. estimates both players' river-start ranges by Bayes' rule over the *real* history: each past
   action contributes the blueprint probability of the abstract action it maps to; an off-tree size
   contributes the translation-weighted mixture of its two neighbouring abstract sizes (soft
   translation), so no single arbitrary mapping decides the range. After a desync, later actions
   leave the ranges unchanged (nothing reliable is known about them);
2. builds the river subgame in real chips from the real river-start pot, with the real river
   actions so far forced into the tree, and solves it with CFR+ (`resolve.RiverSolver`);
3. plays its own hand's average strategy at the current node.

This is unsafe (non-gadget) re-solving: it trusts the range estimates. It is used only where the
blueprint's own translation is known to be off-tree, which is where the earlier river re-solver's
losses against on-tree opponents are avoided by construction. Whether it helps must be measured.
"""
import re
from collections import Counter
from itertools import combinations
import numpy as np
from .abstraction import information_key
from .cards import DECK, parse_information
from .fullaction import Desync, FullActionAgent, parse_sequence, street_base
from .resolve import CALL, RiverSolver

_PRIVATE = re.compile(r'\[Private: [^\]]*\]')
RIVER_SIZES = dict(bet_fracs=(0.5, 1.0, 2.0), raise_fracs=(1.0,), max_bets=3)


class HistoryResolvingAgent(FullActionAgent):
    def __init__(self, base, iterations=200, min_reach=1e-3, **kw):
        super().__init__(base, **kw)
        if self.betting == 'fcpa':
            raise ValueError('Real-history re-solving needs a bet-size-scheme checkpoint')
        self.iterations, self.min_reach = iterations, min_reach
        self.resolve_stats = Counter()
        self._cache = {}

    def distribution(self, observation):
        info = observation.information
        if '[Round 3]' not in info:
            return super().distribution(observation)
        if info in self._cache:
            return self._cache[info]
        trace, desync = [], False
        try:
            self._replay(info, observation.player, trace)
        except Desync:
            desync = True
        off_tree = any(e[0] != 'river_start' and e[6] for e in trace)
        if not (desync or off_tree):
            result = super().distribution(observation)
        else:
            self.resolve_stats['desync' if desync else 'off_tree'] += 1
            try:
                result = self._resolve(observation, trace)
            except Desync:
                self.resolve_stats['failed'] += 1
                result = super().distribution(observation)
        if len(self._cache) > 64:
            self._cache.clear()
        self._cache[info] = result
        return result

    def _likelihood(self, hands, entry, rep, table):
        """Blueprint probability of the recorded (soft-translated) action for each candidate hand."""
        _, _, text, legal, _, bracket, _ = entry
        out = np.empty(len(hands))
        for i, h in enumerate(hands):
            row = table.get(information_key(_PRIVATE.sub(f'[Private: {h[0]}{h[1]}]', text), legal, rep))
            if row is None:
                out[i] = sum(w for _, w in bracket) / len(legal)
                continue
            avg = np.asarray(row[1], float)
            total = avg.sum()
            probs = avg / total if total > 0 else np.full(len(legal), 1.0 / len(legal))
            out[i] = sum(w * probs[legal.index(a)] for a, w in bracket if a in legal)
        return out

    def _ranges(self, me, hole, board, trace):
        rep = self.base.data.get('representation', 'exact-v1')
        table = self.base.data['table']
        everything = [h for h in combinations(DECK, 2) if not set(h) & set(board)]
        ranges = []
        for player in (0, 1):
            hands = everything if player == me else [h for h in everything if not set(h) & set(hole)]
            weights = np.ones(len(hands))
            for entry in trace:
                if entry[0] == 'river_start' or entry[1] != player or entry[0] == 3:
                    continue
                shown = board[:0 if entry[0] == 0 else entry[0] + 2]
                alive = np.array([not set(h) & set(shown) for h in hands])
                weights = np.where(alive, weights, 0.0)
                idx = np.flatnonzero(weights > 0)
                weights[idx] *= self._likelihood([hands[i] for i in idx], entry, rep, table)
            if weights.max() <= 0:
                weights = np.ones(len(hands))
            keep = weights > self.min_reach * weights.max()
            if player == me:
                keep[hands.index(hole)] = True
                weights[hands.index(hole)] = max(weights[hands.index(hole)], 1e-12)
            ranges.append(([h for h, k in zip(hands, keep) if k], weights[keep]))
        return ranges

    def _resolve(self, observation, trace):
        info, me = observation.information, observation.player
        hole, board = parse_information(info)
        hole = tuple(sorted(hole, key=DECK.index))
        base = street_base(info)
        forced = [CALL if kind == 'c' else amount for kind, amount, _ in parse_sequence(info)[3]]
        if any(kind == 'f' for kind, _, _ in parse_sequence(info)[3]):
            raise Desync('fold on the river path')
        (h0, w0), (h1, w1) = self._ranges(me, hole, board, trace)
        solver = RiverSolver(board, [h0, h1], [w0, w1], (base, base), self.stack, first=1,
                             forced=tuple(forced), iterations=self.iterations, **RIVER_SIZES)
        actions, probabilities = solver.strategy_at(forced, me, hole)
        legal = set(observation.legal_actions)
        if not set(actions) <= legal:
            raise Desync('re-solved action not legal in the real game')
        probabilities = np.asarray(probabilities, float)
        keep = probabilities > 0
        self.resolve_stats['resolved'] += 1
        return [a for a, k in zip(actions, keep) if k], probabilities[keep] / probabilities[keep].sum(), 'resolve'
