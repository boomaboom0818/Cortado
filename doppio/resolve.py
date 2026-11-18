"""River re-solving: a CFR+ subgame solve over real chip amounts at decision time.

At a river decision the agent
1. estimates both players' river-start ranges by Bayes' rule from the blueprint strategy along the
   (translated) public history; the opponent is modelled as playing the blueprint,
2. builds a river betting tree in real engine chips with default sizes PLUS the exact actions already
   taken on this river (so no bet-size translation is needed on the river),
3. solves it with vectorized CFR+ (alternating updates, linear averaging), and
4. plays its own hand's average strategy at the current node.

This is *unsafe* (non-gadget) nested re-solving: it assumes the river-start ranges are correct and
carries no guarantee against exploitation of the range estimate. Its purpose here is to remove
river bet-size translation error; whether it helps must be measured.
"""
import re
from itertools import combinations
import numpy as np
from phevaluator import evaluate_cards
from .abstraction import information_key
from .cards import DECK, parse_information
from .fullaction import Desync, FullActionAgent, commitments, parse_sequence, street_base
from .game import Observation

FOLD, CALL = 0, 1


class Node:
    __slots__ = ('player', 'actions', 'children', 'commits', 'terminal', 'regret', 'average', 'index')

    def __init__(self, player=None, commits=None, terminal=None):
        self.player, self.commits, self.terminal = player, commits, terminal
        self.actions, self.children = [], []


def build_tree(commits, stack, first, bet_fracs=(.33, .66, 1.0, 2.0), raise_fracs=(1.0,), max_bets=3, forced=()):
    """River tree in engine chips. Raise ids are whole-hand raise-to totals, as in the engine.

    `forced` lists the actual river actions so far; each is guaranteed to exist on its path.
    """
    def legal_raises(c, actor, last_increment):
        top = max(c)
        minimum = min(stack, top + max(last_increment, 2))
        return minimum, stack

    def grow(c, actor, bets, last_inc, checked, path):
        node = Node(actor, tuple(c))
        options = []
        facing = c[1 - actor] - c[actor]
        if facing > 0:
            options.append(FOLD)
        options.append(CALL)
        if bets < max_bets and max(c) < stack and c[actor] + facing < stack:
            lo, hi = legal_raises(c, actor, last_inc)
            pot_after_call = 2 * max(c)
            for f in (raise_fracs if facing else bet_fracs):
                options.append(int(min(max(round(max(c) + f * pot_after_call), lo), hi)))
            options.append(hi)
        if path:
            options.append(path[0])
        node.actions = sorted(set(options))
        for a in node.actions:
            rest = path[1:] if path and a == path[0] else ()
            if a == FOLD:
                child = Node(terminal=('fold', actor), commits=tuple(c))
            elif a == CALL:
                nc = list(c); nc[actor] = max(c)
                if facing > 0 or checked:
                    child = Node(terminal=('showdown',), commits=tuple(nc))
                else:
                    child = grow(nc, 1 - actor, bets, last_inc, True, rest)
            else:
                nc = list(c); nc[actor] = a
                child = grow(nc, 1 - actor, bets + 1, max(last_inc, a - max(c)), False, rest)
            node.children.append(child)
        return node
    for amount in forced:
        if not isinstance(amount, int):
            raise ValueError('forced actions are engine ids')
    return grow(list(commits), first, 0, 0, False, tuple(forced))


class Showdown:
    """O(n) blocker-aware river evaluation of own hands against an opponent reach vector."""

    def __init__(self, own, own_ranks, opp, opp_ranks):
        self.order = np.argsort(opp_ranks, kind='stable')
        sorted_ranks = np.asarray(opp_ranks)[self.order]
        self.right = np.searchsorted(sorted_ranks, own_ranks, 'right')
        self.left = np.searchsorted(sorted_ranks, own_ranks, 'left')
        index = {h: j for j, h in enumerate(opp)}
        self.same = np.array([index.get(h, -1) for h in own])
        flat, starts, segment = [], {}, {}
        for card in DECK:
            members = [j for j, h in enumerate(opp) if card in h]
            members.sort(key=lambda j: opp_ranks[j])
            starts[card] = len(flat)
            segment[card] = np.array([opp_ranks[j] for j in members])
            flat.extend(members)
        self.flat = np.array(flat, dtype=int)
        self.card_pos = {}
        for slot in (0, 1):
            cards = [h[slot] for h in own]
            start = np.array([starts[c] for c in cards])
            length = np.array([len(segment[c]) for c in cards])
            right = np.array([np.searchsorted(segment[c], r, 'right') for c, r in zip(cards, own_ranks)])
            left = np.array([np.searchsorted(segment[c], r, 'left') for c, r in zip(cards, own_ranks)])
            self.card_pos[slot] = (start, length, right, left)

    def evaluate(self, q):
        """(wins minus losses, compatible opponent mass) for every own hand."""
        cq = np.concatenate([[0.0], np.cumsum(q[self.order])])
        total = cq[-1]
        win, lose = total - cq[self.right], cq[self.left]
        cf = np.concatenate([[0.0], np.cumsum(q[self.flat])])
        mass = total - 0.0
        for slot in (0, 1):
            start, length, right, left = self.card_pos[slot]
            seg_total = cf[start + length] - cf[start]
            win = win - (seg_total - (cf[start + right] - cf[start]))
            lose = lose - (cf[start + left] - cf[start])
            mass = mass - seg_total
        mass = mass + np.where(self.same >= 0, q[np.maximum(self.same, 0)], 0.0)
        return win - lose, mass


class RiverSolver:
    def __init__(self, board, hands, reach, commits, stack, first=1, forced=(), iterations=300, gadget=None, **sizes):
        """`gadget=(player, values)`: safe re-solving. `player` may terminate at the root with each hand's
        blueprint value (chips, normalised per hand), so no range estimate of that player is trusted."""
        self.board, self.hands = board, hands
        self.reach = [np.asarray(r, float) for r in reach]
        self.root = build_tree(commits, stack, first, forced=forced, **sizes)
        ranks = [np.array([evaluate_cards(*h, *board) for h in hs]) for hs in hands]
        self.ranks = ranks
        self._fast = (Showdown(hands[0], ranks[0], hands[1], ranks[1]),
                      Showdown(hands[1], ranks[1], hands[0], ranks[0]))
        self.subgame = self.root
        if gadget is not None:
            who, values = gadget
            self.gadget_values = np.asarray(values, float)
            top = Node(who, commits)
            top.actions = ['T', 'F']
            top.children = [Node(terminal=('gadget', who), commits=commits), self.root]
            self.root = top
        self._init(self.root)
        for t in range(1, iterations + 1):
            for p in (0, 1):
                self._cfr(self.root, p, self.reach[0], self.reach[1], t)

    @property
    def result(self):
        """Dense +1/0/-1 matrix from player 0's view (for checks only; O(n^2))."""
        return np.sign(self.ranks[1][None, :] - self.ranks[0][:, None]).astype(float)

    @property
    def valid(self):
        return np.array([[not (set(a) & set(b)) for b in self.hands[1]] for a in self.hands[0]], float)

    def _init(self, node):
        if node.terminal:
            return
        n = len(self.hands[node.player])
        node.regret = np.zeros((n, len(node.actions)))
        node.average = np.zeros((n, len(node.actions)))
        for child in node.children:
            self._init(child)

    @staticmethod
    def _current(node):
        positive = np.maximum(node.regret, 0)
        total = positive.sum(axis=1, keepdims=True)
        uniform = np.full_like(positive, 1.0 / positive.shape[1])
        return np.where(total > 0, positive / np.where(total > 0, total, 1), uniform)

    def _terminal(self, node, p, opp_reach):
        c = node.commits
        if node.terminal[0] == 'gadget':
            who = node.terminal[1]
            if p == who:
                return self.gadget_values * self._fast[p].evaluate(opp_reach)[1]
            return -self._fast[p].evaluate(opp_reach * self.gadget_values)[1]
        if node.terminal[0] == 'fold':
            folder = node.terminal[1]
            amount = c[folder] if folder == p else c[1 - p]
            sign = -1 if folder == p else 1
            return sign * amount * self._fast[p].evaluate(opp_reach)[1]
        return c[p] * self._fast[p].evaluate(opp_reach)[0]

    def _cfr(self, node, p, r0, r1, t):
        """Counterfactual values for player p's hands; CFR+ update at p's nodes."""
        if node.terminal:
            return self._terminal(node, p, r1 if p == 0 else r0)
        sigma = self._current(node)
        if node.player == p:
            own = r0 if p == 0 else r1
            values = np.stack([self._cfr(child, p, *((r0 * sigma[:, k], r1) if p == 0 else (r0, r1 * sigma[:, k])), t)
                               for k, child in enumerate(node.children)], axis=1)
            value = (sigma * values).sum(axis=1)
            node.regret = np.maximum(node.regret + values - value[:, None], 0)
            node.average += t * own[:, None] * sigma
            return value
        value = 0
        for k, child in enumerate(node.children):
            if node.player == 0:
                value = value + self._cfr(child, p, r0 * sigma[:, k], r1, t)
            else:
                value = value + self._cfr(child, p, r0, r1 * sigma[:, k], t)
        return value

    @staticmethod
    def average(node):
        total = node.average.sum(axis=1, keepdims=True)
        n = node.average.shape[1]
        return np.where(total > 0, node.average / np.where(total > 0, total, 1), 1.0 / n)

    def exploitability(self):
        """Mean best-response gain (chips per hand pair, range-weighted) against the average strategy."""
        def br(node, p, opp_reach):
            if node.terminal:
                return self._terminal(node, p, opp_reach)
            if node.player == p:
                return np.max(np.stack([br(c, p, opp_reach) for c in node.children], axis=1), axis=1)
            avg = self.average(node)
            return sum(br(c, p, opp_reach * avg[:, k]) for k, c in enumerate(node.children))
        mass = self.reach[0] @ self._fast[0].evaluate(self.reach[1])[1]
        v0 = self.reach[0] @ br(self.root, 0, self.reach[1])
        v1 = self.reach[1] @ br(self.root, 1, self.reach[0])
        return float((v0 + v1) / 2 / mass)

    def strategy_at(self, path, player, hand):
        node = self.subgame
        for action in path:
            node = node.children[node.actions.index(action)]
        if node.terminal or node.player != player:
            raise Desync('river path does not lead to our decision')
        return node.actions, self.average(node)[self.hands[player].index(hand)]


_PRIVATE = re.compile(r'\[Private: [^\]]*\]')
_PRE_RIVER = re.compile(r'\[Sequences: (?:[^|\]]*\|){3}')


class ResolvingAgent(FullActionAgent):
    """Blueprint + translation before the river; CFR+ re-solve on the river."""

    def __init__(self, base, iterations=300, min_reach=1e-4, safe=True, **kw):
        super().__init__(base, **kw)
        self.iterations, self.min_reach, self.safe = iterations, min_reach, safe
        self._range_cache, self._decision_cache = {}, {}

    def distribution(self, observation):
        info = observation.information
        if '[Round 3]' not in info:
            return super().distribution(observation)
        if info in self._decision_cache:
            return self._decision_cache[info]
        result = self._river_distribution(observation)
        if len(self._decision_cache) > 64:
            self._decision_cache.clear()
        self._decision_cache[info] = result
        return result

    def _river_distribution(self, observation):
        info = observation.information
        try:
            actions, probabilities = self._resolve(observation)
        except Desync:
            return super().distribution(observation)
        if not set(actions) <= set(observation.legal_actions):
            raise AssertionError('Re-solver produced an illegal action')
        keep = probabilities > 0
        return [a for a, k in zip(actions, keep) if k], probabilities[keep] / probabilities[keep].sum(), 'resolve'

    def _ranges(self, info, me, hole, board, players=(0, 1)):
        trace = []
        self._replay(info, me, trace)
        self._river_start = next((e[1] for e in trace if e[0] == 'river_start'), None)
        rep = self.base.data.get('representation', 'exact-v1')
        everything = [h for h in combinations(DECK, 2) if not set(h) & set(board)]
        ranges = []
        for player in players:
            hands = everything if player == me else [h for h in everything if not set(h) & set(hole)]
            weights = np.ones(len(hands))
            for entry in trace:
                if entry[0] == 'river_start':
                    continue
                street, actor, text, legal, action = entry
                if actor != player or street == 3:
                    continue
                shown = board[:0 if street == 0 else street + 2]
                for i, h in enumerate(hands):
                    if weights[i] == 0:
                        continue
                    if set(h) & set(shown):
                        weights[i] = 0
                        continue
                    fake = _PRIVATE.sub(f'[Private: {h[0]}{h[1]}]', text)
                    row = self.base.data['table'].get(information_key(fake, legal, rep))
                    if row is None:
                        weights[i] *= 1.0 / len(legal)
                    else:
                        avg = np.asarray(row[1], float)
                        weights[i] *= avg[legal.index(action)] / avg.sum() if avg.sum() > 0 else 1.0 / len(legal)
            keep = weights > self.min_reach * weights.max()
            if player == me:
                keep[hands.index(tuple(sorted(hole, key=DECK.index)))] = True
            ranges.append(([h for h, k in zip(hands, keep) if k], weights[keep]))
        return ranges

    def _prepare(self, info, me, hole, board, base):
        if not self.safe:
            (h0, w0), (h1, w1) = self._ranges(info, me, hole, board)
            return [h0, h1], [w0, w1], None
        [(mine, weights)] = self._ranges(info, me, hole, board, players=(me,))
        opp = 1 - me
        theirs = [h for h in combinations(DECK, 2) if not set(h) & set(board)]
        values = self._blueprint_values(board, me, mine, weights, theirs, base)
        hands = [None, None]; reach = [None, None]
        hands[me], hands[opp] = mine, theirs
        reach[me], reach[opp] = weights, np.ones(len(theirs))
        return hands, reach, (opp, values)

    def _blueprint_values(self, board, me, mine, weights, theirs, base):
        """Per-hand expected value (real chips, whole hand) of each opponent hand against our range when
        both follow the blueprint's abstract river strategy. Abstract chips are rescaled to the real pot."""
        start = self._river_start
        if start is None:
            raise Desync('no abstract river start')
        opp = 1 - me
        rep = self.base.data.get('representation', 'exact-v1')
        table = self.base.data['table']
        ranks_t = np.array([evaluate_cards(*h, *board) for h in theirs])
        ranks_m = np.array([evaluate_cards(*h, *board) for h in mine])
        show = Showdown(theirs, ranks_t, mine, ranks_m)
        own = {me: mine, opp: theirs}

        def strategy(state, player):
            text = state.information_state_string(player)
            legal = tuple(state.legal_actions())
            rows = np.empty((len(own[player]), len(legal)))
            for i, h in enumerate(own[player]):
                row = table.get(information_key(_PRIVATE.sub(f'[Private: {h[0]}{h[1]}]', text), legal, rep))
                avg = np.asarray(row[1], float) if row is not None else np.ones(len(legal))
                rows[i] = avg / avg.sum() if avg.sum() > 0 else 1.0 / len(legal)
            return legal, rows

        def value(state, q, commits, folder):
            if folder is not None:
                diff, mass = show.evaluate(q)
                return (-commits[opp] if folder == opp else commits[me]) * mass
            if state.is_terminal() or state.is_chance_node():
                diff, _ = show.evaluate(q)
                return commits[opp] * diff
            player = state.current_player()
            legal, rows = strategy(state, player)
            total = 0.0
            for k, action in enumerate(legal):
                child = state.child(action)
                if action == FOLD:
                    sub_value = value(child, q * rows[:, k] if player == me else q, commits, player)
                else:
                    nc = commitments(child.information_state_string(0), self.stack) if not child.is_terminal() else [max(commits)] * 2
                    sub_value = value(child, q * rows[:, k] if player == me else q, nc, None)
                total = total + (sub_value if player == me else rows[:, k] * sub_value)
            return total

        abstract_commits = commitments(start.information_state_string(0), self.stack)
        cfv = value(start, np.asarray(weights, float), abstract_commits, None)
        mass = show.evaluate(np.asarray(weights, float))[1]
        per_hand = np.where(mass > 0, cfv / np.where(mass > 0, mass, 1), 0.0)
        return per_hand * (base / abstract_commits[0])

    def _resolve(self, observation):
        info, me = observation.information, observation.player
        hole, board = parse_information(info)
        hole = tuple(sorted(hole, key=DECK.index))
        river_start = info[:info.index('[Sequences: ')] + _PRE_RIVER.search(info)[0] + str(me)
        river_start = re.sub(r'\[Pot: \d+\]\[Money: \d+ \d+\]', '', river_start)
        base = street_base(info)
        forced = []
        for kind, amount, _ in parse_sequence(info)[3]:
            forced.append(CALL if kind == 'c' else amount)
        if river_start not in self._range_cache:
            if len(self._range_cache) > 16:
                self._range_cache.clear()
            self._range_cache[river_start] = self._prepare(info, me, hole, board, base)
        hands, reach, gadget = self._range_cache[river_start]
        solver = RiverSolver(board, hands, reach, (base, base), self.stack, first=1,
                             forced=tuple(forced), iterations=self.iterations, gadget=gadget)
        actions, probabilities = solver.strategy_at(forced, me, hole)
        self.last_solver = solver
        return actions, np.asarray(probabilities, float)
