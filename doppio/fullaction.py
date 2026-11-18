"""Full-action adapter: plays an FCPA-abstracted strategy in the unrestricted game.

The trained table knows four actions (fold, call, pot raise, all-in). In the
real game the opponent may bet any integer-chip amount and our table must still
be consulted. The adapter is stateless: every decision rebuilds a *trusted
abstract replay* from the legitimately visible observation (own cards, public
board, public betting) and fills the opponent's unknown cards with dummy cards
that cannot influence our information key.

* Opponent raises are mapped to the abstraction by pseudo-harmonic translation
  (Ganzfried & Sandholm 2013), with randomness seeded by own cards and the
  public prefix so it is consistent within a hand and unpredictable across hands.
* Chosen abstract actions are mapped back to concrete legal chip amounts.
* Known limits: this is translation, not subgame solving. Real and abstract pots
  drift after mapped bets; off-abstraction sizes are still exploitable.
"""
import hashlib
import re
import time
from collections import Counter
import numpy as np
from .buckets import strength
from .cards import DECK, parse_information
from .game import SPEC, Observation, game, stack_of

STACK = int(SPEC['stack'].split()[0])
BLINDS = tuple(int(x) for x in SPEC['blind'].split())  # seat 0 small blind, seat 1 big blind
_ROUND = re.compile(r'\[Round (\d)\]')
_SEQ = re.compile(r'\[Sequences: ([^\]]*)\]')
_MONEY = re.compile(r'\[Money: (\d+) (\d+)\]')
_TOKEN = re.compile(r'r(\d+)|c|f|\|')
FOLD, CALL, BET, ALLIN = 0, 1, 2, 3


class Desync(Exception):
    """Real and abstract replays disagree; the caller plays a safe action."""


def parse_sequence(information):
    """Betting rounds as lists of (kind, amount); amounts are total-commit raise-to."""
    text = _SEQ.search(information)[1]
    rounds, current, position = [], [], 0
    for token in _TOKEN.finditer(text):
        if token.start() != position:
            raise ValueError('Malformed betting sequence')
        position = token.end()
        if token[0] == '|':
            rounds.append(current)
            current = []
        else:
            current.append((token[0][0], int(token[1]) if token[1] else 0, token[0]))
    if position != len(text):
        raise ValueError('Malformed betting sequence')
    rounds.append(current)
    return rounds


def commitments(information, stack=STACK):
    money = _MONEY.search(information)
    return [stack - int(money[1]), stack - int(money[2])]


def harmonic_probability(low, high, x):
    """Probability of choosing the smaller abstract size `low` for observed `x`."""
    if not low <= x <= high or high <= low:
        raise ValueError('x must lie inside a non-degenerate bracket')
    return (high - x) * (1 + low) / ((high - low) * (1 + x))


def pot_fraction(amount, commits, actor):
    """Raise increment over the call, as a fraction of the pot after calling."""
    mine, other = commits[actor], commits[1 - actor]
    return (amount - other) / (mine + other + other - mine)


def to_real(action, legal, commits, me):
    """Concrete legal action for an abstract choice. Raise ids are raise-to totals."""
    raises = [a for a in legal if a >= 2]
    if action == FOLD:
        return 0 if 0 in legal else 1
    if action == CALL or not raises:
        return 1
    if action == ALLIN:
        return max(raises)
    target = 3 * commits[1 - me]  # raise by the pot after calling
    return int(min(max(target, min(raises)), max(raises)))


def street_base(info):
    """Commitment each player had when the current street began (equal for both players)."""
    commits = list(BLINDS)
    rounds = parse_sequence(info)
    for index, actions in enumerate(rounds[:-1]):
        actor = 0 if index == 0 else 1
        for kind, amount, _ in actions:
            commits[actor] = max(commits) if kind == 'c' else amount if kind == 'r' else commits[actor]
            actor ^= 1
    if len(rounds) == 1:
        return 0
    if commits[0] != commits[1]:
        raise Desync('Street started with unequal commitments')
    return commits[0]


class FullActionAgent:
    """Policy-compatible wrapper. `data` is None so evaluation treats it as a full-action policy."""
    data = None

    def __init__(self, base, fallback='heuristic'):
        if base.kind != 'checkpoint':
            raise ValueError('FullActionAgent wraps a checkpoint policy')
        if fallback not in ('heuristic', 'uniform'):
            raise ValueError(fallback)
        self.base, self.fallback = base, fallback
        self.stack_bb = (base.data or {}).get('stack_bb', 100)
        self.betting = (base.data or {}).get('betting', 'fcpa')
        # The abstract game the table was trained on: FCPA, or a bet-size scheme over the full game.
        self.fcpa = game('holdem', stack_bb=self.stack_bb, betting=self.betting)
        self.fcpa_id = str(getattr(self.fcpa, 'game', self.fcpa))
        self.full_id = str(game('holdem', full_actions=True, stack_bb=self.stack_bb))
        self.stack = stack_of(self.full_id)
        self.decisions = self.misses = 0
        self.stats, self.latencies = Counter(), []

    def act(self, observation, rng):
        self.decisions += 1
        tick = time.perf_counter()
        actions, probabilities, source = self.distribution(observation)
        self.latencies.append(time.perf_counter() - tick)
        self.stats[source] += 1
        self.misses += source not in ('table', 'resolve')
        return int(rng.choice(actions, p=probabilities))

    def distribution(self, observation):
        """Exact action distribution (pure function of the observation) and its source.

        Also the white-box interface used by best-response adversaries.
        """
        if observation.game_id != self.full_id:
            raise ValueError('FullActionAgent plays only the full-action game')
        me, info, legal = observation.player, observation.information, observation.legal_actions
        commits = commitments(info, self.stack)
        try:
            state = self._replay(info, me)
        except Desync:
            # Typically a mapped all-in or a near-all-in the abstraction treats as terminal.
            source, mixture = 'desync_heuristic', {self._heuristic(observation, legal): 1.0}
        else:
            abstract = Observation(me, state.information_state_string(me), tuple(state.legal_actions()), self.fcpa_id)
            probabilities, hit = self.base.strategy(abstract)
            if hit or self.fallback == 'uniform':
                source, mixture = ('table' if hit else 'uniform_miss'), dict(zip(abstract.legal_actions, probabilities))
            else:
                source, mixture = 'heuristic_miss', {self._heuristic(observation, legal): 1.0}
        real = {}
        from_table = source in ('table', 'uniform_miss') and self.betting != 'fcpa'
        for abstract_action, weight in mixture.items():
            if from_table:
                action = self._scheme_to_real(abstract_action, state, legal, commits, me)
            else:
                action = to_real(abstract_action, legal, commits, me)
            real[action] = real.get(action, 0.0) + float(weight)
        if not set(real) <= set(legal):
            raise AssertionError('Adapter produced an illegal action')
        actions = sorted(real)
        return actions, np.array([real[a] for a in actions]), source

    # -- translation of public history into the abstract game ---------------------------
    def _replay(self, info, me, trace=None):
        hole, board = parse_information(info)
        rounds = parse_sequence(info)
        used = set(hole) | set(board)
        dummy = [c for c in DECK if c not in used][:2]
        order = (hole + tuple(dummy)) if me == 0 else (tuple(dummy) + hole)
        state = self.fcpa.new_initial_state()
        for card in order:
            state.apply_action(DECK.index(card))
        commits, prefix = list(BLINDS), ''
        for index, actions in enumerate(rounds):
            if index:
                if not state.is_chance_node():
                    raise Desync('abstract round closed late')
                for card in (board[:3] if index == 1 else board[index + 1:index + 2]):
                    state.apply_action(DECK.index(card))
            actor = 0 if index == 0 else 1
            if trace is not None and index == 3:
                trace.append(('river_start', state.clone()))
            for kind, amount, text in actions:
                if state.is_chance_node() or state.is_terminal() or state.current_player() != actor:
                    raise Desync('abstract turn order differs')
                prefix += text
                abstract = self._to_abstract(state, kind, amount, commits, actor, hole, board, prefix)
                if trace is not None:
                    trace.append((index, actor, state.information_state_string(actor), tuple(state.legal_actions()), abstract))
                state.apply_action(abstract)
                commits[actor] = max(commits) if kind == 'c' else amount if kind == 'r' else commits[actor]
                actor ^= 1
            prefix += '|'
        if commits != commitments(info, self.stack):
            raise Desync('real commitments inconsistent with observation')
        if state.is_chance_node() or state.is_terminal() or state.current_player() != me:
            raise Desync('abstract replay is not at our decision')
        return state

    def _scheme_to_real(self, action, state, legal, commits, me):
        """Real legal action with the same pot fraction as an abstract bet-size-scheme action."""
        raises = [a for a in legal if a >= 2]
        if action < 2 or not raises:
            return action if action in legal else 1
        if action >= self.stack:
            return max(raises)
        fraction = pot_fraction(action, list(state.commits), me)
        target = int(round(commits[1 - me] + fraction * 2 * commits[1 - me]))
        return int(min(max(target, min(raises)), max(raises)))

    def _scheme_to_abstract(self, state, kind, amount, commits, actor, hole, board, prefix):
        """Opponent's real action in the bet-size scheme: exact sizes map exactly, others by
        pseudo-harmonic translation between the neighbouring abstract fractions."""
        legal = state.legal_actions()
        if kind == 'f':
            return FOLD if FOLD in legal else CALL
        raises = [a for a in legal if a >= 2]
        if kind == 'c' or not raises:
            return CALL
        if amount >= self.stack:
            return max(raises)
        f = pot_fraction(amount, commits, actor)
        if len(raises) > 1 and self.stack - amount > 0.25 * 2 * amount:
            # Not effectively committed: an abstract all-in would end the abstract hand while the
            # real hand continues (replay desync), so translate among the non-all-in sizes only.
            raises = raises[:-1]
        points = sorted([(0.0, CALL)] + [(pot_fraction(a, list(state.commits), actor), a) for a in raises])
        if f >= points[-1][0]:
            choice = points[-1][1]
        else:
            i = max(j for j in range(len(points)) if points[j][0] <= f)
            (low, low_action), (high, high_action) = points[i], points[i + 1]
            if f == low:
                choice = low_action
            else:
                digest = hashlib.blake2b(repr((sorted(hole), sorted(board), prefix)).encode(), digest_size=8).digest()
                u = np.random.default_rng(int.from_bytes(digest, 'big')).random()
                choice = low_action if u < harmonic_probability(low, high, f) else high_action
        if choice == CALL and self._call_closes(state):
            return min(raises)
        return choice

    def _to_abstract(self, state, kind, amount, commits, actor, hole, board, prefix):
        if self.betting != 'fcpa':
            return self._scheme_to_abstract(state, kind, amount, commits, actor, hole, board, prefix)
        legal = state.legal_actions()
        if kind == 'f':
            return FOLD if FOLD in legal else CALL
        if kind == 'c':
            return CALL
        if amount >= self.stack:
            return ALLIN if ALLIN in legal else BET if BET in legal else CALL
        f = pot_fraction(amount, commits, actor)
        abs_commits = commitments(state.information_state_string(actor), self.stack)
        f_all = pot_fraction(self.stack, abs_commits, actor)
        high_action, high = (BET, 1.0) if BET in legal else (ALLIN, f_all)
        if ALLIN not in legal and BET not in legal:
            return CALL
        low_action, low = CALL, 0.0
        if f >= 1.0 and BET in legal and ALLIN in legal and f_all > 1.0:
            low_action, low, high_action, high = BET, 1.0, ALLIN, f_all
        if f >= high:
            return high_action
        if f <= low:
            choice = low_action
        else:
            digest = hashlib.blake2b(repr((sorted(hole), sorted(board), prefix)).encode(), digest_size=8).digest()
            u = np.random.default_rng(int.from_bytes(digest, 'big')).random()
            choice = low_action if u < harmonic_probability(low, high, f) else high_action
        if choice == CALL and self._call_closes(state):
            return high_action  # a raise must keep the betting round open
        return choice

    @staticmethod
    def _call_closes(state):
        child = state.child(CALL)
        return child.is_chance_node() or child.is_terminal()

    # -- coverage fallback ---------------------------------------------------------------
    def _heuristic(self, observation, legal):
        """Equity-threshold coverage fallback. Not CFR-derived; reported separately."""
        hole, board = parse_information(observation.information)
        commits = commitments(observation.information, self.stack)
        me = observation.player
        e = strength(hole, board)
        facing = commits[1 - me] - commits[me]
        pot = commits[0] + commits[1]
        can_raise = any(a >= 2 for a in legal)
        if facing > 0:
            if e < facing / (pot + facing) + 0.05:
                return FOLD
            return BET if e > 0.8 and can_raise else CALL
        return BET if e > 0.7 and can_raise else CALL
