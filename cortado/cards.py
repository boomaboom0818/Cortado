"""Card parsing and hand-strength estimates for abstraction and opponents.

Uses the independent MIT/Apache-licensed `phevaluator` hand evaluator, not
OpenSpiel's evaluator, so strategy code does not share engine internals.
"""
import re
from itertools import combinations, permutations
import numpy as np
from phevaluator import evaluate_cards

RANKS = '23456789TJQKA'
SUITS = 'cdhs'
DECK = tuple(r + s for r in RANKS for s in SUITS)
_INFO = re.compile(r'\[Private: ([^\]]*)\]\[Public: ([^\]]*)\]')


def split_cards(text):
    if len(text) % 2 or not all(text[i] in RANKS and text[i+1] in SUITS for i in range(0, len(text), 2)):
        raise ValueError('Malformed card string')
    return tuple(text[i:i+2] for i in range(0, len(text), 2))


def parse_information(information):
    """Return (hole, board) from an OpenSpiel universal_poker information string."""
    match = _INFO.search(information)
    if not match:
        raise ValueError('Information string has no card fields')
    return split_cards(match[1]), split_cards(match[2])


def canonical_cards(hole, board):
    """Suit-isomorphism canonical form; hole and board are unordered sets."""
    best = None
    for order in permutations(SUITS):
        m = dict(zip(SUITS, order))
        key = (tuple(sorted(c[0] + m[c[1]] for c in hole)),
               tuple(sorted(c[0] + m[c[1]] for c in board)))
        if best is None or key < best:
            best = key
    return best


def showdown_value(hole, opp, board):
    mine, theirs = evaluate_cards(*hole, *board), evaluate_cards(*opp, *board)
    return 1.0 if mine < theirs else 0.5 if mine == theirs else 0.0


_ID = {card: index for index, card in enumerate(DECK)}  # phevaluator integer ids use the same order


def equity(hole, board, samples=128, rng=None):
    """Monte Carlo equity against a uniformly random opponent hand and random runout."""
    rng = rng or np.random.default_rng(0)
    known = set(hole) | set(board)
    rest = [_ID[c] for c in DECK if c not in known]
    need = 5 - len(board)
    draws = np.argsort(rng.random((samples, len(rest))), axis=1)[:, :2 + need]
    hole_ids = [_ID[c] for c in hole]
    board_ids = tuple(_ID[c] for c in board)
    mine_cache = {}
    total = 0.0
    for row in draws:
        cards = [rest[i] for i in row]
        full = board_ids + tuple(cards[2:])
        mine = mine_cache.get(full)
        if mine is None:
            mine = mine_cache[full] = evaluate_cards(*hole_ids, *full)
        theirs = evaluate_cards(cards[0], cards[1], *full)
        total += 1.0 if mine < theirs else 0.5 if mine == theirs else 0.0
    return total / samples


def exact_river_equity(hole, board):
    rest = [c for c in DECK if c not in set(hole) | set(board)]
    return sum(showdown_value(hole, opp, board) for opp in combinations(rest, 2)) / (len(rest) * (len(rest) - 1) / 2)
