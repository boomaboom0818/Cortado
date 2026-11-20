"""`buckets-v3`: exact, potential-aware card abstraction from precomputed tables.

* Preflop: the 169 lossless classes (as in v1).
* Flop and turn: E[HS^2], the mean over all remaining runouts of the squared river hand strength,
  where river hand strength is the exact equity against a uniformly random opponent holding
  (blocker-aware, ties count half). Squaring rewards hands whose strength can improve, so draws
  and made hands of equal average strength are separated (Johanson et al., 2013).
* River: exact hand strength.

Everything is computed by full enumeration (no sampling) for every suit-isomorphic board, so
the tables are deterministic and reproducible with `python -m cortado.buckets3 build`.
Each street is split into `BUCKETS` equal-mass buckets over uniformly random deals. Tables are
generated into `cortado/data/buckets-v3/` (ignored by git); their digest is stored in checkpoints.
"""
import argparse
import hashlib
import json
import os
from functools import lru_cache
from itertools import combinations, permutations
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from phevaluator._pheval import evaluate_7cards
from .cards import DECK, parse_information

VERSION = 'buckets-v3'
BUCKETS = {3: 50, 4: 50, 5: 50}
DIR = Path(__file__).with_name('data') / VERSION
_ID = {card: index for index, card in enumerate(DECK)}  # id = 4 * rank + suit
PERMS = [np.array(p) for p in permutations(range(4))]
PAIR = np.full((52, 52), -1, dtype=np.int32)          # holding (x < y) -> index 0..1325
for _i, (_x, _y) in enumerate(combinations(range(52), 2)):
    PAIR[_x, _y] = PAIR[_y, _x] = _i


# -- exact strength computations ------------------------------------------------------------
def river_strength(board):
    """Exact HS of every holding on a 5-card board: dict-free arrays (pair indices, values)."""
    rest = [c for c in range(52) if c not in board]
    holds = list(combinations(rest, 2))
    b = tuple(board)
    ranks = np.array([evaluate_7cards(*b, x, y) for x, y in holds])     # lower is better
    order = np.sort(ranks)
    n = len(ranks)
    worse = n - np.searchsorted(order, ranks, 'right')
    tie = np.searchsorted(order, ranks, 'right') - np.searchsorted(order, ranks, 'left')
    first = np.array([h[0] for h in holds]); second = np.array([h[1] for h in holds])
    for card in rest:
        members = (first == card) | (second == card)
        seg = np.sort(ranks[members])
        for slot in (first, second):
            mask = slot == card
            r = ranks[mask]
            right, left = np.searchsorted(seg, r, 'right'), np.searchsorted(seg, r, 'left')
            worse[mask] -= len(seg) - right
            tie[mask] -= right - left
    tie += 1  # inclusion-exclusion: the holding itself was in both card segments; net effect removes it once
    opponents = (len(rest) - 2) * (len(rest) - 3) // 2
    hs = (worse + 0.5 * tie) / opponents
    return PAIR[first, second], hs


def turn_strength(board):
    """E[HS] and E[HS^2] over the river for every holding on a 4-card board."""
    total, square, count = np.zeros(1326), np.zeros(1326), np.zeros(1326)
    for river in range(52):
        if river in board:
            continue
        index, hs = river_strength(tuple(board) + (river,))
        total[index] += hs; square[index] += hs * hs; count[index] += 1
    valid = count > 0
    return np.flatnonzero(valid), total[valid] / count[valid], square[valid] / count[valid]


def flop_strength(board):
    """E[HS] and E[HS^2] over all unordered turn-river runouts for every holding on a flop."""
    total, square, count = np.zeros(1326), np.zeros(1326), np.zeros(1326)
    rest = [c for c in range(52) if c not in board]
    for turn, river in combinations(rest, 2):
        index, hs = river_strength(tuple(board) + (turn, river))
        total[index] += hs; square[index] += hs * hs; count[index] += 1
    valid = count > 0
    return np.flatnonzero(valid), total[valid] / count[valid], square[valid] / count[valid]


# -- suit isomorphism -------------------------------------------------------------------------
def canonical(board):
    """(canonical sorted board ids, permutation applied) under suit relabelling."""
    ids = np.asarray(board)
    ranks, suits = ids // 4, ids % 4
    best, best_perm = None, None
    for perm in PERMS:
        key = tuple(sorted((ranks * 4 + perm[suits]).tolist()))
        if best is None or key < best:
            best, best_perm = key, perm
    return best, best_perm


def boards(size):
    """Canonical boards of a given size with their multiplicity (number of raw boards)."""
    combos = np.array(list(combinations(range(52), size)), dtype=np.int64)
    ranks, suits = combos // 4, combos % 4
    best = None
    for perm in PERMS:
        mapped = np.sort(ranks * 4 + perm[suits], axis=1)
        code = np.zeros(len(combos), dtype=np.int64)
        for column in range(size):
            code = code * 52 + mapped[:, column]
        best = code if best is None else np.minimum(best, code)
    codes, counts = np.unique(best, return_counts=True)
    keys = []
    for code in codes.tolist():
        cards = []
        for _ in range(size):
            code, card = divmod(code, 52)
            cards.append(card)
        keys.append(tuple(reversed(cards)))
    return list(zip(keys, counts.tolist()))


# -- table generation -------------------------------------------------------------------------
def _work(args):
    size, board = args
    if size == 5:
        index, hs = river_strength(board)
        return index, hs.astype(np.float32)
    index, _, square = (flop_strength if size == 3 else turn_strength)(board)
    return index, square.astype(np.float32)


def build(workers=None):
    DIR.mkdir(parents=True, exist_ok=True)
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    meta = dict(version=VERSION, buckets=BUCKETS, streets={})
    for size in (5, 4, 3):
        listing = boards(size)
        keys = [k for k, _ in listing]
        weights = np.array([w for _, w in listing], dtype=float)
        values = np.full((len(keys), 1326), np.nan, dtype=np.float32)
        with Pool(workers) as pool:
            for row, (index, value) in enumerate(pool.imap(_work, [(size, k) for k in keys], chunksize=8)):
                values[row, index] = value
        # Equal-mass edges over uniformly random deals: each raw board equally likely, then each
        # valid holding equally likely.
        valid = ~np.isnan(values)
        # Weighted histogram on a 2^20 grid (values lie in [0, 1]); edges sit on grid boundaries.
        bins = 1 << 20
        cells = np.minimum((np.nan_to_num(values) * bins).astype(np.int32), bins - 1)
        mass = np.zeros(bins)
        for row in range(len(keys)):
            mass += np.bincount(cells[row][valid[row]], minlength=bins) * weights[row]
        cumulative = np.cumsum(mass) / mass.sum()
        n = BUCKETS[size]
        edges = [float((np.searchsorted(cumulative, q / n) + 1) / bins) for q in range(1, n)]
        table = np.full(values.shape, 255, dtype=np.uint8)
        table[valid] = np.searchsorted(np.array(edges), values[valid], side='right').astype(np.uint8)
        np.save(DIR / f'table-{size}.npy', table)
        np.save(DIR / f'boards-{size}.npy', np.array(keys, dtype=np.uint8))
        meta['streets'][size] = dict(boards=len(keys), edges=edges)
        print(f'street {size}: {len(keys)} canonical boards', flush=True)
    (DIR / 'meta.json').write_text(json.dumps(meta))
    print('digest', digest())


# -- runtime lookup ----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _tables():
    out = {}
    for size in (3, 4, 5):
        table = np.load(DIR / f'table-{size}.npy', mmap_mode='r')
        keys = np.load(DIR / f'boards-{size}.npy')
        out[size] = (table, {tuple(k.tolist()): i for i, k in enumerate(keys)})
    return out


@lru_cache(maxsize=1)
def digest():
    h = hashlib.sha256()
    for name in ('meta.json', 'table-3.npy', 'table-4.npy', 'table-5.npy', 'boards-3.npy', 'boards-4.npy', 'boards-5.npy'):
        h.update((DIR / name).read_bytes())
    return h.hexdigest()


@lru_cache(maxsize=1_000_000)
def bucket(hole, board):
    """Postflop bucket of a holding (card strings) on a 3-5 card board."""
    table, index = _tables()[len(board)]
    key, perm = canonical([_ID[c] for c in board])
    x, y = (_ID[c] for c in hole)
    x, y = (x // 4) * 4 + perm[x % 4], (y // 4) * 4 + perm[y % 4]
    value = int(table[index[key], PAIR[x, y]])
    if value == 255:
        raise ValueError('Holding overlaps the board')
    return value


def bucket_information(information):
    from .buckets import preflop_class, _CARDS
    hole, board = parse_information(information)
    label = preflop_class(hole) if not board else f'S{len(board)}V{bucket(tuple(sorted(hole)), tuple(sorted(board)))}'
    return _CARDS.sub(f'[Cards: {label}]', information, count=1)


def main():
    p = argparse.ArgumentParser(description='Build or inspect the exact buckets-v3 tables')
    p.add_argument('command', choices=['build', 'digest'])
    p.add_argument('--workers', type=int)
    a = p.parse_args()
    if a.command == 'build':
        build(a.workers)
    else:
        print(digest())


if __name__ == '__main__':
    main()
