"""Versioned equity-bucket card abstraction (`buckets-v1`).

Preflop is lossless (169 strategic classes). Postflop keeps only the street and
an expected-hand-strength bucket (equity versus a uniformly random hand and
random runout) of the current street; earlier postflop buckets are forgotten
(imperfect recall). Public betting and board *texture* are not represented
beyond the street, so flush/straight textures and draw-vs-made distinctions are
approximated. Bucket edges are fixed quantiles from `data/buckets-v1.json`.
"""
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import numpy as np
from .cards import RANKS, canonical_cards, equity, parse_information

EDGE_FILE = Path(__file__).with_name('data') / 'buckets-v1.json'
SAMPLES = {0: 300, 3: 128, 4: 128, 5: 192}


def preflop_class(hole):
    (a, b) = sorted(hole, key=lambda c: RANKS.index(c[0]), reverse=True)
    if a[0] == b[0]:
        return a[0] + b[0]
    return a[0] + b[0] + ('s' if a[1] == b[1] else 'o')


@lru_cache(maxsize=1)
def edges():
    data = json.loads(EDGE_FILE.read_text())
    if data['version'] != 'buckets-v1':
        raise ValueError('Unexpected bucket edge file')
    return {int(k): np.array(v) for k, v in data['edges'].items()}, data['buckets']


def edge_digest():
    return hashlib.sha256(EDGE_FILE.read_bytes()).hexdigest()


@lru_cache(maxsize=400_000)
def _strength(hole, board):
    # Seeded by suit-canonical cards: isomorphic holdings get identical values.
    key = canonical_cards(hole, board)
    digest = hashlib.blake2b(repr(key).encode(), digest_size=8).digest()
    rng = np.random.default_rng(int.from_bytes(digest, 'big'))
    return equity(key[0], key[1], SAMPLES[len(board)], rng)


def strength(hole, board):
    return _strength(tuple(sorted(hole)), tuple(sorted(board)))


def bucket(hole, board, boundaries=None):
    table, count = edges()
    if boundaries is None:
        boundaries = table[len(board)]
    return int(np.searchsorted(boundaries, strength(hole, board), side='right'))


def bucket_information(information):
    """Replace private/public cards with a lossless preflop class or postflop bucket."""
    hole, board = parse_information(information)
    label = preflop_class(hole) if not board else f'S{len(board)}B{bucket(hole, board)}'
    return _CARDS.sub(f'[Cards: {label}]', information, count=1)

import re
_CARDS = re.compile(r'\[Private: [^\]]*\]\[Public: [^\]]*\]')
