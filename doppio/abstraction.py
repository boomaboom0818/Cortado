"""Lossless suit relabeling for future abstract-game integration.

Retains board order (and therefore street history), ranks, suit relationships,
and the entire betting information. Not yet used by the solver/checkpoints.
"""
import re
from itertools import permutations

SUITS='cdhs'

def canonical_information(information):
    def candidate(order):
        mapping=dict(zip(SUITS,order))
        return re.sub(r'([2-9TJQKA])([cdhs])',
                      lambda m:m[1]+mapping[m[2]],information)
    return min(candidate(order) for order in permutations(SUITS))
