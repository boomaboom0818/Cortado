"""Lossless suit relabeling for future abstract-game integration.

Retains board order (and therefore street history), ranks, suit relationships,
and the entire betting information. Used by the versioned solver adapter and deployed checkpoint policy.
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

REPRESENTATIONS = ('exact-v1', 'suits-v1', 'buckets-v1', 'buckets-v2', 'buckets-v3')

def information_key(information, actions, representation='exact-v1'):
    if representation not in REPRESENTATIONS:
        raise ValueError('Unknown information representation')
    if representation == 'exact-v1':
        return information
    # Include ordered action IDs so merged keys cannot silently change semantics.
    if representation == 'buckets-v3':
        from .buckets3 import bucket_information
        return bucket_information(information) + '|actions:' + ','.join(map(str,actions))
    if representation in ('buckets-v1', 'buckets-v2'):
        from .buckets import bucket_information
        return bucket_information(information, representation == 'buckets-v2') + '|actions:' + ','.join(map(str,actions))
    return canonical_information(information) + '|actions:' + ','.join(map(str,actions))

class AbstractState:
    """Trusted solver adapter; never passed to a deployed policy."""
    def __init__(self, state, representation):
        self.state, self.representation = state, representation

    def __getattr__(self, name):
        return getattr(self.state, name)

    def information_state_string(self, player):
        return information_key(self.state.information_state_string(player),
                               self.state.legal_actions(), self.representation)

    def child(self, action):
        return AbstractState(self.state.child(action), self.representation)

class AbstractGame:
    def __init__(self, game, representation):
        self.game, self.representation = game, representation

    def __getattr__(self, name):
        return getattr(self.game, name)

    def new_initial_state(self):
        return AbstractState(self.game.new_initial_state(), self.representation)
