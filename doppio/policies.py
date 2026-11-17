"""Policies receive immutable information sets, never engine states."""
from .checkpoints import load
from .abstraction import information_key
import numpy as np

class Policy:
    def __init__(self, kind='uniform', checkpoint=None):
        if kind not in ('uniform', 'call', 'aggressive', 'checkpoint'):
            raise ValueError(kind)
        self.kind = kind
        self.data = load(checkpoint) if checkpoint else None
        if kind == 'checkpoint' and (not self.data or self.data['schema'] not in (1, 2, 3)):
            raise ValueError('Versioned checkpoint required')
        self.misses = 0
        self.decisions = 0
        if self.data:
            from .game import game
            self.game_id = str(game(self.data["game"]))
            if self.data.get('representation') == 'buckets-v1':
                from .buckets import edge_digest
                if self.data.get('abstraction_digest') != edge_digest():
                    raise ValueError('Checkpoint bucket edges do not match installed edges')

    def strategy(self, observation):
        """Average-strategy probabilities over legal actions and whether the table hit."""
        if self.data and observation.game_id != self.game_id:
            raise ValueError("Checkpoint game/action encoding mismatch")
        actions = observation.legal_actions
        weights, hit = np.ones(len(actions)), False
        if self.data:
            row = self.data['table'].get(information_key(observation.information, actions, self.data.get('representation', 'exact-v1')))
            if row is not None:
                if len(row[1]) != len(actions):
                    raise ValueError('Checkpoint action mismatch')
                weights, hit = np.array(row[1]), True
        total = weights.sum()
        return (weights / total if total else np.ones(len(actions)) / len(actions)), hit

    def act(self, observation, rng):
        self.decisions += 1
        actions = observation.legal_actions
        if self.kind == 'call':
            return 1 if 1 in actions else actions[0]
        if self.kind == 'aggressive':
            return actions[-1]
        probabilities, hit = self.strategy(observation)
        self.misses += bool(self.data) and not hit
        return int(rng.choice(actions, p=probabilities))
