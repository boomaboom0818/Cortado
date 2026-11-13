"""Policies receive immutable information sets, never engine states."""
import json
import numpy as np

class Policy:
    def __init__(self, kind='uniform', checkpoint=None):
        if kind not in ('uniform', 'call', 'aggressive', 'checkpoint'):
            raise ValueError(kind)
        self.kind = kind
        self.data = json.load(open(checkpoint)) if checkpoint else None
        if kind == 'checkpoint' and (not self.data or self.data['schema'] != 1):
            raise ValueError('Versioned checkpoint required')
        self.misses = 0

    def act(self, observation, rng):
        actions = observation.legal_actions
        if self.kind == 'call':
            return 1 if 1 in actions else actions[0]
        if self.kind == 'aggressive':
            return actions[-1]
        weights = np.ones(len(actions))
        if self.data:
            row = self.data['table'].get(observation.information)
            if row is None:
                self.misses += 1
            else:
                if len(row[1]) != len(actions):
                    raise ValueError('Checkpoint action mismatch')
                weights = np.array(row[1])
        weights = weights / weights.sum() if weights.sum() else np.ones(len(actions))/len(actions)
        return int(rng.choice(actions, p=weights))
