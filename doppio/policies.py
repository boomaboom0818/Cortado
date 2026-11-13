"""Policies receive immutable information sets, never engine states."""
from .checkpoints import load
import numpy as np

class Policy:
    def __init__(self, kind='uniform', checkpoint=None):
        if kind not in ('uniform', 'call', 'aggressive', 'checkpoint'):
            raise ValueError(kind)
        self.kind = kind
        self.data = load(checkpoint) if checkpoint else None
        if kind == 'checkpoint' and (not self.data or self.data['schema'] not in (1, 2)):
            raise ValueError('Versioned checkpoint required')
        self.misses = 0
        self.decisions = 0
        if self.data:
            from .game import game
            self.game_id = str(game(self.data["game"]))

    def act(self, observation, rng):
        if self.data and observation.game_id != self.game_id:
            raise ValueError("Checkpoint game/action encoding mismatch")
        self.decisions += 1
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
