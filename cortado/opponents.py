"""Bayesian hand-range primitive; likelihoods must come from calibrated public-action models."""
from itertools import combinations

class Range:
    def __init__(self, known_cards=()):
        self.weights = {h:1.0 for h in combinations(range(52),2) if not set(h)&set(known_cards)}
        self.update((), lambda h:1)

    def update(self, public_cards, likelihood, observations=0, prior_strength=100):
        # Shrink unvalidated/sparse evidence toward an uninformative likelihood.
        alpha = observations/(observations+prior_strength)
        updated = {h:w*((1-alpha)+alpha*max(0,float(likelihood(h))))
                   for h,w in self.weights.items() if not set(h)&set(public_cards)}
        total = sum(updated.values())
        if total <= 0:
            raise ValueError('Impossible evidence')
        self.weights = {h:w/total for h,w in updated.items()}
