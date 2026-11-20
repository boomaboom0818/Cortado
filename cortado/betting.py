"""Versioned bet-size abstractions over OpenSpiel's unrestricted ('fullgame') no-limit game.

`fcpa` is OpenSpiel's built-in fold / call / pot / all-in abstraction. The schemes below instead
restrict the *full* game to fold, call, all-in and a few raise sizes given as pot fractions:
the raise increment over a call divided by the pot after calling (the same convention as
`fullaction.pot_fraction`). Because the abstract game is a subset of the real game, abstract
action ids are real raise-to totals in engine chips, and a real bet that matches an abstract
size needs no translation.

Sizes depend on the street and on how many raises the current street already contains;
past the listed levels only all-in remains. Sizes below the minimum raise are lifted to it,
sizes at or above the stack become all-in, and duplicates are removed. Never edit a scheme in
place: checkpoints record the scheme name, so add a new version instead.
"""

SCHEMES = {
    'sizes-v1': {
        # street -> list per raise level (0 = first bet/raise of the street) of pot fractions
        0: [(0.75, 1.5), (1.0,), (0.7,)],          # open 2.5 / 4 BB; 3-bet ~3x; one 4-bet size
        1: [(0.33, 0.75, 2.0), (1.0,), ()],        # flop: small, medium, 2x-pot overbet; one raise size
        2: [(0.66, 2.0), (1.0,), ()],
        3: [(0.5, 1.0, 2.0), (1.0,), ()],
    },
}
BETTINGS = ('fcpa',) + tuple(SCHEMES)


def raise_targets(scheme, street, raises, commits, actor, legal):
    """Abstract raise-to totals available to `actor` (a subset of `legal`, always incl. all-in)."""
    full = [a for a in legal if a >= 2]
    if not full:
        return []
    low, high = min(full), max(full)
    levels = SCHEMES[scheme][street]
    fractions = levels[raises] if raises < len(levels) else ()
    other = commits[1 - actor]
    targets = {high}
    for f in fractions:
        target = int(round(other + f * 2 * other))
        targets.add(high if target >= high else max(target, low))
    return sorted(targets)


class BettingState:
    """Full-game state restricted to a bet-size scheme. Tracks street, raises and commitments."""
    __slots__ = ('state', 'scheme', 'street', 'raises', 'commits', 'dealt')

    def __init__(self, state, scheme, street=0, raises=0, commits=(1, 2), dealt=0):
        self.state, self.scheme = state, scheme
        self.street, self.raises, self.commits, self.dealt = street, raises, tuple(commits), dealt

    def __getattr__(self, name):
        return getattr(self.state, name)

    def legal_actions(self, player=None):
        legal = self.state.legal_actions()
        if self.state.is_chance_node() or self.state.is_terminal():
            return legal
        actor = self.state.current_player()
        calls = [a for a in legal if a < 2]
        return calls + raise_targets(self.scheme, self.street, self.raises, self.commits, actor, legal)

    def _advance(self, action):
        street, raises, commits, dealt = self.street, self.raises, list(self.commits), self.dealt
        if self.state.is_chance_node():
            dealt += 1
            # 4 hole cards, then 3 flop, 1 turn, 1 river cards.
            street = {7: 1, 8: 2, 9: 3}.get(dealt, street)
            if dealt in (7, 8, 9):
                raises = 0
        else:
            actor = self.state.current_player()
            if action == 1:
                commits[actor] = max(commits)
            elif action >= 2:
                commits[actor] = action
                raises += 1
        return street, raises, commits, dealt

    def child(self, action):
        street, raises, commits, dealt = self._advance(action)
        return BettingState(self.state.child(action), self.scheme, street, raises, commits, dealt)

    def apply_action(self, action):
        self.street, self.raises, commits, self.dealt = self._advance(action)
        self.commits = tuple(commits)
        self.state.apply_action(action)

    def clone(self):
        return BettingState(self.state.clone(), self.scheme, self.street, self.raises, self.commits, self.dealt)


class BettingGame:
    def __init__(self, game, scheme):
        if scheme not in SCHEMES:
            raise ValueError(f'Unknown betting scheme {scheme}')
        self.game, self.scheme = game, scheme

    def __getattr__(self, name):
        return getattr(self.game, name)

    def new_initial_state(self):
        return BettingState(self.game.new_initial_state(), self.scheme)
