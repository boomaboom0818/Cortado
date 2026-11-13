"""Trusted simulator boundary. Units are half a big blind."""
from dataclasses import dataclass
import pyspiel

SPEC = dict(betting='nolimit', numPlayers=2, numRounds=4, blind='1 2',
            firstPlayer='1 2 2 2', numSuits=4, numRanks=13, numHoleCards=2,
            numBoardCards='0 3 1 1', stack='200 200', bettingAbstraction='fcpa')

def game(name='holdem'):
    if name == 'kuhn':
        return pyspiel.load_game('kuhn_poker')
    if name != 'holdem':
        raise ValueError(name)
    return pyspiel.load_game('universal_poker', SPEC)

@dataclass(frozen=True)
class Observation:
    player: int
    information: str
    legal_actions: tuple[int, ...]

def observe(state):
    if state.is_terminal() or state.is_chance_node():
        raise ValueError('Not a decision node')
    p = state.current_player()
    return Observation(p, state.information_state_string(p), tuple(state.legal_actions()))
