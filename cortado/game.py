"""Trusted simulator boundary. Units are half a big blind."""
from dataclasses import dataclass
import pyspiel

SPEC = dict(betting='nolimit', numPlayers=2, numRounds=4, blind='1 2',
            firstPlayer='1 2 2 2', numSuits=4, numRanks=13, numHoleCards=2,
            numBoardCards='0 3 1 1', stack='200 200', bettingAbstraction='fcpa')

def spec(stack_bb=100, betting='fcpa'):
    """Engine specification. Bet-size schemes (cortado.betting) restrict the engine's full game."""
    if stack_bb < 2:
        raise ValueError('stack must be at least two big blinds')
    return dict(SPEC, stack=f'{2*stack_bb} {2*stack_bb}', bettingAbstraction='fcpa' if betting == 'fcpa' else 'fullgame')

def game(name='holdem', *, full_actions=False, stack_bb=100, betting='fcpa'):
    """Engine game. With a bet-size scheme (and not full_actions) returns the restricted abstract game."""
    if name in ('kuhn', 'leduc'):
        return pyspiel.load_game(name + '_poker')
    if name != 'holdem':
        raise ValueError(name)
    if full_actions or betting == 'fcpa':
        return pyspiel.load_game('universal_poker', dict(spec(stack_bb), bettingAbstraction='fullgame' if full_actions else 'fcpa'))
    from .betting import BettingGame
    return BettingGame(pyspiel.load_game('universal_poker', spec(stack_bb, betting)), betting)

@dataclass(frozen=True)
class Observation:
    player: int
    information: str
    legal_actions: tuple[int, ...]
    game_id: str = ""

def stack_of(game_id):
    """Per-player stack in chips from an engine game string."""
    import re
    return int(re.search(r'stack=(\d+) (\d+)', game_id)[1])

def observe(state):
    if state.is_terminal() or state.is_chance_node():
        raise ValueError('Not a decision node')
    p = state.current_player()
    return Observation(p, state.information_state_string(p), tuple(state.legal_actions()), str(state.get_game()))
