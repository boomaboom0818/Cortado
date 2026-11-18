"""Client for the hosted Slumbot benchmark (https://slumbot.com, API documented in its sample_api.py).

Slumbot rules: heads-up NLHE, blinds 50/100 chips, 20,000-chip (200 BB) stacks reset every hand, no rake.
Our engine uses 1/2 chips per 0.5/1 BB, so one engine chip = 50 Slumbot chips and the matching
candidate must be a `stack_bb=200` checkpoint.

Wire format: "k" check, "c" call, "f" fold, "b<N>" bet/raise where N is the chips the bettor has put
in *on the current street*, "/" ends a street. `client_pos` 0 = we are the big blind (engine seat 1),
1 = small blind/button (engine seat 0).

The policy only ever sees an engine Observation built from our own hole cards, the public board and
the public action string. The opponent's seat receives dummy cards chosen from cards we have not seen;
they cannot influence our information string. Each request is sequential with a polite delay.
"""
import argparse
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path
import numpy as np
from .cards import DECK
from .fullaction import BLINDS, FullActionAgent, commitments, parse_sequence
from .game import game, observe, stack_of
from .policies import Policy

HOST = 'https://slumbot.com/slumbot/api/'
SCALE = 50          # Slumbot chips per engine chip
STACK_BB = 200
_TOKEN = re.compile(r'b(\d+)|k|c|f|/')


class ProtocolError(Exception):
    pass


def parse_slumbot(action):
    """Streets as lists of (kind, street_amount) with kind in k/c/f/b."""
    streets, current, position = [], [], 0
    for token in _TOKEN.finditer(action):
        if token.start() != position:
            raise ProtocolError(f'Malformed action {action!r}')
        position = token.end()
        if token[0] == '/':
            streets.append(current)
            current = []
        else:
            current.append((token[0][0], int(token[1]) if token[1] else 0))
    if position != len(action):
        raise ProtocolError(f'Malformed action {action!r}')
    streets.append(current)
    return streets


def nearest_legal(target, legal):
    raises = [a for a in legal if a >= 2]
    if not raises:
        raise ProtocolError('Opponent raise where no raise is legal')
    return min(raises, key=lambda a: (abs(a - target), a))


def replay(full_game, our_seat, hole, board, action, stats=None):
    """Engine state for our decision. Returns (state, rounding_events)."""
    used = set(hole) | set(board)
    dummy = tuple(c for c in DECK if c not in used)[:2]
    seats = (hole, dummy) if our_seat == 0 else (dummy, hole)
    state = full_game.new_initial_state()
    for cards in seats:
        for card in cards:
            state.apply_action(DECK.index(card))
    stack = stack_of(str(full_game))
    streets = parse_slumbot(action)
    dealt, rounded = 0, 0
    for index, moves in enumerate(streets):
        if index:
            need = 3 if index == 1 else 1
            if state.is_terminal():
                break
            if not state.is_chance_node():
                raise ProtocolError('Street ended in Slumbot string but not in engine')
            for card in board[dealt:dealt + need]:
                state.apply_action(DECK.index(card))
            dealt += need
        base = 0 if index == 0 else commitments(state.information_state_string(0), stack)[0]
        for kind, amount in moves:
            if state.is_terminal() or state.is_chance_node():
                raise ProtocolError('Action after engine round ended')
            legal = state.legal_actions()
            if kind in 'kc':
                move = 1
            elif kind == 'f':
                move = 0
            else:
                exact = base + amount / SCALE
                move = nearest_legal(round(exact), legal)
                rounded += move != exact
            if move not in legal:
                raise ProtocolError(f'Illegal translated action {kind}{amount}')
            state.apply_action(move)
    if stats is not None:
        stats['rounded_opponent_bets'] = stats.get('rounded_opponent_bets', 0) + rounded
    return state


def street_base(info):
    """Commitment each player had when the current street began (equal for both players)."""
    commits = list(BLINDS)
    rounds = parse_sequence(info)
    for index, actions in enumerate(rounds[:-1]):
        actor = 0 if index == 0 else 1
        for kind, amount, _ in actions:
            commits[actor] = max(commits) if kind == 'c' else amount if kind == 'r' else commits[actor]
            actor ^= 1
    if len(rounds) == 1:
        return 0
    if commits[0] != commits[1]:
        raise ProtocolError('Street started with unequal commitments')
    return commits[0]


def to_incr(state, move):
    """Our engine action id as a Slumbot incremental action string."""
    info = state.information_state_string(state.current_player())
    commits = commitments(info, stack_of(str(state.get_game())))
    if move == 0:
        return 'f'
    if move == 1:
        return 'k' if commits[0] == commits[1] else 'c'
    # Engine raise ids are whole-hand totals; Slumbot wants the street total.
    return f'b{(move - street_base(info)) * SCALE}'


class Client:
    def __init__(self, delay=0.25, timeout=30, opener=None):
        self.token, self.delay, self.timeout = None, delay, timeout
        self.opener = opener or self._post

    def _post(self, endpoint, body):
        request = urllib.request.Request(HOST + endpoint, data=json.dumps(body).encode(),
                                         headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read())

    def call(self, endpoint, body):
        # Only new_hand is safe to resend; a lost /act reply leaves the server state unknown.
        attempts = 7 if endpoint == 'new_hand' else 1
        for attempt in range(attempts):
            try:
                result = self.opener(endpoint, body)
                break
            except OSError:
                if attempt == attempts - 1:
                    raise
                time.sleep(min(60, 2 ** attempt))
        if 'error_msg' in result:
            raise ProtocolError(result['error_msg'])
        self.token = result.get('token') or self.token
        time.sleep(self.delay)
        return result

    def new_hand(self):
        return self.call('new_hand', {'token': self.token} if self.token else {})

    def act(self, incr):
        return self.call('act', {'token': self.token, 'incr': incr})


def completed(log_path):
    path = Path(log_path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def play(agent, hands, seed, log_path, client=None, verify=True):
    client = client or Client()
    full = game('holdem', full_actions=True, stack_bb=STACK_BB)
    rng = np.random.default_rng(seed)
    stats = {}
    log = open(log_path, 'a')
    results = []
    start_number = len(completed(log_path))
    for number in range(start_number, start_number + hands):
        response = client.new_hand()
        record = dict(hand=number, decisions=[])
        try:
            response = _play_hand(agent, client, full, rng, stats, response, record)
        except OSError as error:
            # Network failure mid-hand: outcome unknown; log and move on (independent of the cards).
            record.update(abandoned=repr(error))
            log.write(json.dumps(record) + '\n'); log.flush()
            continue
        record.update(final=response, seat=1 if response['client_pos'] == 0 else 0)
        if verify and response.get('bot_hole_cards'):
            record['engine_check'] = _verify(full, response)
        log.write(json.dumps(record) + '\n'); log.flush()
        results.append(record)
    return results, stats


def _play_hand(agent, client, full, rng, stats, response, record):
    while 'winnings' not in response:
        seat = 1 if response['client_pos'] == 0 else 0
        state = replay(full, seat, tuple(response['hole_cards']), tuple(response['board']),
                       response['action'], stats)
        if state.current_player() != seat:
            raise ProtocolError(f"Not our turn after {response['action']!r}")
        obs = observe(state)
        tick = time.perf_counter()
        move = agent.act(obs, rng)
        latency = time.perf_counter() - tick
        incr = to_incr(state, move)
        record['decisions'].append(dict(action=response['action'], move=move, incr=incr, ms=latency * 1000))
        response = client.act(incr)
    return response


def _verify(full, response):
    """Replay the finished hand with both hands and compare engine payoff to Slumbot's winnings."""
    seat = 1 if response['client_pos'] == 0 else 0
    hole, bot = tuple(response['hole_cards']), tuple(response['bot_hole_cards'])
    seats = (hole, bot) if seat == 0 else (bot, hole)
    state = full.new_initial_state()
    for cards in seats:
        for card in cards:
            state.apply_action(DECK.index(card))
    board = tuple(response['board'])
    dealt = 0
    for index, moves in enumerate(parse_slumbot(response['action'])):
        if index and not state.is_terminal():
            while state.is_chance_node() and dealt < len(board):
                state.apply_action(DECK.index(board[dealt])); dealt += 1
        base = 0 if index == 0 else commitments(state.information_state_string(0), stack_of(str(full)))[0]
        for kind, amount in moves:
            state.apply_action(1 if kind in 'kc' else 0 if kind == 'f' else nearest_legal(round(base + amount / SCALE), state.legal_actions()))
    while state.is_chance_node() and dealt < len(board):
        state.apply_action(DECK.index(board[dealt])); dealt += 1
    if not state.is_terminal():
        return dict(ok=False, reason='engine not terminal')
    engine = state.returns()[seat] * SCALE
    return dict(ok=abs(engine - response['winnings']) < 1e-6, engine=engine, slumbot=response['winnings'])


def main():
    p = argparse.ArgumentParser(description='Play a 200 BB candidate against the hosted Slumbot API')
    p.add_argument('checkpoint')
    p.add_argument('--hands', type=int, default=200)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--log', required=True)
    p.add_argument('--delay', type=float, default=0.25)
    p.add_argument('--resume', action='store_true', help='Append to an existing log until --hands total are logged')
    args = p.parse_args()
    if args.hands % 2:
        p.error('Use an even number of hands so both positions are played equally')
    base = Policy('checkpoint', args.checkpoint)
    if base.data.get('stack_bb', 100) != STACK_BB:
        p.error('Slumbot plays 200 BB; the checkpoint must be trained with stack_bb=200')
    with open(args.checkpoint, 'rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    agent = FullActionAgent(base)
    start = time.perf_counter()
    done = len(completed(args.log))
    if done and not args.resume:
        p.error('Log exists; pass --resume to continue it')
    _, stats = play(agent, args.hands - done, args.seed, args.log, Client(delay=args.delay))
    logged = completed(args.log)
    results = [r for r in logged if 'final' in r]
    chips = np.array([r['final']['winnings'] for r in results], dtype=float)
    bb = chips / 100
    n = len(bb)
    from scipy.stats import t
    half = float(t.ppf(.975, n - 1) * bb.std(ddof=1) / np.sqrt(n) * 100) if n > 1 else float('nan')
    summary = dict(log=str(args.log), checkpoint=args.checkpoint, checkpoint_sha256=digest, hands=n, mean_bb_per_100=float(bb.mean() * 100),
                   ci95=[float(bb.mean() * 100 - half), float(bb.mean() * 100 + half)],
                   hands_logged=len(logged), hands_abandoned=len(logged) - len(results),
                   sources_this_invocation=dict(agent.stats), protocol_stats_this_invocation=stats,
                   engine_checks=[r['engine_check'] for r in results if 'engine_check' in r],
                   baseline_keys=sorted({k for r in results for k in r['final']}),
                   wall_seconds=time.perf_counter() - start,
                   note='Independent hands, unpaired; Slumbot is not a duplicate-match opponent.')
    if all('baseline_winnings' in r['final'] for r in results):
        # Slumbot-in-our-seat with the same cards; self-play is zero-sum over seats, so the
        # difference is an unbiased, lower-variance estimate of our win rate against Slumbot.
        diff = (chips - np.array([r['final']['baseline_winnings'] for r in results], dtype=float)) / 100
        h = float(t.ppf(.975, n - 1) * diff.std(ddof=1) / np.sqrt(n) * 100)
        summary['baseline_adjusted_bb_per_100'] = float(diff.mean() * 100)
        summary['baseline_adjusted_ci95'] = [float(diff.mean() * 100 - h), float(diff.mean() * 100 + h)]
    summary['engine_checks_ok'] = sum(c['ok'] for c in summary.pop('engine_checks'))
    summary['engine_checks_total'] = sum(1 for r in results if 'engine_check' in r)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
