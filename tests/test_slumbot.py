import numpy as np
import pytest
from doppio.cards import DECK
from doppio.fullaction import FullActionAgent, commitments
from doppio.game import game, stack_of
from doppio.policies import Policy
from doppio.slumbot import SCALE, ProtocolError, parse_slumbot, play, replay, to_incr
from doppio.training import train

FULL = game('holdem', full_actions=True, stack_bb=200)


def test_documented_example_matches_engine_commitments():
    # From Slumbot's sample_api.py: "b200c/kb400": the flop bet is pot-size to 400; called = 600 each.
    state = replay(FULL, 1, ('Ac', '9d'), ('2c', '7h', 'Ks'), 'b200c/kb400')
    assert commitments(state.information_state_string(1), 400) == [12, 4]
    assert state.current_player() == 1 and to_incr(state, 1) == 'c'
    assert to_incr(state, 36) == f'b{(36 - 4) * SCALE}'   # engine raise-to-total 36 = 1600 on the street


def test_malformed_strings_rejected():
    with pytest.raises(ProtocolError):
        parse_slumbot('b200x')
    assert parse_slumbot('b20000c///') == [[('b', 20000), ('c', 0)], [], [], []]


class FakeSlumbot:
    """Engine-backed stand-in for the server: a check/call bot, real cards, Slumbot wire format."""
    def __init__(self, seed):
        self.rng, self.hand = np.random.default_rng(seed), 0

    def __call__(self, endpoint, body):
        if endpoint == 'new_hand':
            self.client_pos = self.hand % 2
            self.hand += 1
            self.deck = [DECK[i] for i in self.rng.permutation(52)]
            self.seat = 1 if self.client_pos == 0 else 0
            self.state = FULL.new_initial_state()
            holes = (self.deck[0:2], self.deck[2:4])
            for cards in holes:
                for c in cards:
                    self.state.apply_action(DECK.index(c))
            self.hole, self.bot = holes[self.seat], holes[1 - self.seat]
            self.action, self.dealt = '', 4
        else:
            self._apply(body['incr'])
        self._bot_moves()
        out = dict(token='t', action=self.action, client_pos=self.client_pos, hole_cards=list(self.hole),
                   board=self._board())
        if self.state.is_terminal():
            out.update(winnings=self.state.returns()[self.seat] * SCALE, bot_hole_cards=list(self.bot))
        return out

    def _board(self):
        return list(self.deck[4:self.dealt])

    def _apply(self, incr):
        info = self.state.information_state_string(self.state.current_player())
        from doppio.fullaction import street_base
        if incr[0] == 'b':
            move = street_base(info) + int(incr[1:]) // SCALE
        else:
            move = 0 if incr == 'f' else 1
        assert move in self.state.legal_actions(), incr
        self.state.apply_action(move)
        self.action += incr
        self._deal()

    def _deal(self):
        while self.state.is_chance_node():
            need = 3 if self.dealt == 4 else 1
            for c in self.deck[self.dealt:self.dealt + need]:
                self.state.apply_action(DECK.index(c))
            self.dealt += need
            self.action += '/'

    def _bot_moves(self):
        while not self.state.is_terminal() and self.state.current_player() != self.seat:
            self._apply(to_incr(self.state, 1))


def test_round_trip_against_engine_backed_fake(tmp_path):
    path = tmp_path / 'deep.json.gz'
    train('holdem', 40, 2, path, representation='buckets-v1', stack_bb=200)
    from doppio.slumbot import Client
    client = Client(delay=0, opener=FakeSlumbot(3))
    results, stats = play(FullActionAgent(Policy('checkpoint', path)), 30, 5, tmp_path / 'log.jsonl', client)
    assert len(results) == 30 and all(r['engine_check']['ok'] for r in results)
    assert {r['seat'] for r in results} == {0, 1}


def test_lost_act_reply_abandons_hand_and_resume_continues(tmp_path):
    path = tmp_path / 'deep.json.gz'
    train('holdem', 20, 2, path, representation='buckets-v1', stack_bb=200)
    from doppio.slumbot import Client, completed
    fake, calls = FakeSlumbot(4), {'n': 0}

    def flaky(endpoint, body):
        calls['n'] += 1
        if endpoint == 'act' and calls['n'] == 5:
            raise OSError('connection reset')
        return fake(endpoint, body)
    agent = FullActionAgent(Policy('checkpoint', path))
    log = tmp_path / 'log.jsonl'
    play(agent, 6, 1, log, Client(delay=0, opener=flaky))
    play(agent, 4, 2, log, Client(delay=0, opener=fake))
    records = completed(log)
    assert [r['hand'] for r in records] == list(range(10))
    assert sum('abandoned' in r for r in records) == 1
    assert all(r['engine_check']['ok'] for r in records if 'final' in r)
