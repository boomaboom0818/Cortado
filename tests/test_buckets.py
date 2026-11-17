import numpy as np
import pytest
from doppio.abstraction import information_key
from doppio.buckets import bucket, bucket_information, edge_digest, preflop_class, strength
from doppio.cards import canonical_cards, exact_river_equity
from doppio.game import game, observe


def test_preflop_classes_are_lossless_169():
    deck = [r + s for r in '23456789TJQKA' for s in 'cdhs']
    classes = {preflop_class((a, b)) for i, a in enumerate(deck) for b in deck[i + 1:]}
    assert len(classes) == 169
    assert preflop_class(('Ah', 'Kh')) == preflop_class(('Kd', 'Ad')) == 'AKs'
    assert preflop_class(('Ah', 'Kd')) == 'AKo'


def test_suit_isomorphic_hands_share_buckets_and_strength():
    a = (('Ah', 'Kh'), ('Qh', '7h', '2c'))
    b = (('As', 'Ks'), ('Qs', '7s', '2d'))
    assert canonical_cards(*a) == canonical_cards(*b)
    assert strength(*a) == strength(*b) and bucket(*a) == bucket(*b)


def test_bucket_is_monotone_in_hand_strength_and_tracks_exact_equity():
    board = ('Ah', 'Kd', '7c', '3s', '2d')
    strong, weak = ('As', 'Ac'), ('9c', '8h')
    assert bucket(strong, board) > bucket(weak, board)
    assert abs(strength(strong, board) - exact_river_equity(strong, board)) < .12
    assert abs(strength(weak, board) - exact_river_equity(weak, board)) < .12


def test_bucket_key_hides_cards_but_keeps_betting():
    g = game('holdem')
    s = g.new_initial_state()
    for c in (0, 1, 9, 10):
        s.apply_action(c)
    s.apply_action(2)
    o = observe(s)
    key = information_key(o.information, o.legal_actions, 'buckets-v1')
    assert 'Private' not in key and 'Public' not in key and 'Sequences: r6' in key and '[Cards: ' in key
    assert len(edge_digest()) == 64
