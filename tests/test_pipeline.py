import numpy as np
import pytest
from doppio.game import game, observe
from doppio.policies import Policy
from doppio.evaluation import evaluate
from doppio.opponents import Range
from doppio.training import train

def dealt():
    s=game().new_initial_state()
    while s.is_chance_node(): s.apply_action(s.chance_outcomes()[0][0])
    return s

def test_order_and_fold():
    s=dealt()
    assert s.current_player()==0
    assert '2s2h' not in observe(s).information
    s.apply_action(0)
    assert s.returns()==[-1,1]

def test_postflop_order():
    s=dealt()
    s.apply_action(1); s.apply_action(1)
    while s.is_chance_node(): s.apply_action(s.chance_outcomes()[0][0])
    assert s.current_player()==1

def test_random_conservation():
    rng=np.random.default_rng(2)
    for _ in range(100):
        s=game().new_initial_state()
        while not s.is_terminal():
            actions=[a for a,p in s.chance_outcomes()] if s.is_chance_node() else s.legal_actions()
            s.apply_action(int(rng.choice(actions)))
        assert sum(s.returns())==0
        assert max(abs(x) for x in s.returns())<=200

def test_training(tmp_path):
    r=train('kuhn',10000,42,tmp_path/'k.json')
    assert r['exact_exploitability']<.03

def test_range():
    r=Range([0,1]); r.update([2,3,4],lambda h:2 if 5 in h else 1,100)
    assert sum(r.weights.values())==pytest.approx(1)
    assert all(not set(h)&{0,1,2,3,4} for h in r.weights)

def test_reproducibility():
    a=evaluate(Policy(),Policy('call'),10,23)
    b=evaluate(Policy(),Policy('call'),10,23)
    assert a['mean']==b['mean'] and a['ci95']==b['ci95']
