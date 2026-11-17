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

def test_exact_resume(tmp_path):
    import json
    train('kuhn',200,7,tmp_path/'whole.json')
    train('kuhn',100,7,tmp_path/'part.json')
    train('kuhn',100,999,tmp_path/'resumed.json',tmp_path/'part.json')
    whole=json.loads((tmp_path/'whole.json').read_text())
    resumed=json.loads((tmp_path/'resumed.json').read_text())
    for key in ('table','rng','iterations','exact_exploitability','seed'):
        assert whole[key]==resumed[key]

def test_full_raise_bounds_and_short_allin():
    s=game(full_actions=True).new_initial_state()
    while s.is_chance_node(): s.apply_action(s.chance_outcomes()[0][0])
    assert s.legal_actions()==[0,1]+list(range(4,201))
    s.apply_action(150)
    # Full raise would exceed effective stack; only a short all-in is available.
    assert s.legal_actions()==[0,1,200]
    s.apply_action(200)
    assert s.legal_actions()==[0,1]
    s.apply_action(1)
    while not s.is_terminal():
        assert s.is_chance_node()
        s.apply_action(s.chance_outcomes()[0][0])
    assert sum(s.returns())==0

def test_hidden_cards_invariance():
    states=[]
    for opponent_cards in ((2,3),(4,5)):
        s=game().new_initial_state()
        for card in (0,1,*opponent_cards): s.apply_action(card)
        states.append(s)
    assert observe(states[0])==observe(states[1])

def test_reject_checkpoint_spec(tmp_path):
    import json
    train('kuhn',2,1,tmp_path/'c.json')
    data=json.loads((tmp_path/'c.json').read_text())
    data['specification']={'unexpected':1}
    (tmp_path/'c.json').write_text(json.dumps(data))
    with pytest.raises(ValueError): Policy('checkpoint',tmp_path/'c.json')

def test_board_royal_flush_tie():
    s=game(full_actions=True).new_initial_state()
    for card in (0,1,4,5): s.apply_action(card)
    s.apply_action(200); s.apply_action(1)
    # Tc Jc Qc Kc Ac: both players must play the board.
    for card in (32,36,40,44,48):
        assert s.is_chance_node()
        s.apply_action(card)
    assert s.is_terminal() and s.returns()==[0,0]

def test_suit_canonicalization():
    from doppio.abstraction import canonical_information as canonical
    assert canonical('[Private: AcKd][Public: QcJh2s]') == canonical('[Private: AhKs][Public: QhJd2c]')
    assert canonical('[Private: AcKc]') != canonical('[Private: AcKd]')

def test_action_encoding_mismatch(tmp_path):
    train('holdem',1,42,tmp_path/'c.json')
    policy=Policy('checkpoint',tmp_path/'c.json')
    s=game(full_actions=True).new_initial_state()
    while s.is_chance_node(): s.apply_action(s.chance_outcomes()[0][0])
    with pytest.raises(ValueError): policy.act(observe(s),np.random.default_rng(1))

def test_bot_profile_and_deferred_human(tmp_path):
    import json
    from doppio.profiles import active_profile, PROFILES
    assert active_profile('bot').status == 'active'
    assert PROFILES['bot'].objective == PROFILES['human'].objective
    assert PROFILES['bot'].opponent_identity != PROFILES['human'].opponent_identity
    with pytest.raises(ValueError,match='deferred'):
        train('kuhn',1,42,tmp_path/'human.json',target='human')
    assert not (tmp_path/'human.json').exists()
    train('kuhn',2,42,tmp_path/'bot.json')
    checkpoint=json.loads((tmp_path/'bot.json').read_text())
    assert checkpoint['profile']['target']=='bot'

def test_cli_rejects_deferred_human():
    import subprocess,sys
    result=subprocess.run([sys.executable,'-m','doppio.cli','evaluate','--mode','human'],capture_output=True,text=True)
    assert result.returncode==2
    assert 'deferred' in result.stderr

def test_leduc_cfr_validation(tmp_path):
    r=train('leduc',10000,42,tmp_path/'leduc.json.gz')
    assert r['exact_exploitability'] < 0.3
    assert r['information_sets'] == 936

def test_raise_action_mapping():
    s=dealt()
    assert [s.action_to_string(a).split('move=')[-1] for a in s.legal_actions()]==['Fold','Call','Bet','AllIn']
    s.apply_action(2)
    assert s.current_player()==1
    assert 'move=Call' in s.action_to_string(1)
    assert 1 in s.legal_actions()


def test_full_minimum_reraise_and_uncalled_return():
    s=game(full_actions=True).new_initial_state()
    while s.is_chance_node():s.apply_action(s.chance_outcomes()[0][0])
    s.apply_action(6)
    assert s.legal_actions()==[0,1]+list(range(10,201))
    s.apply_action(0)
    assert s.is_terminal() and s.returns()==[2,-2]
