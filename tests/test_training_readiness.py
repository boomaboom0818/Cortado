import json
import os
import signal
import numpy as np
import pytest
from doppio.abstraction import AbstractGame, information_key
from doppio.game import game, observe
from doppio.training import train
from doppio.checkpoints import load, save
from doppio.policies import Policy


def test_abstract_resume_and_inference(tmp_path):
    whole=tmp_path/'whole.json.gz'
    partial=tmp_path/'partial.json.gz'
    train('holdem',20,17,whole,representation='suits-v1',checkpoint_every=7)
    train('holdem',10,17,partial,representation='suits-v1')
    train('holdem',10,0,partial,resume=partial)
    a,b=load(whole),load(partial)
    assert a['table']==b['table'] and a['rng']==b['rng']
    policy=Policy('checkpoint',whole)
    state=game().new_initial_state()
    for card in (0,1,4,5): state.apply_action(card)
    observation=observe(state)
    key=information_key(observation.information,observation.legal_actions,'suits-v1')
    # A deterministic synthetic average strategy checks the deployment lookup.
    a['table'][key]=[[0]*4,[0,1,0,0]]
    save(whole,a)
    policy=Policy('checkpoint',whole)
    assert policy.act(observation,np.random.default_rng(1))==1
    assert policy.misses==0
    with pytest.raises(ValueError,match='representation'):
        train('holdem',1,1,partial,resume=partial,representation='exact-v1')


def test_adapter_preserves_hidden_information():
    observations=[]
    for cards in ((0,1,4,5),(0,1,8,9)):
        s=game().new_initial_state()
        for card in cards: s.apply_action(card)
        obs=observe(s)
        observations.append(information_key(obs.information,obs.legal_actions,'suits-v1'))
    assert observations[0]==observations[1]


def test_periodic_metrics_and_budget(tmp_path):
    checkpoint=tmp_path/'c.json.gz'
    metrics=tmp_path/'metrics.jsonl'
    report=train('kuhn',5,42,checkpoint,checkpoint_every=2,metrics_path=metrics)
    rows=[json.loads(line) for line in metrics.read_text().splitlines()]
    assert [r['iterations'] for r in rows]==[2,4,5]
    report=train('kuhn',100,42,checkpoint,resume=checkpoint,max_seconds=1e-9)
    assert report['stop_reason']=='time_budget' and report['iterations']==6


def test_signal_saves_completed_iteration(tmp_path,monkeypatch):
    from open_spiel.python.algorithms.external_sampling_mccfr import ExternalSamplingSolver
    original=ExternalSamplingSolver.iteration
    def interrupted(self):
        original(self)
        os.kill(os.getpid(),signal.SIGTERM)
    monkeypatch.setattr(ExternalSamplingSolver,'iteration',interrupted)
    report=train('kuhn',100,1,tmp_path/'signal.json')
    assert report['stop_reason']=='SIGTERM' and report['iterations']==1
    assert load(tmp_path/'signal.json')['iterations']==1


def test_failed_save_preserves_existing(tmp_path):
    path=tmp_path/'data.json'
    save(path,{'valid':1})
    with pytest.raises(ValueError): save(path,{'invalid':float('nan')})
    assert json.loads(path.read_text())=={'valid':1}
    assert list(tmp_path.iterdir())==[path]


def test_run_manifest_and_resume(tmp_path):
    from doppio.run import DEFAULT, run
    config=tmp_path/'config.json'
    config.write_text(json.dumps(dict(DEFAULT,game='kuhn',representation='exact-v1',iterations=2)))
    directory=tmp_path/'run'
    assert run(config,directory)['iterations']==2
    with pytest.raises(ValueError,match='already initialized'): run(config,directory)
    assert run(config,directory,True)['iterations']==4
    assert (directory/'iteration-4.json.gz').exists()
    data=json.loads(config.read_text()); data['seed']=99; config.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='differs'): run(config,directory,True)
