import numpy as np
import pytest
from cortado.storage import DiskTable
from cortado.training import train
from cortado.checkpoints import load, save
from cortado.policies import Policy
from cortado.game import game, observe


def materialize(table):
    return {k:np.asarray(v).tolist() for k,v in table.items()}


def test_evicted_mutations(tmp_path):
    t=DiskTable(tmp_path/'work.sqlite',cache_rows=1)
    t['a']=[np.ones(2),np.ones(2)]
    t['a'][0][1]+=10
    t['b']=[np.zeros(2),np.ones(2)]
    assert t['a'][0][1]==11
    assert len(t)==2
    t.close()
    r=DiskTable(tmp_path/'work.sqlite',readonly=True)
    assert r['a'][0][1]==11
    with pytest.raises(TypeError): r['c']=[[1],[1]]
    r.close()


@pytest.mark.parametrize('name,iterations',[('kuhn',1000),('leduc',500),('holdem',30)])
def test_disk_matches_memory_and_exact_resume(tmp_path,name,iterations):
    train(name,iterations,42,tmp_path/'memory.json.gz')
    train(name,iterations//2,42,tmp_path/'disk.sqlite',checkpoint_every=17)
    train(name,iterations-iterations//2,0,tmp_path/'disk.sqlite',resume=tmp_path/'disk.sqlite')
    a,b=load(tmp_path/'memory.json.gz'),load(tmp_path/'disk.sqlite')
    assert a['rng']==b['rng'] and a['iterations']==b['iterations']
    assert a['table']==materialize(b['table'])
    if name!='holdem': assert a['exact_exploitability']==b['exact_exploitability']
    policy=Policy('checkpoint',tmp_path/'disk.sqlite')
    s=game(name).new_initial_state()
    while s.is_chance_node():s.apply_action(s.chance_outcomes()[0][0])
    assert policy.act(observe(s),np.random.default_rng(1)) in s.legal_actions()
    b['table'].close()


def test_failed_sqlite_save_preserves_checkpoint(tmp_path):
    path=tmp_path/'checkpoint.sqlite'
    train('kuhn',10,42,path)
    old=load(path)
    broken=dict(old);broken['seed']=float('nan')
    with pytest.raises(ValueError):save(path,broken)
    assert load(path)['iterations']==10
    old['table'].close()


def test_disk_bootstrap_and_resume(tmp_path):
    import json
    from cortado.parallel import launch
    from cortado.run import DEFAULT
    config=tmp_path/'config.json'
    config.write_text(json.dumps(dict(DEFAULT,game='kuhn',representation='exact-v1',iterations=10,storage='sqlite')))
    launch(config,tmp_path/'a',[42,43])
    launch(config,tmp_path/'b',[42,43],bootstrap_from=tmp_path/'a')
    r=launch(config,tmp_path/'b',[42,43],resume=True)
    assert all(s['iterations']==30 for s in r['summaries'].values())


def test_recursive_solver_with_one_cached_row(tmp_path,monkeypatch):
    import cortado.training as training
    original=training.DiskTable
    class TinyTable(original):
        def __init__(self,path,**kwargs):
            super().__init__(path,cache_rows=1,**kwargs)
    monkeypatch.setattr(training,'DiskTable',TinyTable)
    train('holdem',50,81,tmp_path/'expected.json.gz',representation='suits-v1')
    train('holdem',50,81,tmp_path/'actual.sqlite',representation='suits-v1')
    expected=load(tmp_path/'expected.json.gz')
    actual=load(tmp_path/'actual.sqlite')
    assert expected['table']==materialize(actual['table'])
    actual['table'].close()


def test_memory_guard_ignores_inherited_highwater(tmp_path,monkeypatch):
    import types
    import cortado.training as training
    monkeypatch.setattr(training.resource,'getrusage',lambda _:types.SimpleNamespace(ru_maxrss=10**12))
    result=train('kuhn',10,42,tmp_path/'guard.sqlite',max_rss_mb=1024)
    assert result['stop_reason']=='completed'
    assert result['current_rss_mb']<1024
    assert result['os_peak_rss_mb']>1024


def test_sqlite_crash_leaves_committed_snapshot(tmp_path):
    import subprocess,sys
    snapshot=tmp_path/'saved.sqlite'
    train('kuhn',10,42,snapshot)
    script="""
import os,sys
from cortado.storage import DiskTable
from cortado.checkpoints import load
saved=load(sys.argv[1])
working=DiskTable(sys.argv[2])
saved['table'].connection.backup(working.connection)
working['uncommitted']=[[10.],[1.]]
working.flush()
os._exit(9)
"""
    result=subprocess.run([sys.executable,'-c',script,str(snapshot),str(tmp_path/'working.sqlite')])
    assert result.returncode==9
    data=load(snapshot)
    assert data['iterations']==10 and 'uncommitted' not in data['table']
    data['table'].close()
