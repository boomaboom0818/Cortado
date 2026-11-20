import json
import pytest
from cortado.parallel import launch
from cortado.run import DEFAULT
from cortado.checkpoints import load


def test_independent_workers_and_resume(tmp_path):
    config=tmp_path/'config.json'
    config.write_text(json.dumps(dict(DEFAULT,game='kuhn',representation='exact-v1',iterations=20)))
    root=tmp_path/'experiment'
    report=launch(config,root,[42,43])
    assert report['exit_codes']==[0,0]
    a=load(root/'seed-42/latest.json.gz')
    b=load(root/'seed-43/latest.json.gz')
    assert a['iterations']==b['iterations']==20
    assert a['seed']==42 and b['seed']==43
    assert a['table']!=b['table']
    with pytest.raises(ValueError,match='exists'): launch(config,root,[42,43])
    report=launch(config,root,[42,43],True)
    assert all(s['iterations']==40 for s in report['summaries'].values())
    with pytest.raises(ValueError,match='identical'): launch(config,root,[42,44],True)
    with pytest.raises(ValueError,match='distinct'): launch(config,tmp_path/'bad',[42,42])


def test_bootstrap_fresh_budget(tmp_path):
    config=tmp_path/'config.json'
    config.write_text(json.dumps(dict(DEFAULT,game='kuhn',representation='exact-v1',iterations=2)))
    launch(config,tmp_path/'first',[42])
    updated=json.loads(config.read_text());updated['iterations']=3;config.write_text(json.dumps(updated))
    result=launch(config,tmp_path/'second',[42],bootstrap_from=tmp_path/'first')
    assert result['summaries']['42']['iterations']==5
    assert load(tmp_path/'first/seed-42/latest.json.gz')['iterations']==2


def test_status_without_loading_checkpoint(tmp_path):
    from cortado.status import status
    worker=tmp_path/'seed-42';worker.mkdir()
    (worker/'latest.json.gz').write_bytes(b'not loaded by health check')
    (worker/'metrics.jsonl').write_text('{"iterations": 123}\n{"partial":')
    r=status(tmp_path)
    assert r['workers'][0]['checkpoint_exists']
    assert r['workers'][0]['last_saved']['iterations']==123
    assert r['workers'][0]['checkpoint_age_seconds']>=0
