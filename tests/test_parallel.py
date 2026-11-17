import json
import pytest
from doppio.parallel import launch
from doppio.run import DEFAULT
from doppio.checkpoints import load


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
