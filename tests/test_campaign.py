import json
from cortado.campaign import campaign
from cortado.run import DEFAULT
from cortado.training import train


def test_bounded_cycles_and_terminal_reentry(tmp_path):
    config=tmp_path/'config.json'
    config.write_text(json.dumps(dict(DEFAULT,storage='sqlite',iterations=2,checkpoint_every=1)))
    historical=tmp_path/'historical.json.gz'
    train('holdem',1,1,historical)
    root=tmp_path/'campaign'
    state=campaign(config,root,[42],historical,cycles=2,pairs=2)
    assert state['status']=='completed' and state['completed_cycles']==2
    assert (root/'cycle-2-evaluation.json').exists()
    assert campaign(config,root,[42],historical,cycles=2,pairs=2)==state
    summary=json.loads((root/'experiment-summary.json').read_text())
    assert summary['summaries']['42']['iterations']==4


def test_resource_stop_never_restarts(tmp_path):
    config=tmp_path/'config.json'
    config.write_text(json.dumps(dict(DEFAULT,storage='sqlite',iterations=10,max_rss_mb=.001)))
    historical=tmp_path/'historical.json.gz'
    train('holdem',1,1,historical)
    root=tmp_path/'campaign'
    state=campaign(config,root,[42],historical,cycles=2,pairs=2)
    assert state['status']=='resource_stop' and state['completed_cycles']==1
    assert not (root/'cycle-2-training.json').exists()
