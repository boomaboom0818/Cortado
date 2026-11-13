"""Strict, non-executable checkpoint validation."""
import json
import gzip
import numpy as np
from .game import SPEC

ENGINE = 'open_spiel==2.0.2'

def load(path, name=None):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as stream:
        data = json.load(stream)
    if data.get('schema') not in (1, 2) or data.get('engine') != ENGINE:
        raise ValueError('Unsupported checkpoint schema or engine')
    if data.get('algorithm') != 'external_sampling_mccfr':
        raise ValueError('Unsupported algorithm')
    if data.get('game') not in ('holdem', 'kuhn') or (name and data['game'] != name):
        raise ValueError('Checkpoint game mismatch')
    if data.get('specification') != (SPEC if data['game']=='holdem' else {}):
        raise ValueError('Checkpoint specification mismatch')
    for key, row in data['table'].items():
        values = np.asarray(row, dtype=float)
        if not isinstance(key,str) or values.ndim != 2 or values.shape[0] != 2 or not values.shape[1]:
            raise ValueError('Malformed checkpoint row')
        if not np.isfinite(values).all() or (values[1] < 0).any():
            raise ValueError('Invalid checkpoint weights')
    return data
