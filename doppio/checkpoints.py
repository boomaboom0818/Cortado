"""Strict, non-executable checkpoint validation."""
import json
import gzip
import os
from pathlib import Path
import tempfile
from .abstraction import REPRESENTATIONS
import numpy as np
from .game import SPEC

ENGINE = 'open_spiel==2.0.2'

def load(path, name=None):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt') as stream:
        data = json.load(stream)
    if data.get('schema') not in (1, 2, 3) or data.get('engine') != ENGINE:
        raise ValueError('Unsupported checkpoint schema or engine')
    if data.get('algorithm') != 'external_sampling_mccfr':
        raise ValueError('Unsupported algorithm')
    if data.get('game') not in ('holdem', 'kuhn', 'leduc') or (name and data['game'] != name):
        raise ValueError('Checkpoint game mismatch')
    if data.get('specification') != (SPEC if data['game']=='holdem' else {}):
        raise ValueError('Checkpoint specification mismatch')
    if data.get('representation', 'exact-v1') not in REPRESENTATIONS:
        raise ValueError('Unknown checkpoint representation')
    if data['game'] in ('kuhn', 'leduc') and data.get('representation', 'exact-v1') != 'exact-v1':
        raise ValueError('Kuhn requires exact representation')
    for key, row in data['table'].items():
        values = np.asarray(row, dtype=float)
        if not isinstance(key,str) or values.ndim != 2 or values.shape[0] != 2 or not values.shape[1]:
            raise ValueError('Malformed checkpoint row')
        if not np.isfinite(values).all() or (values[1] < 0).any():
            raise ValueError('Invalid checkpoint weights')
    return data


def save(path, data):
    """Atomic, durable replacement; compression matches destination suffix."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=target.parent)
    os.close(fd)
    try:
        opener = gzip.open if str(target).endswith('.gz') else open
        with opener(temporary, 'wt') as stream:
            json.dump(data, stream, sort_keys=True, allow_nan=False)
        with open(temporary, 'rb') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, target)
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
