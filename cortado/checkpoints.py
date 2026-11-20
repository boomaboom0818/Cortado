"""Strict, non-executable checkpoint validation."""
import json
import gzip
import os
from pathlib import Path
import tempfile
import sqlite3
import shutil
from .storage import DiskTable
from .abstraction import REPRESENTATIONS
import numpy as np
from .game import spec

ENGINE = 'open_spiel==2.0.2'

def load(path, name=None):
    if str(path).endswith('.sqlite'):
        table = DiskTable(path, readonly=True)
        if table.connection.execute('PRAGMA quick_check').fetchone()[0] != 'ok':
            raise ValueError('Corrupt SQLite checkpoint')
        data = json.loads(table.connection.execute('SELECT json FROM metadata').fetchone()[0])
        if data.pop('storage_schema', None) != 1:
            raise ValueError('Unsupported storage schema')
        data['table'] = table
    else:
        opener = gzip.open if str(path).endswith('.gz') else open
        with opener(path, 'rt') as stream:
            data = json.load(stream)
    if data.get('schema') not in (1, 2, 3) or data.get('engine') != ENGINE:
        raise ValueError('Unsupported checkpoint schema or engine')
    if data.get('algorithm') != 'external_sampling_mccfr':
        raise ValueError('Unsupported algorithm')
    if data.get('game') not in ('holdem', 'kuhn', 'leduc') or (name and data['game'] != name):
        raise ValueError('Checkpoint game mismatch')
    if data.get('specification') != (spec(data.get('stack_bb', 100), data.get('betting', 'fcpa')) if data['game']=='holdem' else {}):
        raise ValueError('Checkpoint specification mismatch')
    if data.get('representation', 'exact-v1') not in REPRESENTATIONS:
        raise ValueError('Unknown checkpoint representation')
    if data['game'] in ('kuhn', 'leduc') and data.get('representation', 'exact-v1') != 'exact-v1':
        raise ValueError('Kuhn requires exact representation')
    # Disk rows are validated lazily on every read; quick_check checks the file.
    rows = () if isinstance(data['table'], DiskTable) else data['table'].items()
    for key, row in rows:
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
    if str(target).endswith('.sqlite') and isinstance(data.get('table'), DiskTable):
        required=data['table'].path.stat().st_size+64*1024**2
        if shutil.disk_usage(target.parent).free < required:
            raise OSError('Insufficient disk for atomic checkpoint; previous checkpoint retained')
    fd, temporary = tempfile.mkstemp(dir=target.parent)
    os.close(fd)
    try:
        if str(target).endswith('.sqlite'):
            _save_sqlite(temporary, data)
        else:
            opener = gzip.open if str(target).endswith('.gz') else open
            with opener(temporary, 'wt') as stream:
                json.dump(data, stream, sort_keys=False, allow_nan=False,
                          default=lambda value: value.tolist() if isinstance(value, np.ndarray) else _unsupported(value))
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


def _unsupported(value):
    raise TypeError(f"Unsupported checkpoint value: {type(value).__name__}")


def _save_sqlite(path, data):
    table = data['table']
    if isinstance(table, DiskTable):
        table.flush()
        connection = sqlite3.connect(path)
        try:
            table.connection.backup(connection)
        finally:
            connection.close()
    else:
        output = DiskTable(path)
        try:
            for key, row in table.items():
                output[key] = row
            output.flush()
        finally:
            output.close()
    metadata = {k:v for k,v in data.items() if k != 'table'}
    metadata['storage_schema'] = 1
    with sqlite3.connect(path) as connection:
        connection.execute('CREATE TABLE IF NOT EXISTS metadata (json TEXT NOT NULL)')
        connection.execute('DELETE FROM metadata')
        connection.execute('INSERT INTO metadata VALUES (?)', (json.dumps(metadata,allow_nan=False),))
