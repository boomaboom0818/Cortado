"""Bounded-memory CFR table. Exact float64 rows, SQLite backing, bounded LRU.

The solver mutates returned arrays; mutable cache entries are flushed on eviction.
Only the single owner process may use a working table. Checkpoints are read-only.
"""
from collections import OrderedDict
from collections.abc import MutableMapping
from pathlib import Path
import sqlite3
import numpy as np


class DiskTable(MutableMapping):
    def __init__(self, path, *, readonly=False, cache_rows=4096):
        if cache_rows < 1:
            raise ValueError('cache_rows must be positive')
        self.path = Path(path)
        self.readonly, self.cache_rows = readonly, cache_rows
        self.cache = OrderedDict()
        if readonly:
            self.connection = sqlite3.connect(self.path.resolve().as_uri()+'?mode=ro',uri=True)
        else:
            self.connection = sqlite3.connect(self.path)
            self.connection.execute('PRAGMA journal_mode=DELETE')
            self.connection.execute('CREATE TABLE IF NOT EXISTS states (key TEXT PRIMARY KEY, n INTEGER NOT NULL, values_blob BLOB NOT NULL) WITHOUT ROWID')
        self.connection.execute('PRAGMA cache_size=-8192')
        self.connection.execute('PRAGMA mmap_size=0')
        self.hits = self.misses = 0

    def _write(self,key,row):
        values=np.asarray(row,dtype='<f8')
        if values.ndim!=2 or values.shape[0]!=2 or values.shape[1]<1 or not np.isfinite(values).all() or (values[1]<0).any():
            raise ValueError('Invalid CFR row')
        self.connection.execute('INSERT OR REPLACE INTO states VALUES (?,?,?)',(key,values.shape[1],values.tobytes()))

    def _cache(self,key,row):
        self.cache[key]=row
        self.cache.move_to_end(key)
        if len(self.cache)>self.cache_rows:
            old_key,old_row=self.cache.popitem(last=False)
            if not self.readonly:
                self._write(old_key,old_row)

    def __getitem__(self,key):
        if key in self.cache:
            self.hits+=1
            self.cache.move_to_end(key)
            return self.cache[key]
        self.misses+=1
        record=self.connection.execute('SELECT n, values_blob FROM states WHERE key=?',(key,)).fetchone()
        if record is None:
            raise KeyError(key)
        n,blob=record
        if n<1 or len(blob)!=16*n:
            raise ValueError('Corrupt CFR row')
        values=np.frombuffer(blob,dtype='<f8').reshape(2,n).copy()
        if not np.isfinite(values).all() or (values[1]<0).any():
            raise ValueError('Invalid CFR values')
        row=[values[0],values[1]]
        self._cache(key,row)
        return row

    def __setitem__(self,key,row):
        if self.readonly:
            raise TypeError('Checkpoint table is read-only')
        # Keep the same arrays the solver allocated, so in-place updates persist.
        self._cache(key,row)

    def __delitem__(self,key):
        if self.readonly:
            raise TypeError('Checkpoint table is read-only')
        self.cache.pop(key,None)
        self.connection.execute('DELETE FROM states WHERE key=?',(key,))

    def __iter__(self):
        self.flush()
        for row in self.connection.execute('SELECT key FROM states ORDER BY key'):
            yield row[0]

    def __len__(self):
        self.flush()
        return self.connection.execute('SELECT count(*) FROM states').fetchone()[0]

    def flush(self):
        if not self.readonly:
            for key,row in self.cache.items():
                self._write(key,row)
            self.connection.commit()

    def close(self):
        self.flush()
        self.connection.close()

    def __del__(self):
        try:
            self.connection.close()
        except Exception:
            pass
