"""Content-addressed LLM step checkpoints for local research workflows.

Cache identity includes provider/model, policy version, prompt and complete input.
Only successfully decoded, role-shaped responses are stored. No credentials.
"""
from __future__ import annotations
import hashlib
import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path

SCHEMA_VERSION = 'paper-checkpoint-v1'


class StepCache:
    def __init__(self,path:Path,ttl_seconds:float=86400):
        self.path=Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.ttl=ttl_seconds
        with self.connect() as conn:
            conn.executescript('''PRAGMA journal_mode=WAL;
              CREATE TABLE IF NOT EXISTS steps(key TEXT PRIMARY KEY,role TEXT NOT NULL,
              response TEXT NOT NULL,checksum TEXT NOT NULL,created REAL NOT NULL);''')

    @contextmanager
    def connect(self):
        conn=sqlite3.connect(self.path,timeout=30)
        try:
            with conn:yield conn
        finally:conn.close()

    def get(self,key):
        with self.connect() as conn:
            row=conn.execute('SELECT response,checksum,created FROM steps WHERE key=?',(key,)).fetchone()
        if not row or time.time()-row[2]>self.ttl:
            return None
        if hashlib.sha256(row[0].encode()).hexdigest()!=row[1]:
            return None
        try:
            value=json.loads(row[0])
            return value if isinstance(value,dict) else None
        except ValueError:
            return None

    def put(self,key,role,response):
        text=json.dumps(response,sort_keys=True,ensure_ascii=False,allow_nan=False)
        with self.connect() as conn:
            conn.execute('INSERT OR REPLACE INTO steps VALUES(?,?,?,?,?)',
                (key,role,text,hashlib.sha256(text.encode()).hexdigest(),time.time()))


def valid_shape(role,response):
    field={'Planner':'queries','Researcher':'claims','Synthesizer':'order','Reviewer':'verdicts'}.get(role)
    if not isinstance(response,dict) or not field or not isinstance(response.get(field),list):return False
    if role=='Planner':return bool(response[field]) and all(isinstance(x,str) and x.strip() for x in response[field])
    if role=='Synthesizer':return all(type(x)==int for x in response[field])
    return all(isinstance(x,dict) for x in response[field])


class CheckpointProvider:
    def __init__(self,provider,cache:StepCache,revision='2026-09-19',event=None):
        self.inner=provider
        self.cache=cache
        self.revision=revision
        self.stats={'hits':0,'misses':0}
        self.lock=threading.Lock()
        self.event=event

    @property
    def usage(self):return self.inner.usage

    def call(self,role,instruction,payload):
        identity={'schema':SCHEMA_VERSION,'provider_class':type(self.inner).__name__,
            'base':getattr(self.inner,'base','fixture'),'model':getattr(self.inner,'model','fixture'),
            'revision':self.revision,'role':role,'instruction':instruction,'payload':payload,
            'generation':{'temperature':0,'max_tokens':1800}}
        key=hashlib.sha256(json.dumps(identity,ensure_ascii=False,sort_keys=True,allow_nan=False).encode()).hexdigest()
        value=self.cache.get(key)
        hit=value is not None and valid_shape(role,value)
        with self.lock:self.stats['hits' if hit else 'misses']+=1
        if self.event:self.event(role,'checkpoint_hit' if hit else 'checkpoint_miss',cache_key=key)
        if hit:return value
        value=self.inner.call(role,instruction,payload)
        if valid_shape(role,value):self.cache.put(key,role,value)
        return value
