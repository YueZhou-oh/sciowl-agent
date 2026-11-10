"""SQLite state and immutable content-addressed artifacts. One workspace is one trust boundary."""
import hashlib
import json
import os
import sqlite3
import tempfile
import time
import uuid
from pathlib import Path
from contextlib import contextmanager


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


class Store:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "artifacts").mkdir(exist_ok=True)
        (self.root / "work").mkdir(exist_ok=True)
        with self.connect() as db:
            db.executescript('''
            CREATE TABLE IF NOT EXISTS artifacts(id TEXT PRIMARY KEY, path TEXT NOT NULL,
              sha256 TEXT NOT NULL, kind TEXT NOT NULL, metadata TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY, request_key TEXT UNIQUE NOT NULL,
              spec TEXT NOT NULL, status TEXT NOT NULL, result TEXT, error TEXT,
              created REAL NOT NULL, heartbeat REAL, owner TEXT, cancel INTEGER DEFAULT 0);
            CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY, state TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT,
              session_id TEXT NOT NULL, payload TEXT NOT NULL, created REAL NOT NULL);
            ''')

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.root / "state.sqlite", timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        try:
            with db:
                yield db
        finally:
            db.close()

    def put_bytes(self, data, kind, suffix, metadata=None):
        if not suffix.startswith('.') or not suffix[1:].isalnum():
            raise ValueError("Invalid internal suffix")
        digest = hashlib.sha256(data).hexdigest()
        # Include kind and metadata in identity; identical bytes may have distinct provenance.
        aid = "a_" + hashlib.sha256((digest + kind + canonical(metadata or {})).encode()).hexdigest()
        target = self.root / "artifacts" / (aid + suffix)
        fd, tmp = tempfile.mkstemp(dir=target.parent)
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, target)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO artifacts VALUES(?,?,?,?,?)",
                       (aid, target.name, digest, kind, canonical(metadata or {})))
        return aid

    def put_json(self, obj, kind="result", metadata=None):
        return self.put_bytes(canonical(obj).encode(), kind, ".json", metadata)

    def artifact(self, aid, kind=None):
        with self.connect() as db:
            row = db.execute("SELECT * FROM artifacts WHERE id=?", (aid,)).fetchone()
        if row is None or (kind and row["kind"] != kind):
            raise ValueError("Unknown artifact or incorrect artifact type")
        path = (self.root / "artifacts" / row["path"]).resolve()
        if path.parent != (self.root / "artifacts").resolve():
            raise ValueError("Artifact path escaped workspace")
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Artifact integrity check failed")
        return path

    def get_json(self, aid, kind=None):
        return json.loads(self.artifact(aid, kind).read_text())

    def submit(self, spec, request_key):
        encoded = canonical(spec)
        jid = "j_" + uuid.uuid4().hex
        with self.connect() as db:
            db.execute("INSERT OR IGNORE INTO jobs(id,request_key,spec,status,created) VALUES(?,?,?,'queued',?)",
                       (jid, request_key, encoded, time.time()))
            row = db.execute("SELECT * FROM jobs WHERE request_key=?", (request_key,)).fetchone()
            if row["spec"] != encoded:
                raise ValueError("Idempotency key reused with a different request")
        return self.job(row["id"])

    def job(self, jid):
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (jid,)).fetchone()
        if row is None:
            raise ValueError("Unknown job")
        out = dict(row)
        for field in ["spec", "result", "error"]:
            out[field] = json.loads(out[field]) if out[field] else None
        return out

    def claim(self):
        owner = uuid.uuid4().hex
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT id FROM jobs WHERE status='queued' ORDER BY created LIMIT 1").fetchone()
            if row is None:
                return None
            db.execute("UPDATE jobs SET status='running',heartbeat=?,owner=? WHERE id=?",
                       (time.time(), owner, row["id"]))
        return self.job(row["id"])

    def heartbeat(self, jid, owner):
        with self.connect() as db:
            cur = db.execute("UPDATE jobs SET heartbeat=? WHERE id=? AND owner=? AND status='running'",
                             (time.time(), jid, owner))
            if not cur.rowcount:
                raise RuntimeError("Worker no longer owns job")
        if self.job(jid)["cancel"]:
            raise InterruptedError("Cancellation requested")

    def finish(self, jid, owner, status, result=None, error=None):
        with self.connect() as db:
            cur = db.execute("UPDATE jobs SET status=?,result=?,error=?,heartbeat=? WHERE id=? AND owner=? AND status='running'",
                             (status, canonical(result) if result is not None else None,
                              canonical(error) if error else None, time.time(), jid, owner))
            if not cur.rowcount:
                raise RuntimeError("Worker no longer owns job")

    def cancel(self, jid):
        self.job(jid)
        with self.connect() as db:
            db.execute("UPDATE jobs SET cancel=1,status=CASE WHEN status='queued' THEN 'cancelled' ELSE status END WHERE id=?", (jid,))
        return self.job(jid)

    def recover_stale(self, seconds):
        # Never resubmit automatically: a stale worker may still be executing externally.
        with self.connect() as db:
            cur = db.execute("UPDATE jobs SET status='interrupted',owner=NULL,error=? WHERE status='running' AND heartbeat<?",
                (canonical({"code":"STALE_WORKER", "message":"Inspect worker before explicitly submitting a new job"}), time.time()-seconds))
        return cur.rowcount

    def save_session(self, sid, state):
        with self.connect() as db:
            db.execute("INSERT INTO sessions VALUES(?,?) ON CONFLICT(id) DO UPDATE SET state=excluded.state", (sid, canonical(state)))

    def load_session(self, sid):
        with self.connect() as db:
            row = db.execute("SELECT state FROM sessions WHERE id=?", (sid,)).fetchone()
        return json.loads(row[0]) if row else None

    def event(self, sid, payload):
        with self.connect() as db:
            db.execute("INSERT INTO events(session_id,payload,created) VALUES(?,?,?)", (sid, canonical(payload), time.time()))

