"""BIRD Mini-Dev adapter (realism stress test).

BIRD Mini-Dev ships multi-table SQLite databases plus gold SQL over dirty, larger, realistic schemas.
Structure (after extraction, paths discovered robustly):
    external/raw/bird/minidev/MINIDEV/mini_dev_sqlite.json   (list of {question, SQL, db_id, ...})
    external/raw/bird/minidev/MINIDEV/dev_databases/<db_id>/<db_id>.sqlite

Run self-test: python3 -m experiments.sqlledger.adapters.bird
"""
from __future__ import annotations
import glob
import json
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
BIRD_ROOT = REPO / "external" / "raw" / "bird"


def _find_json():
    pats = ["minidev/**/mini_dev_sqlite.json", "minidev/**/mini_dev*.json", "**/mini_dev_sqlite.json",
            "**/dev.json"]
    for pat in pats:
        hits = sorted(glob.glob(str(BIRD_ROOT / pat), recursive=True))
        for h in hits:
            if "sqlite" in Path(h).name or "mini" in Path(h).name.lower():
                return Path(h)
    return None


def _find_db_root():
    for pat in ["minidev/**/dev_databases", "**/dev_databases"]:
        hits = sorted(glob.glob(str(BIRD_ROOT / pat), recursive=True))
        if hits:
            return Path(hits[0])
    return None


def db_path(db_id: str):
    root = _find_db_root()
    if not root:
        return ""
    return str(root / db_id / f"{db_id}.sqlite")


@dataclass
class Parent:
    parent_id: str
    dataset: str
    split: str
    db_id: str
    db_path: str
    question: str
    sql: str
    sql_struct: dict = field(default_factory=dict)
    header: list = field(default_factory=list)
    types: list = field(default_factory=list)


class BIRDDataset:
    supports_distractor = False

    def __init__(self, split="mini_dev", root: Path = BIRD_ROOT):
        self.split = split
        self.root = Path(root)
        self._json = _find_json()
        if self._json is None:
            raise FileNotFoundError(f"BIRD Mini-Dev json not found under {BIRD_ROOT}")

    def iter_parents(self, limit=None):
        data = json.load(open(self._json))
        n = 0
        for i, q in enumerate(data):
            dbid = q.get("db_id")
            sql = q.get("SQL") or q.get("query") or q.get("sql")
            if not dbid or not sql:
                continue
            p = db_path(dbid)
            if not p or not Path(p).exists():
                continue
            yield Parent(parent_id=f"bird/{self.split}/{i}", dataset="bird", split=self.split,
                         db_id=dbid, db_path=p, question=q.get("question", ""), sql=sql)
            n += 1
            if limit and n >= limit:
                return


def _selftest():
    if _find_json() is None:
        print("SKIP bird selftest: data absent"); return
    import sqlite3
    from ..runtime.sqlite_reads import capture_reads
    ds = BIRDDataset()
    ok = multi = err = total = 0
    con_cache = {}
    for p in ds.iter_parents(limit=100):
        total += 1
        con = con_cache.get(p.db_path) or sqlite3.connect(p.db_path); con_cache[p.db_path] = con
        rows, cap, e = capture_reads(con, p.sql)
        if e is None:
            ok += 1
            if len(cap.read_tables()) >= 2:
                multi += 1
        else:
            err += 1
    for c in con_cache.values():
        c.close()
    print(f"bird mini-dev: {ok}/{total} executed, {multi} multi-table, {err} errored")
    print("bird selftest OK" if ok >= max(1, total // 2) else "bird selftest WEAK")


if __name__ == "__main__":
    _selftest()
