"""Spider 1.0 adapter (multi-table complexity layer).

Each Spider example is {db_id, question, query (gold SQL)} over a real multi-table SQLite database
at ``database/<db_id>/<db_id>.sqlite``. The gold SQL is executable directly; multi-table joins are
what let us construct a GENUINE partial-omission c1 (declare all read tables but one).

Layout (downloaded, not in git):
    external/raw/spider/spider/{dev.json,train_spider.json,tables.json,database/<db_id>/<db_id>.sqlite}

Run self-test: python3 -m experiments.sqlledger.adapters.spider
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SPIDER = REPO / "external" / "raw" / "spider" / "spider"


def db_path(db_id: str) -> str:
    return str(SPIDER / "database" / db_id / f"{db_id}.sqlite")


@dataclass
class Parent:
    parent_id: str
    dataset: str
    split: str
    db_id: str
    db_path: str
    question: str
    sql: str
    sql_struct: dict = field(default_factory=dict)   # unused for Spider (kept for interface parity)
    header: list = field(default_factory=list)
    types: list = field(default_factory=list)


class SpiderDataset:
    supports_distractor = False   # cross-db schema-compatible distractors are out of scope here

    def __init__(self, split: str = "dev", root: Path = SPIDER):
        self.split = split
        self.root = Path(root)
        self._file = "dev.json" if split == "dev" else f"{split}.json"

    def iter_parents(self, limit=None):
        data = json.load(open(self.root / self._file))
        n = 0
        for i, q in enumerate(data):
            dbid = q["db_id"]
            p = db_path(dbid)
            if not Path(p).exists():
                continue
            yield Parent(parent_id=f"spider/{self.split}/{i}", dataset="spider", split=self.split,
                         db_id=dbid, db_path=p, question=q["question"], sql=q["query"])
            n += 1
            if limit and n >= limit:
                return


def _selftest():
    if not (SPIDER / "dev.json").exists():
        print("SKIP spider selftest: data absent"); return
    import sqlite3
    from ..runtime.sqlite_reads import capture_reads
    ds = SpiderDataset("dev")
    ok = 0; multi = 0; err = 0; total = 0
    con_cache = {}
    for p in ds.iter_parents(limit=100):
        total += 1
        con = con_cache.get(p.db_path) or sqlite3.connect(p.db_path)
        con_cache[p.db_path] = con
        rows, cap, e = capture_reads(con, p.sql)
        if e is None:
            ok += 1
            if len(cap.read_tables()) >= 2:
                multi += 1
        else:
            err += 1
    for c in con_cache.values():
        c.close()
    print(f"spider dev: {ok}/{total} executed, {multi} multi-table, {err} errored")
    assert ok >= 80, f"only {ok}/{total} Spider queries executed"
    print("spider selftest OK")


if __name__ == "__main__":
    _selftest()
