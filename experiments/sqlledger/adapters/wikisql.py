"""WikiSQL adapter.

Loads WikiSQL (official salesforce release) and turns each {question, structured sql, table} triple
into an executable SQLite query over the shipped ``{split}.db``. The SQL string is built to match the
official ``lib/dbengine.py`` semantics (lower-cased string condition values, ``col<i>`` columns,
``table_<id>`` tables, agg/cond op vocab). NOTE: a formal cross-check against the official WikiSQL
``DBEngine``/denotation evaluator is a Phase-1 TODO; the selftest currently asserts executability,
not denotation equivalence.

Everything is a single literal SQL string (values inlined + safely quoted) so the same string flows
through capture / restricted replay / mutation without a separate params channel.

Layout expected (downloaded, not in git):
    external/raw/wikisql/data/{train,dev,test}.{jsonl,tables.jsonl,db}

Run self-test: python3 -m experiments.sqlledger.adapters.wikisql
"""
from __future__ import annotations
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
WIKI = REPO / "external" / "raw" / "wikisql" / "data"

AGG_OPS = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
COND_OPS = ["=", ">", "<", "OP"]


def table_name(table_id: str) -> str:
    return "table_" + table_id.replace("-", "_")


def _lit(val, lower=True):
    """Inline a condition value as a SQL literal, matching dbengine (lower-cased strings)."""
    if isinstance(val, str):
        v = val.lower() if lower else val
        return "'" + v.replace("'", "''") + "'"
    # numeric literal
    return repr(val)


def build_sql(sql_struct: dict, table_id: str, lower=True) -> str:
    tbl = table_name(table_id)
    sel = f"col{sql_struct['sel']}"
    agg = AGG_OPS[sql_struct["agg"]]
    select = f"{agg}({sel})" if agg else sel
    conds = []
    for col_idx, op, val in sql_struct.get("conds", []):
        conds.append(f"col{col_idx} {COND_OPS[op]} {_lit(val, lower)}")
    where = (" WHERE " + " AND ".join(conds)) if conds else ""
    return f'SELECT {select} FROM "{tbl}"{where}'


@dataclass
class Parent:
    parent_id: str
    dataset: str
    split: str
    db_id: str            # the table id (single-table)
    table: str            # sqlite table name
    question: str
    sql_struct: dict
    sql: str              # executable SQL
    header: list
    types: list
    db_path: str = ""     # source sqlite (the split .db); set in iter_parents


class WikiSQLDataset:
    supports_distractor = True

    def __init__(self, split: str, root: Path = WIKI):
        self.split = split
        self.root = Path(root)
        self.db_path = str(self.root / f"{split}.db")
        self._tables = None

    def db(self):
        return sqlite3.connect(self.db_path)

    def tables(self):
        if self._tables is None:
            self._tables = {}
            with open(self.root / f"{self.split}.tables.jsonl") as f:
                for line in f:
                    t = json.loads(line)
                    self._tables[t["id"]] = t
        return self._tables

    def iter_parents(self, limit=None):
        tabs = self.tables()
        n = 0
        with open(self.root / f"{self.split}.jsonl") as f:
            for i, line in enumerate(f):
                q = json.loads(line)
                tid = q["table_id"]
                meta = tabs.get(tid, {})
                yield Parent(
                    parent_id=f"wikisql/{self.split}/{i}", dataset="wikisql", split=self.split,
                    db_id=tid, table=table_name(tid), question=q["question"],
                    sql_struct=q["sql"], sql=build_sql(q["sql"], tid),
                    header=meta.get("header", []), types=meta.get("types", []),
                    db_path=self.db_path)
                n += 1
                if limit and n >= limit:
                    return


def _selftest():
    if not (WIKI / "dev.db").exists():
        print("SKIP wikisql selftest: data absent"); return
    ds = WikiSQLDataset("dev")
    con = ds.db()
    ok = 0
    for p in ds.iter_parents(limit=50):
        try:
            rows = con.execute(p.sql).fetchall()
            ok += 1
        except Exception as e:
            print("FAIL", p.parent_id, p.sql, e)
    con.close()
    assert ok >= 45, f"only {ok}/50 WikiSQL queries executed"
    # show one built SQL
    p = next(WikiSQLDataset("dev").iter_parents(limit=1))
    print("sample:", p.sql)
    print(f"wikisql selftest OK ({ok}/50 executed)")


if __name__ == "__main__":
    _selftest()
