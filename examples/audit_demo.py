"""Small examples using the recovered LEDGER components; no model or dataset needed."""
from pathlib import Path
import json
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.sqlledger.runtime.sqlite_reads import capture_reads
from experiments.sqlledger.runtime.restricted_replay import restricted_replay
from kramabench_integration.provenance import ConstrainedPool


def main():
    con = sqlite3.connect(":memory:")
    try:
        con.execute("CREATE TABLE sales(value REAL)")
        con.executemany("INSERT INTO sales VALUES(?)", [(100,), (200,), (300,)])
        con.execute("CREATE TABLE cache(value REAL)")
        con.execute("INSERT INTO cache VALUES(200)")
        con.commit()
        declared = ["sales"]
        cases = []
        for name, sql, stated in [
            ("average_from_declared_sales", "SELECT AVG(value) FROM sales", 200),
            ("right_number_from_undeclared_cache", "SELECT value FROM cache", 200),
            ("reproducible_sum_for_an_average_question", "SELECT SUM(value) FROM sales", 600),
        ]:
            _, reads, error = capture_reads(con, sql)
            host_authorized = set(reads.read_tables()).issubset(declared)
            replay = restricted_replay(con, sql, f_claim=declared)
            match = replay.status == "success_comparable" and replay.scalar == stated
            cases.append({
                "case": name,
                "declared_tables": declared,
                "observed_host_reads": reads.read_tables(),
                "host_reads_authorized": host_authorized,
                "replay_status": replay.status,
                "reported_value": stated,
                "replay_value": replay.scalar,
                "execution_checks_pass": error is None and host_authorized and match,
                "semantic_status": "not assessed by this replay-only demo",
            })
        assert [c["execution_checks_pass"] for c in cases] == [True, False, True]
        pool = ConstrainedPool([600, 3], ops=("sum", "diff", "ratio", "pctchg", "prod"))
        depth = pool.check(200, max_depth=2, percent=False)
        assert depth == "depth1"
        print(json.dumps({"replay_examples": cases, "arithmetic_example": {
            "operands": [600, 3], "target": 200, "matching_depth": depth,
            "interpretation": "bounded derivability; task semantics remain separate",
        }, "status": "PASS"}, indent=2))
    finally:
        con.close()


if __name__ == "__main__":
    main()
