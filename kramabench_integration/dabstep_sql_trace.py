"""LEDGER SQL trace type (P1-7, host-agnostic second action class).
The agent answers DABStep questions with a single DuckDB SQL query over the 3 CSV tables
(payments/acquirer_countries/merchant_category_codes). Support determination is then done
in the SQL idiom: cited evidence = tables in FROM/JOIN; citation-restricted re-execution =
re-run the query against a catalog that registers ONLY the cited tables (an uncited table
reference is a catalog miss = fail-stop); A3 counterfactual = perturb the cited tables and
re-run. Verdict/subtype taxonomy identical to the pandas trace. Pure-CPU audit, reader on
12.43. Run: python3 -m kramabench_integration.dabstep_sql_trace --limit 25
"""
from __future__ import annotations
import argparse, json, os, re, time
from pathlib import Path
import duckdb
from openai import OpenAI

REPO = Path(__file__).resolve().parents[1]
CTX = REPO / "datasets" / "dabstep" / "data" / "context"
TABLES = {"payments": "payments.csv", "acquirer_countries": "acquirer_countries.csv",
          "merchant_category_codes": "merchant_category_codes.csv"}
REC_OUT = REPO / "records" / "records_ledger_dabstep_sql.jsonl"
DOC_OUT = REPO / "docs" / "SQL_TRACE.md"
TOL = 1e-4


def _env():
    e = {}
    for line in open(REPO / ".env"):
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.strip().split("=", 1); e[k] = v
    return e


def _reader():
    return OpenAI(api_key=os.environ.get("LEDGER_MODEL_API_KEY", "local-placeholder"),
                  base_url=os.environ.get("LEDGER_MODEL_BASE_URL", "http://127.0.0.1:8000/v1"),
                  timeout=120)


def _preview():
    parts = []
    for t, f in TABLES.items():
        cols = duckdb.sql(f"DESCRIBE SELECT * FROM read_csv_auto('{CTX/f}')").df()
        rows = duckdb.sql(f"SELECT * FROM read_csv_auto('{CTX/f}') LIMIT 3").df()
        parts.append(f"table {t}: columns={list(cols['column_name'])}\n  sample={rows.to_json(orient='records')[:600]}")
    return "\n".join(parts)


SYS = ("You answer a data question with ONE DuckDB SQL query over these tables. "
       "Use table names exactly: payments, acquirer_countries, merchant_category_codes. "
       "Return only a ```sql ...``` block: a single SELECT that computes the final answer "
       "(alias the result column as answer). CAST text columns for arithmetic. No prose.")


def _extract_sql(text):
    m = re.search(r"```sql(.*?)```", text or "", re.DOTALL)
    return (m.group(1) if m else text or "").strip()


def _cited_tables(sql):
    low = sql.lower()
    return sorted({t for t in TABLES if re.search(r"\b" + t + r"\b", low)})


def _run(sql, cited, perturb=None):
    """Run sql in a catalog registering ONLY cited tables. perturb in {None,'num','row'}."""
    con = duckdb.connect()
    for t in cited:
        f = CTX / TABLES[t]
        con.execute(f"CREATE TABLE {t} AS SELECT * FROM read_csv_auto('{f}', ignore_errors=true, sample_size=-1)")
        if perturb:
            cols = con.execute(f"DESCRIBE {t}").df()
            if perturb == "num":
                for _, r in cols.iterrows():
                    if r["column_type"] in ("BIGINT", "DOUBLE", "INTEGER", "FLOAT", "DECIMAL"):
                        con.execute(f'UPDATE {t} SET "{r["column_name"]}" = "{r["column_name"]}" * 1.37')
            else:  # row: drop deterministic 10%
                con.execute(f"CREATE TEMP TABLE _r AS SELECT *, row_number() OVER () rn FROM {t}")
                con.execute(f"DELETE FROM {t}")
                con.execute(f'INSERT INTO {t} SELECT * EXCLUDE(rn) FROM _r WHERE rn % 10 <> 0')
    try:
        df = con.execute(sql).df()
        val = df.iloc[0, 0] if df.shape[0] else None
        return val, None
    except Exception as e:
        msg = str(e)
        kind = "uncited_read" if ("does not exist" in msg or "not found" in msg.lower() or "Catalog" in msg) else \
               ("missing_operand" if "column" in msg.lower() else "exec_error")
        return None, kind
    finally:
        con.close()


def _num(x):
    try:
        return float(str(x).replace(",", "").replace("%", "").strip())
    except Exception:
        return None


def audit(sql, f_claim):
    """Support determination for a SQL trace.

    `f_claim` = the tables the trace INDEPENDENTLY declared as its cited sources. It MUST NOT be parsed
    from the SQL (that would make c1 tautological — the bug documented in
    docs/sqlledger/IMPLEMENTATION_AUDIT.md §4). Restricted replay registers ONLY `f_claim`; any runtime
    reference to a table outside `f_claim` is a catalog miss = c1 (undeclared read).
    """
    referenced = _cited_tables(sql)                # static estimate of F_read (for the c2 'no FROM' check)
    if not referenced:
        return {"label": "unsupported", "subtype": "c2", "recomputed": None, "detail": "no FROM table"}
    val, err = _run(sql, f_claim)                  # restricted catalog = ONLY the declared tables
    if err == "uncited_read":
        undeclared = sorted(set(referenced) - set(f_claim))
        return {"label": "unsupported", "subtype": "c1", "recomputed": None,
                "detail": f"reads table(s) not in F_claim: {undeclared}"}
    if err:
        return {"label": "unknown", "subtype": None, "recomputed": None, "detail": err}
    if val is None:
        return {"label": "unknown", "subtype": None, "recomputed": None, "detail": "no rows"}
    # A3 counterfactual on the DECLARED tables
    nb = _num(val)
    if nb is not None:
        insens = 0
        for mode in ("num", "row"):
            v2, e2 = _run(sql, f_claim, perturb=mode)
            n2 = _num(v2)
            if e2 is None and n2 is not None and abs(n2 - nb) <= TOL * max(1, abs(nb)):
                insens += 1
        if insens == 2:
            return {"label": "unsupported", "subtype": "a3", "recomputed": val, "detail": "counterfactual-insensitive"}
    return {"label": "supported", "subtype": None, "recomputed": val, "detail": f"F_claim={sorted(f_claim)}"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=25); a = ap.parse_args()
    tasks = [json.loads(l) for l in open(REPO / "datasets" / "normalized" / "dabstep_tasks.jsonl")]
    tasks = sorted(tasks, key=lambda t: t["taskid"])[:a.limit]
    cli = _reader(); prev = _preview()
    out = []
    for i, t in enumerate(tasks):
        try:
            r = cli.chat.completions.create(model=os.environ.get("LEDGER_MODEL_NAME", "qwen3.6-35b"), temperature=0, max_tokens=2048,
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": f"{prev}\n\nQuestion: {t['query']}\nOne DuckDB SQL:"}],
                extra_body={"chat_template_kwargs": {"enable_thinking": False}})
            sql = _extract_sql(r.choices[0].message.content)
            # NOTE: DABStep gives the agent no independent source-declaration channel — the only
            # available F_claim is self-declared from the SQL. In this degenerate mode c1 is NOT
            # evaluable (referenced ⊆ F_claim by construction). Valid c1 evidence comes from the
            # SQLLedger layer (experiments/sqlledger), which omits a declared table independently.
            f_claim = _cited_tables(sql)
            val, err = _run(sql, f_claim)
            au = audit(sql, f_claim)
            rec = {"taskid": t["taskid"], "level": t.get("level"), "query": t["query"][:200],
                   "answer": t.get("answer", ""), "sql": sql, "cited_tables": _cited_tables(sql),
                   "f_claim": f_claim, "f_claim_source": "self_declared_from_sql (c1 not evaluable)",
                   "pred": str(val), **{f"g_{k}": v for k, v in au.items()},
                   "grounding_label": au["label"], "unsupported_subtype": au["subtype"]}
            out.append(rec)
            print(f"  {i+1}/{len(tasks)} {t['taskid']}: {au['label']}/{au['subtype']} cited={rec['cited_tables']}", flush=True)
        except Exception as e:
            print(f"  {i+1} {t['taskid']} ERR {str(e)[:80]}", flush=True)
    with open(REC_OUT, "w") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False, default=str) + "\n")
    import collections
    dist = collections.Counter(o["grounding_label"] for o in out)
    sub = collections.Counter(o["unsupported_subtype"] for o in out if o["unsupported_subtype"])
    devs = [o for o in out if str(o.get("answer") or "").strip()]
    dev_sw = [o for o in devs if o["grounding_label"] == "supported"
              and str(o["pred"]).strip().lower() != str(o["answer"]).strip().lower()]
    doc = ["# LEDGER SQL trace type (P1-7): host-agnostic second action class\n",
           "The agent answers with a single DuckDB SQL query; support is audited in the SQL idiom "
           "(cited = FROM/JOIN tables; restricted re-execution = catalog with only cited tables, so an "
           "uncited table reference is a catalog miss = fail-stop c1; A3 = perturb cited tables and re-run). "
           "Same verdict taxonomy as the pandas trace, showing the trace-support semantics is not tied to "
           "one action type.\n",
           f"- tasks: {len(out)}",
           f"- grounding: {dict(dist)}",
           f"- unsupported subtypes: {dict(sub)}",
           f"- dev (answer available): {len(devs)}; supported-but-wrong: {len(dev_sw)}",
           f"\nRecords: records/records_ledger_dabstep_sql.jsonl."]
    DOC_OUT.write_text("\n".join(doc) + "\n")
    print(json.dumps({"n": len(out), "grounding": dict(dist), "subtypes": dict(sub),
                      "dev": len(devs), "dev_supported_wrong": len(dev_sw)}, indent=1))


if __name__ == "__main__":
    main()
