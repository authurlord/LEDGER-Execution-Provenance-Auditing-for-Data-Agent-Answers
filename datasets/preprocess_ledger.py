"""LEDGER preprocessing — normalize KramaBench + DABStep + ConvFinQA into ONE task schema.

Output (datasets/normalized/*.jsonl), one record per task:
  {taskid, benchmark, split, domain, query, answer, answer_type,
   gold_sources: [...], data_root: "<abs file-lake dir or ''>", level, meta: {...}}

- answer is null when the benchmark holds it out (DABStep test) -> cost-only axis; accuracy needs answers.
- gold_sources drives source_precision/recall (present for KramaBench; unknown/[] for DABStep's fixed lake).
GPU-free. Run: conda run -n kramabench python datasets/preprocess_ledger.py
"""
import os, json, glob
ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)
OUT = os.path.join(ROOT, "normalized"); os.makedirs(OUT, exist_ok=True)


def _write(name, rows):
    p = os.path.join(OUT, name)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
    print(f"[write] {name}: {len(rows)} tasks -> {p}")
    return len(rows)


def prep_kramabench():
    rows = []
    wl = os.path.join(REPO, "KramaBench", "workload")
    for jf in sorted(glob.glob(os.path.join(wl, "*.json"))):
        base = os.path.basename(jf)
        if base.startswith(("quick-start",)) or "-tiny" in base:
            continue
        domain = base[:-5]
        try:
            tasks = json.load(open(jf))
        except Exception:
            continue
        if not isinstance(tasks, list):
            continue
        droot = os.path.join(REPO, "KramaBench", "data", domain)
        for t in tasks:
            rows.append({
                "taskid": t.get("id"), "benchmark": "kramabench", "split": "full",
                "domain": domain, "query": t.get("query"), "answer": t.get("answer"),
                "answer_type": t.get("answer_type"), "gold_sources": t.get("data_sources", []),
                "data_root": droot, "level": (t.get("id", "").split("-")[1] if "-" in str(t.get("id", "")) else ""),
                "meta": {"n_subtasks": len(t.get("subtasks", [])), "runtime": t.get("runtime")},
            })
    return _write("kramabench_tasks.jsonl", rows)


def prep_dabstep():
    rows = []
    base = os.path.join(ROOT, "dabstep", "data")
    droot = os.path.join(base, "context")
    for split, fn in [("dev", "dev.jsonl"), ("test", "all.jsonl")]:
        fp = os.path.join(base, "tasks", fn)
        if not os.path.exists(fp):
            continue
        for line in open(fp):
            t = json.loads(line)
            ans = t.get("answer")
            ans = ans if (ans is not None and str(ans).strip() != "") else None
            rows.append({
                "taskid": f"dabstep-{t.get('task_id')}", "benchmark": "dabstep",
                "split": split, "domain": "finance-payments", "query": t.get("question"),
                "answer": ans, "answer_type": "factoid",
                "gold_sources": [],   # DABStep does not annotate per-task source; fixed 7-file lake
                "data_root": droot, "level": t.get("level", ""),
                "meta": {"guidelines": t.get("guidelines"),
                         "answers_held_out": ans is None},
            })
    return _write("dabstep_tasks.jsonl", rows)


def prep_convfinqa(sample=60):
    """ConvFinQA (flare) — self-contained numeric multi-step QA; used to SEED a provenance support-label set.
    The context table is inline in `query`; there is no separate file lake. This HF variant lacks the gold
    DSL program (the original czyssrs/ConvFinQA has it) -> support labels here come from re-running the
    inline table, not the gold program. Emit only a small sample for annotation (STEP 4)."""
    import pandas as pd
    vf = sorted(glob.glob(os.path.join(ROOT, "convfinqa", "data", "*valid*")))
    if not vf:
        print("[skip] convfinqa valid parquet not found"); return 0
    df = pd.read_parquet(vf[0]).head(sample)
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "taskid": f"convfinqa-{r['id']}", "benchmark": "convfinqa", "split": "valid",
            "domain": "finance-filings", "query": r["query"], "answer": r["answer"],
            "answer_type": "numeric", "gold_sources": ["inline-table"], "data_root": "",
            "level": "", "meta": {"dialogue_id": int(r["dialogue_id"]), "turn": int(r["turn"]),
                                  "note": "context inline in query; support label via inline-table recompute"},
        })
    return _write("convfinqa_provenance_seed.jsonl", rows)


if __name__ == "__main__":
    n = {}
    n["kramabench"] = prep_kramabench()
    n["dabstep"] = prep_dabstep()
    try:
        n["convfinqa"] = prep_convfinqa()
    except Exception as e:
        print("[warn] convfinqa prep failed:", e); n["convfinqa"] = 0
    print("NORMALIZED TASK COUNTS:", json.dumps(n))
