"""LEDGER v2 STEP 3f — build the 40-task stratified HUMAN annotation sheet (labels are Authur's).

Stratified by grounding_label × domain from the fresh STEP-3 DISCOVERY records; every unsupported
class (a/b/c/d) that occurred is included. human_support_label / human_note stay EMPTY — this script
NEVER fills them (protocol; enforced downstream by eval_support_labels.py refusing empty-label runs).

Run: conda run -n kramabench python -m kramabench_integration.make_annotation_sheet
"""
from __future__ import annotations
import os, csv, json, glob, random

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "docs", "annotation")
N_TARGET = 40


def load_records():
    rows = []
    for f in glob.glob(os.path.join(REPO, "records", "records_ledger_n2_kb_*_discovery.jsonl")):
        for l in open(f):
            rows.append(json.loads(l))
    return rows


def main():
    rows = load_records()
    if not rows:
        raise SystemExit("no discovery records yet — run kb_trace_provenance first")
    rng = random.Random(7)
    # strata: every unsupported class first (all occurrences up to 4/class/domain), then supported/unknown
    picked, seen = [], set()

    def take(r):
        if r["taskid"] not in seen:
            seen.add(r["taskid"]); picked.append(r)

    for cls in ("a", "b", "c", "d"):
        for r in rows:
            if r.get("unsupported_class") == cls:
                take(r)
    by_stratum = {}
    for r in rows:
        by_stratum.setdefault((r["grounding_label"], r["domain"]), []).append(r)
    strata = sorted(by_stratum)
    while len(picked) < min(N_TARGET, len(rows)):
        progressed = False
        for k in strata:
            pool = [r for r in by_stratum[k] if r["taskid"] not in seen]
            if pool and len(picked) < N_TARGET:
                take(rng.choice(pool)); progressed = True
        if not progressed:
            break

    os.makedirs(OUT_DIR, exist_ok=True)
    sheet = os.path.join(OUT_DIR, "KB_SUPPORT_ANNOTATION.csv")
    gm = {json.loads(l)["taskid"]: json.loads(l) for l in open(os.path.join(REPO, "datasets/normalized/kramabench_tasks.jsonl"))}
    cols = ["taskid", "domain", "question", "stated_answer", "cited_files", "cited_operands",
            "recomputed_value", "exec_status", "snippet_paths", "checker_label", "unsupported_class",
            "official_score", "human_support_label", "human_note"]
    with open(sheet, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in sorted(picked, key=lambda x: (x["domain"], x["taskid"])):
            det = r.get("support_detail", "")
            w.writerow({
                "taskid": r["taskid"], "domain": r["domain"],
                "question": gm.get(r["taskid"], {}).get("query", "")[:400],
                "stated_answer": str(r.get("pred"))[:120],
                "cited_files": ";".join(r.get("sources_selected", [])[:10]),
                "cited_operands": ";".join(sorted({c for p in r.get("provenance", []) for c in p.get("columns", [])})[:15]),
                "recomputed_value": (det.split("recomputed=")[-1][:60] if "recomputed=" in det else ""),
                "exec_status": ("failed" if r.get("exec_failed") else "ok"),
                "snippet_paths": r.get("trace_path", ""),
                "checker_label": r.get("grounding_label", ""),
                "unsupported_class": r.get("unsupported_class") or "",
                "official_score": r.get("score", -1),
                "human_support_label": "",   # EMPTY by protocol — Authur only
                "human_note": "",            # EMPTY by protocol — Authur only
            })
    readme = os.path.join(OUT_DIR, "ANNOTATION_README.md")
    with open(readme, "w") as f:
        f.write("""# KB support-label annotation protocol (G2)

**Task**: for each row decide `human_support_label` ∈ {yes, no, unclear}:
- **yes** — the cited sources (open `cited_files` under `KramaBench/data/<domain>/`, or read the program
  at `snippet_paths`) contain the rows/operands needed to derive `stated_answer` via the claimed
  operation. Support ≠ correctness: a WRONG answer that genuinely comes from the cited data is still *yes*.
- **no** — the cited sources cannot yield the stated answer (wrong file, missing operands, value not
  derivable, or the program's own recomputation contradicts the stated answer).
- **unclear** — cannot decide (note why in `human_note`).

Label **blind to the gold answer** where possible (the sheet omits gold; `official_score` is present
only for the secondary correctness analysis — ignore it while labeling).
`checker_label`/`unsupported_class` are the machine's outputs — you are auditing them, not copying them.
Classes: a=stated≠recomputed, b=exec-failed-but-answered, c=uncited-source dependency, d=missing operands.

When done: run `python -m kramabench_integration.eval_support_labels` → G2 verdict block.
""")
    print(f"sheet: {sheet} ({len(picked)} rows; strata: " +
          json.dumps({f"{k[0]}/{k[1]}": len(v) for k, v in by_stratum.items()}) + ")")
    print(f"readme: {readme}")
    print("human columns EMPTY — labels are Authur's.")


if __name__ == "__main__":
    main()
