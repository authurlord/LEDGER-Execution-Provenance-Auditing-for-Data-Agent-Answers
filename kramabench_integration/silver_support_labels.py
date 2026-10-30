"""DeepSeek silver support labels for KB annotation triage.

These labels are for pre-annotation triage only. They are not human gold, are not
used as ground truth, and never enter LEDGER's verdict path.

Smoke:
  python3 -m kramabench_integration.silver_support_labels --mode smoke --limit 3

Full:
  python3 -m kramabench_integration.silver_support_labels --mode full
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
import json
import os
import re
import time
from typing import Any

import requests

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(REPO, ".env")
IN_CSV = os.path.join(REPO, "docs", "annotation", "KB_SUPPORT_ANNOTATION.csv")
OUT_JSONL = os.path.join(REPO, "records", "kb_silver_support_labels.jsonl")
OUT_CSV = os.path.join(REPO, "docs", "annotation", "KB_SUPPORT_ANNOTATION_SILVER.csv")

DEFAULT_MODEL = "deepseek-v4-flash"
LABELS = ("supported", "unsupported", "unknown")
REPEATS = 3

PRICE_PER_1M_INPUT_USD = 0.14
PRICE_PER_1M_OUTPUT_USD = 0.28

NOTICE = (
    "LLM silver PRE-annotation for triage only; NOT human gold; NOT used as ground truth "
    "or in any verdict/metric; human_support_label to be filled by human annotator (Authur)."
)


def read_env(path=ENV_PATH):
    vals = {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"missing env file: {path}")
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            vals[k.strip()] = v.strip().strip('"').strip("'")
    missing = [k for k in ("DEEPSEEK_BASE_URL", "DEEPSEEK_API_KEY") if not vals.get(k)]
    if missing:
        raise RuntimeError(f"missing required .env keys: {', '.join(missing)}")
    return vals


def _split_semicolon(value: str) -> list[str]:
    return [x.strip() for x in (value or "").split(";") if x.strip()]


def load_rows(limit: int | None = None):
    with open(IN_CSV, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []
    if limit is not None:
        rows = rows[:limit]
    return rows, fields


def _normalize_label(value: Any) -> str:
    s = str(value or "").strip().lower()
    return s if s in LABELS else "unknown"


def _json_extract(text: str) -> dict:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        raise ValueError("no JSON object found")
    return json.loads(m.group(0))


def build_prompt(row: dict):
    cited = {
        "files": _split_semicolon(row.get("cited_files", "")),
        "columns_or_operands": _split_semicolon(row.get("cited_operands", "")),
        "snippet_paths": _split_semicolon(row.get("snippet_paths", "")),
        "exec_status": row.get("exec_status", ""),
    }
    payload = {
        "taskid": row.get("taskid"),
        "domain": row.get("domain"),
        "question": row.get("question") or "",
        "stated_answer": row.get("stated_answer"),
        "cited_files": cited["files"],
        "cited_columns_or_operands": cited["columns_or_operands"],
        "snippet_paths": cited["snippet_paths"],
        "exec_status": cited["exec_status"],
    }
    return [
        {
            "role": "system",
            "content": (
                "You are assisting a human annotator by producing a preliminary silver "
                "support label for a data-agent answer. Decide whether the stated answer "
                "is supported by the cited files and cited columns/operands. Do not use "
                "or infer any hidden gold answer. Return only JSON with keys "
                "silver_support_label, rationale, confidence. silver_support_label must "
                "be one of supported, unsupported, unknown. confidence must be 0 to 1."
            ),
        },
        {
            "role": "user",
            "content": (
                "Produce a silver support label for triage only. No gold answer is provided.\n\n"
                + json.dumps(payload, ensure_ascii=False, indent=2)
            ),
        },
    ]


def call_silver(env: dict, model: str, row: dict, timeout_s: int):
    base = env["DEEPSEEK_BASE_URL"].rstrip("/")
    url = base if base.endswith("/chat/completions") else base + "/chat/completions"
    payload = {
        "model": model,
        "messages": build_prompt(row),
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {env['DEEPSEEK_API_KEY']}", "Content-Type": "application/json"}
    t0 = time.perf_counter()
    resp = requests.post(url, headers=headers, json=payload, timeout=timeout_s)
    latency = round(time.perf_counter() - t0, 3)
    resp.raise_for_status()
    body = resp.json()
    parsed = _json_extract(body["choices"][0]["message"]["content"])
    label = _normalize_label(parsed.get("silver_support_label") or parsed.get("support_label"))
    try:
        conf = max(0.0, min(1.0, float(parsed.get("confidence"))))
    except Exception:
        conf = 0.0
    usage = body.get("usage") or {}
    tin = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
    tout = int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0)
    cost = (tin / 1_000_000) * PRICE_PER_1M_INPUT_USD + (tout / 1_000_000) * PRICE_PER_1M_OUTPUT_USD
    return {
        "label": label,
        "rationale": str(parsed.get("rationale") or "")[:1200],
        "confidence": conf,
        "tokens_in": tin,
        "tokens_out": tout,
        "latency_s": latency,
        "cost_usd": round(cost, 8),
        "parse_ok": True,
    }


def safe_call(env, model, row, timeout_s):
    try:
        return call_silver(env, model, row, timeout_s)
    except Exception as e:
        return {
            "label": "unknown",
            "rationale": "call_or_parse_error",
            "confidence": 0.0,
            "tokens_in": 0,
            "tokens_out": 0,
            "latency_s": 0.0,
            "cost_usd": 0.0,
            "parse_ok": False,
            "error": str(e)[:300],
        }


def majority(votes: list[dict]):
    counts = Counter(v["label"] for v in votes)
    best_count = max(counts.values()) if counts else 0
    tied = sorted([lab for lab, c in counts.items() if c == best_count])
    # Conservative tie-break: unknown, then unsupported, then supported.
    for lab in ("unknown", "unsupported", "supported"):
        if lab in tied:
            label = lab
            break
    else:
        label = "unknown"
    chosen = [v for v in votes if v["label"] == label]
    conf = sum(v.get("confidence") or 0.0 for v in chosen) / len(chosen) if chosen else 0.0
    rationale = chosen[0].get("rationale", "") if chosen else ""
    return label, rationale, conf, best_count / len(votes) if votes else 0.0


def run(rows, model, repeats, timeout_s):
    env = read_env()
    records = []
    for row in rows:
        votes = []
        for rep in range(repeats):
            out = safe_call(env, model, row, timeout_s)
            votes.append({**out, "repeat": rep})
        label, rationale, conf, agreement = majority(votes)
        record = {
            "notice": NOTICE,
            "taskid": row.get("taskid"),
            "domain": row.get("domain"),
            "question": row.get("question"),
            "stated_answer": row.get("stated_answer"),
            "cited_files": _split_semicolon(row.get("cited_files", "")),
            "cited_operands": _split_semicolon(row.get("cited_operands", "")),
            "checker_label": row.get("checker_label"),
            "llm_silver_label": label,
            "llm_rationale": rationale,
            "llm_confidence": round(conf, 6),
            "llm_agreement_3run": round(agreement, 6),
            "votes": votes,
            "parse_ok": all(v.get("parse_ok") for v in votes),
            "tokens_in": sum(v.get("tokens_in", 0) for v in votes),
            "tokens_out": sum(v.get("tokens_out", 0) for v in votes),
            "cost_usd": round(sum(v.get("cost_usd", 0.0) for v in votes), 8),
        }
        assert "grounding_label" not in record
        records.append(record)
    return records


def summarize(records):
    n = len(records)
    dist = Counter(r["llm_silver_label"] for r in records)
    agree_vals = [r["llm_agreement_3run"] for r in records]
    checker_pairs = Counter((r.get("checker_label") or "unknown", r["llm_silver_label"]) for r in records)
    checker_known = [r for r in records if (r.get("checker_label") or "") in LABELS]
    checker_agree = sum(1 for r in checker_known if r.get("checker_label") == r["llm_silver_label"])
    return {
        "n": n,
        "notice": NOTICE,
        "silver_distribution": dict(dist),
        "mean_3run_agreement": round(sum(agree_vals) / len(agree_vals), 6) if agree_vals else None,
        "full_3of3_agreement": sum(1 for x in agree_vals if x == 1.0),
        "parse_error_votes": sum(1 for r in records for v in r["votes"] if not v.get("parse_ok")),
        "tokens_in": sum(r["tokens_in"] for r in records),
        "tokens_out": sum(r["tokens_out"] for r in records),
        "cost_usd": round(sum(r["cost_usd"] for r in records), 8),
        "checker_label_agreement_triage_only": {
            "n": len(checker_known),
            "agree": checker_agree,
            "rate": round(checker_agree / len(checker_known), 6) if checker_known else None,
            "confusion": {f"{a}->{b}": c for (a, b), c in sorted(checker_pairs.items())},
            "note": "Agreement with LEDGER checker_label is for triage reference only; it is not accuracy.",
        },
    }


def write_outputs(rows, fields, records):
    os.makedirs(os.path.dirname(OUT_JSONL), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_JSONL, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    rec_by_task = {r["taskid"]: r for r in records}
    out_fields = list(fields) + ["llm_silver_label", "llm_rationale", "llm_confidence", "llm_agreement_3run"]
    with open(OUT_CSV, "w", newline="") as f:
        f.write("# " + NOTICE + "\n")
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for row in rows:
            rec = rec_by_task[row.get("taskid")]
            out = dict(row)
            out["human_support_label"] = row.get("human_support_label", "")
            out["llm_silver_label"] = rec["llm_silver_label"]
            out["llm_rationale"] = rec["llm_rationale"]
            out["llm_confidence"] = rec["llm_confidence"]
            out["llm_agreement_3run"] = rec["llm_agreement_3run"]
            writer.writerow(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--timeout-s", type=int, default=60)
    args = ap.parse_args(argv)
    limit = args.limit if args.limit is not None else (3 if args.mode == "smoke" else None)
    rows, fields = load_rows(limit=limit)
    records = run(rows, args.model, args.repeats, args.timeout_s)
    summary = summarize(records)
    if args.mode == "full":
        all_rows, all_fields = load_rows(limit=None)
        if len(all_rows) != len(records):
            # Full mode should run all rows; this guard prevents accidental partial files.
            raise RuntimeError(f"full mode expected {len(all_rows)} rows, got {len(records)}")
        write_outputs(all_rows, all_fields, records)
    print(json.dumps({
        "mode": args.mode,
        "rows": len(records),
        "summary": summary,
        "outputs": {"jsonl": os.path.relpath(OUT_JSONL, REPO), "csv": os.path.relpath(OUT_CSV, REPO)} if args.mode == "full" else None,
    }, ensure_ascii=False, indent=1))
    return summary


if __name__ == "__main__":
    main()
