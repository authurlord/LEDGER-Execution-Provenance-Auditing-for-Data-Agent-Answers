"""LEDGER DABStep support-determination run.

Selection-free small-lake setting: every task is offered the fixed DABStep
finance/payments context (7 files), the reader writes a complete Python program,
and LEDGER audits support deterministically with no LLM calls.

Run:
  conda run -n kramabench python -m kramabench_integration.dabstep_support_run --limit 3 --smoke
  conda run -n kramabench python -m kramabench_integration.dabstep_support_run --limit 60
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

try:
    from . import cost_accounting as CA
except ImportError:  # direct script execution fallback
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from kramabench_integration import cost_accounting as CA


REPO = Path(__file__).resolve().parents[1]
TASKS_PATH = REPO / "datasets" / "normalized" / "dabstep_tasks.jsonl"
CONTEXT_DIR = REPO / "datasets" / "dabstep" / "data" / "context"
RECORDS_OUT = REPO / "records" / "records_ledger_dabstep_support.jsonl"
DOC_OUT = REPO / "docs" / "DABSTEP_SUPPORT.md"
KENV_PY = os.environ.get("KENV_PY") or (
    "/home/yanmy/miniforge3/envs/kramabench/bin/python"
    if Path("/home/yanmy/miniforge3/envs/kramabench/bin/python").exists()
    else sys.executable
)

BASE_URL = os.environ.get("LEDGER_MODEL_BASE_URL", "http://127.0.0.1:8000/v1")
API_KEY = os.environ.get("LEDGER_MODEL_API_KEY", "local-placeholder")
MODEL = os.environ.get("LEDGER_MODEL_NAME", "qwen3.6-35b")
ALLOWED_FILES = {
    "payments.csv",
    "fees.json",
    "merchant_data.json",
    "manual.md",
    "acquirer_countries.csv",
    "merchant_category_codes.csv",
    "payments-readme.md",
}
READ_FUNCS = {
    "open",
    "read_csv",
    "read_json",
    "read_excel",
    "read_table",
    "read_parquet",
    "read_html",
}
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
CODE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.S | re.I)


def _scrubbed_env() -> dict[str, str]:
    """Env for model-written code with secrets removed."""
    safe = {}
    for k, v in os.environ.items():
        ku = k.upper()
        if any(s in ku for s in ("KEY", "TOKEN", "SECRET", "PASSWORD", "OPENAI", "DEEPSEEK", "ANTHROPIC")):
            continue
        safe[k] = v
    safe["HOME"] = "/tmp"
    return safe


def _json_default(x: Any) -> str:
    return str(x)


def _last_json_obj(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    dec = json.JSONDecoder()
    last = None
    idx = text.find("{")
    while idx != -1:
        try:
            obj, _ = dec.raw_decode(text[idx:])
            if isinstance(obj, dict):
                last = obj
        except Exception:
            pass
        idx = text.find("{", idx + 1)
    return last


def _answer_from_stdout(stdout: str) -> Any:
    obj = _last_json_obj(stdout)
    if isinstance(obj, dict) and "answer" in obj:
        return obj.get("answer")
    return None


def _extract_code(text: str) -> str:
    match = CODE_RE.search(text or "")
    code = match.group(1) if match else (text or "")
    code = code.strip()
    if code.startswith("python\n"):
        code = code[len("python\n") :]
    return code


def _call_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


def _literal_key(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _static_string_expr(node: ast.AST, env: dict[str, Any] | None = None) -> str | None:
    env = env or {}
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name) and isinstance(env.get(node.id), str):
        return env[node.id]
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        mapping = env.get(node.value.id)
        key = _literal_key(node.slice)
        if isinstance(mapping, dict) and key in mapping:
            return mapping[key]
    if isinstance(node, ast.JoinedStr):
        parts = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            else:
                return None
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _static_string_expr(node.left, env)
        right = _static_string_expr(node.right, env)
        return left + right if left is not None and right is not None else None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "join":
        parts = [_static_string_expr(arg, env) for arg in node.args]
        if all(part is not None for part in parts):
            return os.path.join(*parts)  # type: ignore[arg-type]
    return None


def _basename_if_allowed(pathish: str | None) -> str | None:
    if not pathish:
        return None
    base = os.path.basename(str(pathish))
    return base if base in ALLOWED_FILES else None


def files_from_ast(code: str) -> tuple[list[str], list[str]]:
    """Return (files_cited, files_read) from static AST inspection."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return [], []

    env: dict[str, Any] = {}
    all_literals: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            all_literals.append(node.value)
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            value = _static_string_expr(node.value, env)
            if value:
                env[name] = value
            elif isinstance(node.value, ast.Dict):
                mapping = {}
                for key_node, val_node in zip(node.value.keys, node.value.values):
                    key = _literal_key(key_node) if key_node is not None else None
                    val = _static_string_expr(val_node, env)
                    if key and val:
                        mapping[key] = val
                if mapping:
                    env[name] = mapping

    files_cited = sorted({base for lit in all_literals for base in [_basename_if_allowed(lit)] if base})
    files_read = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        if _call_name(node.func) not in READ_FUNCS:
            continue
        base = _basename_if_allowed(_static_string_expr(node.args[0], env))
        if base:
            files_read.append(base)
    return files_cited, sorted(set(files_read))


def build_context_preview() -> str:
    import pandas as pd

    parts = []
    payments = pd.read_csv(CONTEXT_DIR / "payments.csv", nrows=5)
    parts.append(
        "payments.csv\n"
        f"- columns={list(payments.columns)}\n"
        f"- dtypes={json.dumps({c: str(t) for c, t in payments.dtypes.items()}, ensure_ascii=False)}\n"
        f"- sample_rows={payments.head(4).to_json(orient='records')}"
    )
    for csv_name in ["acquirer_countries.csv", "merchant_category_codes.csv"]:
        df = pd.read_csv(CONTEXT_DIR / csv_name, nrows=5)
        parts.append(
            f"{csv_name}\n"
            f"- columns={list(df.columns)}\n"
            f"- dtypes={json.dumps({c: str(t) for c, t in df.dtypes.items()}, ensure_ascii=False)}\n"
            f"- sample_rows={df.head(4).to_json(orient='records')}"
        )
    for json_name in ["fees.json", "merchant_data.json"]:
        obj = json.load(open(CONTEXT_DIR / json_name))
        sample = obj[:3] if isinstance(obj, list) else obj
        parts.append(
            f"{json_name}\n"
            f"- top_level={type(obj).__name__}, length={len(obj) if hasattr(obj, '__len__') else 'NA'}\n"
            f"- sample={json.dumps(sample, ensure_ascii=False)[:5000]}"
        )
    for md_name in ["manual.md", "payments-readme.md"]:
        text = open(CONTEXT_DIR / md_name, errors="ignore").read()
        headings = [line.strip() for line in text.splitlines() if line.lstrip().startswith("#")]
        bullets = [line.strip() for line in text.splitlines() if line.strip().startswith("- **")]
        parts.append(
            f"{md_name}\n"
            f"- headings={headings[:20]}\n"
            f"- key_bullets={bullets[:30]}"
        )
    return "\n\n".join(parts)


def reader_program(query: str, guidelines: str, preview: str, max_tokens: int = 4096) -> tuple[str, Any, float]:
    from openai import OpenAI

    cli = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=180)
    system = (
        "You are a careful data analyst. Write a complete Python program using pandas/json. "
        "The program must read only files in the current working directory and compute the answer. "
        "Stdout must contain exactly one line: json.dumps({'answer': ...}). No prose, no markdown outside code."
    )
    user = (
        f"Question:\n{query}\n\n"
        f"Answer instructions:\n{guidelines or 'Return the answer only.'}\n\n"
        "Available files are all offered; do not do retrieval. Use these exact local filenames:\n"
        + "\n".join(f"- {name}" for name in sorted(ALLOWED_FILES))
        + "\n\nSchema/head previews:\n"
        + preview
        + "\n\nReturn one complete Python program. It must print exactly json.dumps({'answer': answer})."
    )
    t0 = time.perf_counter()
    resp = cli.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0,
        max_tokens=max_tokens,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    return resp.choices[0].message.content or "", resp.usage, time.perf_counter() - t0


def exec_program(code: str, cwd: Path, timeout: int = 120) -> dict[str, Any]:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False, dir="/tmp") as handle:
        handle.write(code)
        tmp = handle.name
    try:
        proc = subprocess.run(
            [KENV_PY, tmp],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(cwd),
            env=_scrubbed_env(),
        )
        return {
            "stdout": proc.stdout or "",
            "stderr": proc.stderr or "",
            "returncode": proc.returncode,
            "timeout": False,
            "answer": _answer_from_stdout(proc.stdout or ""),
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "stdout": exc.stdout or "",
            "stderr": "TIMEOUT",
            "returncode": None,
            "timeout": True,
            "answer": None,
        }
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass


def reexec_hermetic(code: str, files_cited: list[str], timeout: int = 120) -> tuple[Any, str | None, str]:
    tmpdir = Path(tempfile.mkdtemp(prefix="dabstep_ledger_"))
    try:
        mirror = tmpdir / "context"
        mirror.mkdir(parents=True, exist_ok=True)
        for name in files_cited:
            if name in ALLOWED_FILES:
                shutil.copy2(CONTEXT_DIR / name, mirror / name)
        rewritten = code.replace(str(CONTEXT_DIR), str(mirror))
        rewritten = rewritten.replace("/mnt/data/yanmy/LakeOrch/datasets/dabstep/data/context", str(mirror))
        result = exec_program(rewritten, mirror, timeout=timeout)
        stderr = result["stderr"] or ""
        if result["timeout"]:
            return None, "timeout", stderr
        if result["returncode"] != 0 or stderr.strip():
            if "FileNotFoundError" in stderr or "No such file" in stderr:
                return None, "uncited_read", stderr[-500:]
            lowered = stderr.lower()
            if "keyerror" in lowered or ("columns" in lowered and "not" in lowered):
                return None, "missing_operand", stderr[-500:]
            return None, "exec_error", stderr[-500:]
        if result["answer"] is None:
            return None, "no_output", (result["stdout"] or "")[-500:]
        return result["answer"], None, ""
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _norm_num(value: Any) -> float | None:
    if value is None:
        return None
    match = NUM_RE.search(str(value).replace(",", "").replace("%", ""))
    return float(match.group()) if match else None


def _norm_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def boundary_contains(container: Any, needle: Any) -> bool:
    c = _norm_text(container)
    n = _norm_text(needle)
    if not c or not n:
        return False
    if c == n:
        return True
    return re.search(r"(?<![0-9a-z_])" + re.escape(n) + r"(?![0-9a-z_])", c) is not None


def values_match(stated: Any, recomputed: Any, tol: float = 1e-4) -> bool:
    sn = _norm_num(stated)
    rn = _norm_num(recomputed)
    if sn is not None and rn is not None:
        return abs(sn - rn) <= max(tol * abs(sn), 1e-9)
    if isinstance(stated, (list, tuple)) or isinstance(recomputed, (list, tuple)):
        return _norm_text(json.dumps(stated, sort_keys=True, default=_json_default)) == _norm_text(
            json.dumps(recomputed, sort_keys=True, default=_json_default)
        )
    return boundary_contains(recomputed, stated) or boundary_contains(stated, recomputed)


def support_label(stated: Any, initial_exec: dict[str, Any], code: str) -> tuple[str, str | None, str]:
    files_cited, files_read = files_from_ast(code)
    if stated in (None, ""):
        return "unknown", None, "no stated answer"
    if initial_exec.get("returncode") not in (0, None) and stated is not None:
        return "unsupported", "b", "initial program had execution errors but produced an answer"
    if initial_exec.get("stderr") and str(initial_exec.get("stderr")).strip() and stated is not None:
        return "unsupported", "b", "initial program wrote stderr but produced an answer"
    if not files_cited:
        return "unsupported", "c2", "no DABStep context file cited/read by the program"
    read_not_cited = sorted(set(files_read) - set(files_cited))
    if read_not_cited:
        return "unsupported", "c1", "program read files outside cited set: " + ", ".join(read_not_cited)

    recomputed, err, detail = reexec_hermetic(code, files_cited)
    if err == "uncited_read":
        return "unsupported", "c1", detail
    if err == "missing_operand":
        return "unsupported", "d", detail
    if err in {"exec_error", "timeout", "no_output"}:
        return "unsupported", "a2", f"restricted re-exec {err}: {detail[:250]}"
    if recomputed is None:
        return "unsupported", "a2", "restricted re-exec produced no comparable answer"
    if values_match(stated, recomputed):
        return "supported", None, f"stated={stated!r} recomputed={recomputed!r}"
    return "unsupported", "a1", f"stated={stated!r} recomputed={recomputed!r}"


def official_score(pred: Any, gold: Any) -> float | None:
    if gold in (None, ""):
        return None
    if pred is None:
        return 0.0
    return 1.0 if values_match(pred, gold, tol=1e-4) else 0.0


def load_tasks() -> list[dict[str, Any]]:
    return [json.loads(line) for line in open(TASKS_PATH)]


def select_tasks(limit: int, seed: int, smoke: bool = False) -> list[dict[str, Any]]:
    rows = load_tasks()
    if smoke:
        return rows[:limit]
    dev = [r for r in rows if r.get("answer") is not None]
    selected = {r["taskid"]: r for r in dev}
    remaining = [r for r in rows if r["taskid"] not in selected]
    rng = random.Random(seed)
    target_easy = limit // 2
    target_hard = limit - target_easy

    def fill(level: str, target: int) -> list[dict[str, Any]]:
        have = sum(1 for r in selected.values() if r.get("level") == level)
        need = max(0, target - have)
        pool = [r for r in remaining if r.get("level") == level]
        pool = sorted(pool, key=lambda x: x["taskid"])
        if len(pool) <= need:
            return pool
        return rng.sample(pool, need)

    for row in fill("easy", target_easy) + fill("hard", target_hard):
        selected[row["taskid"]] = row
    if len(selected) < limit:
        pool = [r for r in remaining if r["taskid"] not in selected]
        pool = sorted(pool, key=lambda x: x["taskid"])
        need = min(limit - len(selected), len(pool))
        for row in rng.sample(pool, need):
            selected[row["taskid"]] = row
    return sorted(selected.values(), key=lambda x: (x.get("answer") is None, x.get("level") != "easy", x["taskid"]))[:limit]


def one_task(task: dict[str, Any], preview: str) -> dict[str, Any]:
    meter = CA.CostMeter()
    raw, usage, reader_latency = reader_program(task["query"], task.get("meta", {}).get("guidelines", ""), preview)
    meter.add_llm(usage).turn()
    code = _extract_code(raw)
    initial = exec_program(code, CONTEXT_DIR, timeout=120)
    meter.tool()
    pred = initial["answer"]
    files_cited, files_read = files_from_ast(code)
    label, subtype, detail = support_label(pred, initial, code)
    score = official_score(pred, task.get("answer"))
    trace_dir = REPO / "records" / "dabstep_support_traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"{task['taskid']}.py"
    trace_path.write_text(code, encoding="utf-8")
    rec = {
        "taskid": task["taskid"],
        "level": task.get("level"),
        "query": task["query"],
        "pred": pred,
        "answer": task.get("answer"),
        "grounding_label": label,
        "unsupported_subtype": subtype,
        "files_cited": files_cited,
        "files_read": files_read,
        "tokens": {"in": meter.tokens_in, "out": meter.tokens_out},
        "latency": round(meter.latency_s, 3),
        "reader_latency": round(reader_latency, 3),
        "support_detail": detail[:500],
        "score": score,
        "exec": {
            "returncode": initial["returncode"],
            "timeout": initial["timeout"],
            "stderr": (initial["stderr"] or "")[-500:],
            "stdout": (initial["stdout"] or "")[:1000],
        },
        "trace_path": str(trace_path.relative_to(REPO)),
        "model": MODEL,
        "base_url": BASE_URL,
    }
    return rec


def write_doc(records: list[dict[str, Any]], docs_out: Path) -> None:
    support = Counter(r.get("grounding_label") for r in records)
    subtypes = Counter(r.get("unsupported_subtype") for r in records if r.get("unsupported_subtype"))
    dev = [r for r in records if r.get("answer") is not None]
    dev_scored = [r for r in dev if r.get("score") is not None]
    dev_score = sum(r["score"] for r in dev_scored) / len(dev_scored) if dev_scored else 0.0
    level_dist = Counter(r.get("level") for r in records)
    lines = [
        "# DABStep Support-Determination Run",
        "",
        f"- Records: `{RECORDS_OUT.relative_to(REPO)}`",
        f"- N: {len(records)}; level distribution: {dict(level_dist)}",
        f"- Reader: `{MODEL}` at `{BASE_URL}`, temperature 0.",
        f"- Setting: fixed 7-file finance/payments lake, all files offered, no discovery.",
        "",
        "## Support Distribution",
        "",
        json.dumps(dict(support), ensure_ascii=False, indent=2),
        "",
        "## Unsupported Subtype Distribution",
        "",
        json.dumps(dict(subtypes), ensure_ascii=False, indent=2),
        "",
        "## Dev-10 Official Correctness",
        "",
        f"- Dev records present: {len(dev_scored)} / 10",
        f"- Exact/numeric-tolerant score: {dev_score:.3f} ({sum(r['score'] for r in dev_scored):.0f}/{len(dev_scored) if dev_scored else 0})",
        "",
        "## Scenario Contrast",
        "",
        "DABStep is a selection-free, fixed small-lake finance/payments domain: every task receives the same 7 files, so support failures isolate code reasoning/execution rather than KramaBench-style large-lake file discovery.",
        "",
        "## Dev-10 Detail",
        "",
    ]
    for r in dev_scored:
        lines.append(
            f"- `{r['taskid']}` {r['level']} score={r['score']} support={r['grounding_label']} "
            f"subtype={r.get('unsupported_subtype')} pred={r.get('pred')!r} gold={r.get('answer')!r}"
        )
    docs_out.parent.mkdir(parents=True, exist_ok=True)
    docs_out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(limit: int, seed: int, records_out: Path, docs_out: Path, smoke: bool = False) -> list[dict[str, Any]]:
    tasks = select_tasks(limit, seed=seed, smoke=smoke)
    preview = build_context_preview()
    records_out.parent.mkdir(parents=True, exist_ok=True)
    records_out.write_text("", encoding="utf-8")
    records: list[dict[str, Any]] = []
    print(f"[dabstep-support] tasks={len(tasks)} smoke={smoke} context_files={len(ALLOWED_FILES)} -> {records_out}", flush=True)
    for idx, task in enumerate(tasks, 1):
        try:
            rec = one_task(task, preview)
        except Exception as exc:
            rec = {
                "taskid": task["taskid"],
                "level": task.get("level"),
                "query": task["query"],
                "pred": None,
                "answer": task.get("answer"),
                "grounding_label": "unknown",
                "unsupported_subtype": None,
                "files_cited": [],
                "files_read": [],
                "tokens": {"in": 0, "out": 0},
                "latency": 0.0,
                "error": str(exc)[:500],
                "score": official_score(None, task.get("answer")),
            }
        records.append(rec)
        with open(records_out, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n")
        print(
            f"  {idx:03d}/{len(tasks)} {rec['taskid']} {rec.get('level')} "
            f"pred={str(rec.get('pred'))[:28]!r} support={rec.get('grounding_label')} "
            f"subtype={rec.get('unsupported_subtype')} tok={rec.get('tokens')}",
            flush=True,
        )
    write_doc(records, docs_out)
    print(f"[dabstep-support] wrote docs -> {docs_out}", flush=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--records-out", type=Path, default=RECORDS_OUT)
    parser.add_argument("--docs-out", type=Path, default=DOC_OUT)
    parser.add_argument("--smoke", action="store_true", help="Run the first --limit tasks without dev-10 inclusion.")
    args = parser.parse_args()
    run(args.limit, args.seed, args.records_out, args.docs_out, smoke=args.smoke)


if __name__ == "__main__":
    main()
