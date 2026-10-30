"""LEDGER v2 STEP 3b/3c — post-hoc trace capture + provenance over DS-Guru SUT artifacts.

Consumes KramaBench/system_scratch/<SUT>/<query_id>/_intermediate/ (pipeline-N.py, pipeline-N_out.json,
errors-N.txt, token_count-N.json, ledger_cost.json) — the SUT's reasoning is UNTOUCHED; everything here
is post-hoc and deterministic (no LLM anywhere).

Per task:
  TRACE    : cited files = file-path string literals actually referenced by the FINAL executed program
             (AST string scan, filtered to existing files); columns/ops summary; exec status.
  SUPPORT  : re-execute the final program in a scrubbed env with the filesystem view RESTRICTED to the
             cited files (cited files copied into a temp mirror; the data-root prefix in the code is
             rewritten to the mirror). An uncited read then fails loudly.
  CLASSES  : (a) stated != recomputed on cited sources (numeric tol = frozen point 1e-4)
             (b) exec failed but the system still stated an answer
             (c) uncited-source dependency (mirror re-exec hits FileNotFoundError)
             (d) required operands absent from cited sources (KeyError / missing column)
             else supported; 'unknown' for non-recomputable cases (counted separately).
Emits records/records_ledger_n2_kb_<domain>_<setting>.jsonl via cost_accounting.build_record
(LLM cost from ledger_cost.json + token_count files; checker steps tokens=0).

Run: conda run -n kramabench python -m kramabench_integration.kb_trace_provenance \
       --sut BaselineLLMSystemLocalQwenFewShot --setting oracle --domains environment,wildfire,legal,archeology
"""
from __future__ import annotations
import os, re, io, json, glob, shutil, argparse, subprocess, tempfile, contextlib, sys, unicodedata, csv
from . import cost_accounting as CA

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB = os.path.join(REPO, "KramaBench")
SCRATCH = os.path.join(KB, "system_scratch")
DATA = os.path.join(KB, "data")
# hermetic re-exec interpreter: env override, else the running interpreter
# (jcyj02 base has pandas/numpy/duckdb; smart01 used miniforge kramabench).
KENV_PY = os.environ.get("KENV_PY", sys.executable)
TOL_REL = 1e-4          # frozen operating point (LEDGER_G1_VERDICT.md)
CF_DELTA = 0.37         # deterministic counterfactual numeric perturbation
_NUM = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_READ_FUNCS = {
    "open", "read_csv", "read_excel", "read_json", "read_table", "read_html", "read_parquet",
    "read_feather", "read_sas", "read_stata", "read_pickle", "read_fwf", "read_xml",
    "read_orc", "read_hdf", "read_csv_auto",
}


# ---------------- trace capture (3b) ----------------
def _last_try(idir):
    tries = sorted(int(m.group(1)) for f in os.listdir(idir)
                   for m in [re.match(r"pipeline-(\d+)\.py$", f)] if m)
    return tries[-1] if tries else None


def _string_literals(code):
    import ast
    lits = []
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                lits.append(node.value)
    except SyntaxError:
        lits = re.findall(r"['\"]([^'\"]{4,300})['\"]", code)
    return lits


_BASENAME_IDX = {}


def _domain_index(domain):
    """basename -> relpath index for the domain lake (built once; light domains have unique basenames)."""
    if domain not in _BASENAME_IDX:
        droot = os.path.join(DATA, domain)
        idx = {}
        for dirpath, _, files in os.walk(droot):
            for f in files:
                idx.setdefault(f, os.path.relpath(os.path.join(dirpath, f), droot))
        _BASENAME_IDX[domain] = idx
    return _BASENAME_IDX[domain]


def _resolve_data_path(s, domain):
    """Resolve a path-like string against the domain lake. -> (relpath, unresolved)."""
    if not re.search(r"\.(csv|xlsx|txt|json|tsv|html)$", str(s), re.I):
        return None, None
    droot = os.path.join(DATA, domain)
    idx = _domain_index(domain)
    cands = ([s] if os.path.isabs(s) else [os.path.join(droot, s)])
    base = os.path.basename(s)
    if base in idx:
        cands.append(os.path.join(droot, idx[base]))
    hit = next((c for c in cands if os.path.exists(c)), None)
    if hit:
        real = os.path.realpath(hit)
        if real.startswith(os.path.realpath(droot)):
            return os.path.relpath(real, droot), None
    return None, s


def _literal_key(node):
    if hasattr(node, "value") and isinstance(node.value, str):
        return node.value
    return None


def _static_string_expr(node, env=None):
    """Best-effort AST string expression extraction for common open/read_csv path expressions."""
    import ast
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
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
            else:
                return None
        return "".join(parts)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        a, b = _static_string_expr(node.left, env), _static_string_expr(node.right, env)
        return a + b if a is not None and b is not None else None
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "join":
        parts = [_static_string_expr(a, env) for a in node.args]
        parts = [p for p in parts if p]
        if parts:
            return os.path.join(*parts)
    return None


def _call_name(func):
    if hasattr(func, "id"):
        return func.id
    if hasattr(func, "attr"):
        return func.attr
    return ""


def _files_read_from_ast(code, domain):
    """AST-level actual file reads via open/read_csv/read_* calls."""
    import ast
    read, unresolved = [], []
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return [], []
    env = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        val = _static_string_expr(node.value, env)
        if val:
            env[name] = val
        elif isinstance(node.value, ast.Dict):
            d = {}
            for k, v in zip(node.value.keys, node.value.values):
                kk, vv = _literal_key(k), _static_string_expr(v, env)
                if kk and vv:
                    d[kk] = vv
            if d:
                env[name] = d
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not node.args:
            continue
        if _call_name(node.func) not in _READ_FUNCS:
            continue
        s = _static_string_expr(node.args[0], env)
        if not s:
            continue
        rel, bad = _resolve_data_path(s, domain)
        if rel:
            read.append(rel)
        elif bad:
            unresolved.append(bad)
    return sorted(set(read)), sorted(set(unresolved))


def capture_trace(qdir, domain):
    """-> dict(trace) or None if no executed pipeline."""
    idir = os.path.join(qdir, "_intermediate")
    if not os.path.isdir(idir):
        return None
    n = _last_try(idir)
    if n is None:
        return None
    code = open(os.path.join(idir, f"pipeline-{n}.py"), errors="ignore").read()
    err_fp = os.path.join(idir, f"errors-{n}.txt")
    out_fp = os.path.join(idir, f"pipeline-{n}_out.json")
    stderr = open(err_fp, errors="ignore").read() if os.path.exists(err_fp) else ""
    stdout = open(out_fp, errors="ignore").read() if os.path.exists(out_fp) else ""
    cited, uncited_paths = [], []
    for s in _string_literals(code):
        rel, bad = _resolve_data_path(s, domain)
        if rel:
            cited.append(rel)
        elif bad:
            uncited_paths.append(bad)             # referenced but not found/resolvable in the domain lake
    files_read, read_unresolved = _files_read_from_ast(code, domain)
    uncited_paths.extend(read_unresolved)
    cols = sorted(set(re.findall(r"""\[['"]([A-Za-z_][A-Za-z0-9 _%/().-]{1,40})['"]\]""", code)))[:30]
    ops = sorted(set(re.findall(r"\.(groupby|merge|sum|mean|max|min|count|sort_values|query|pivot|value_counts|nunique|corr)\(", code)))
    # per-try token totals
    tin = tout = 0
    for tf in glob.glob(os.path.join(idir, "token_count-*.json")):
        try:
            t = json.load(open(tf)); tin += t.get("token_usage_input", 0); tout += t.get("token_usage_output", 0)
        except Exception:
            pass
    cost = {}
    cfp = os.path.join(idir, "ledger_cost.json")
    if os.path.exists(cfp):
        try: cost = json.load(open(cfp))
        except Exception: pass
    return {"try": n, "code": code, "stderr": stderr, "stdout": stdout,
            "cited": sorted(set(cited)), "uncited_refs": sorted(set(uncited_paths)),
            "files_read": files_read, "files_cited": sorted(set(cited)),
            "columns": cols, "ops": ops, "turns": n + 1,
            "tokens_in": cost.get("tokens_in", tin), "tokens_out": cost.get("tokens_out", tout),
            "latency_s": cost.get("latency_s", 0.0),
            "offered": cost.get("subset_files_offered", []),
            "code_path": os.path.join(idir, f"pipeline-{n}.py")}


def _last_json_obj(text):
    """Parse the LAST JSON object in mixed stdout (debug prints + pretty-printed json)."""
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    dec = json.JSONDecoder()
    last = None
    i = text.find("{")
    while i != -1:
        try:
            obj, _ = dec.raw_decode(text[i:])
            if isinstance(obj, dict):
                last = obj
        except Exception:
            pass
        i = text.find("{", i + 1)
    return last


def stated_answer(trace):
    """The system's stated main-task answer from the executed stdout json (None if absent/unparseable)."""
    d = _last_json_obj(trace["stdout"])
    if isinstance(d, dict):
        return d.get("main-task", d.get("main_task"))
    return None


# ---------------- restricted re-execution (3c) ----------------
def _scrubbed_env():
    safe = {}
    for k, v in os.environ.items():
        if any(s in k.upper() for s in ("KEY", "TOKEN", "SECRET", "PASSWORD", "OPENAI", "DEEPSEEK", "ANTHROPIC")):
            continue
        safe[k] = v
    safe["HOME"] = "/tmp"
    return safe


def _rebase_code_to_mirror(trace, domain, mirror):
    droot = os.path.realpath(os.path.join(DATA, domain))
    # trace was generated on smart01 with absolute paths under /mnt/data/yanmy/LakeOrch;
    # rebase onto this machine's REPO first, then restrict the data root to the cited mirror.
    code = trace["code"].replace(os.environ.get("SMART01_REPO", "/mnt/data/yanmy/LakeOrch"), REPO)
    return code.replace(droot, mirror).replace(os.path.join(DATA, domain), mirror)


def _copy_cited_mirror(trace, domain, mirror):
    droot = os.path.realpath(os.path.join(DATA, domain))
    copied = []
    for rel in trace["cited"]:
        src = os.path.join(droot, rel); dst = os.path.join(mirror, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        try:
            shutil.copy(src, dst)
            copied.append(rel)
        except Exception:
            pass
    return copied


def _run_code_in_mirror(trace, domain, mirror, tmp, timeout):
    code = _rebase_code_to_mirror(trace, domain, mirror)
    cf = os.path.join(tmp, "prog.py")
    open(cf, "w").write(code)
    r = subprocess.run([KENV_PY, cf], capture_output=True, text=True, timeout=timeout,
                       cwd=mirror, env=_scrubbed_env())
    if r.returncode != 0 or r.stderr.strip():
        s = r.stderr
        if "FileNotFoundError" in s or "No such file" in s:
            return None, "uncited_read", s[-400:]
        if "KeyError" in s or "columns" in s.lower() and "not" in s.lower():
            return None, "missing_operand", s[-400:]
        return None, "exec_error", s[-400:]
    if not r.stdout.strip():
        return None, "no_output", ""
    d = _last_json_obj(r.stdout)
    if isinstance(d, dict):
        return d.get("main-task", d.get("main_task")), None, ""
    return r.stdout.strip().splitlines()[-1], None, "non-json stdout tail"


def reexecute_restricted(trace, domain, timeout=180):
    """Re-run the final program with the FS view restricted to cited files. -> (value, err_kind, detail)
    err_kind in {None, 'uncited_read', 'missing_operand', 'exec_error', 'timeout', 'no_output'}."""
    tmp = tempfile.mkdtemp(prefix="ledger_prov_")
    try:
        mirror = os.path.join(tmp, "data")
        _copy_cited_mirror(trace, domain, mirror)
        return _run_code_in_mirror(trace, domain, mirror, tmp, timeout)
    except subprocess.TimeoutExpired:
        return None, "timeout", ""
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _norm_num(x):
    try:
        s = str(x).replace(",", "").replace("%", "").strip()
        m = _NUM.search(s)
        return float(m.group()) if m else None
    except Exception:
        return None


def _parse_cell_num(x):
    s = str(x).strip()
    if not s:
        return None
    s = s.replace(",", "")
    if s.startswith("$"):
        s = s[1:]
    if s.endswith("%"):
        s = s[:-1]
    if re.fullmatch(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", s):
        try:
            return float(s)
        except Exception:
            return None
    return None


def _format_cell_num(x):
    return f"{x:.12g}"


def _csv_delimiter(path):
    return "\t" if path.lower().endswith(".tsv") else ","


def _perturb_numeric_file(path, delta=CF_DELTA):
    if not re.search(r"\.(csv|tsv)$", path, re.I):
        return {"applicable": False, "cells": 0, "rows": 0, "reason": "not_csv_tsv"}
    delim = _csv_delimiter(path)
    try:
        with open(path, newline="", errors="ignore") as f:
            rows = list(csv.reader(f, delimiter=delim))
    except Exception as e:
        return {"applicable": False, "cells": 0, "rows": 0, "reason": type(e).__name__}
    if len(rows) <= 1:
        return {"applicable": False, "cells": 0, "rows": max(len(rows) - 1, 0), "reason": "no_data_rows"}
    cells = 0
    out = [rows[0]]
    for row in rows[1:]:
        nr = []
        for cell in row:
            val = _parse_cell_num(cell)
            if val is None:
                nr.append(cell)
            else:
                nr.append(_format_cell_num(val * (1.0 + delta)))
                cells += 1
        out.append(nr)
    if not cells:
        return {"applicable": False, "cells": 0, "rows": len(rows) - 1, "reason": "no_numeric_cells"}
    with open(path, "w", newline="") as f:
        csv.writer(f, delimiter=delim).writerows(out)
    return {"applicable": True, "cells": cells, "rows": len(rows) - 1, "reason": ""}


def _drop_rows_file(path):
    if not re.search(r"\.(csv|tsv)$", path, re.I):
        return {"applicable": False, "dropped": 0, "rows": 0, "reason": "not_csv_tsv"}
    delim = _csv_delimiter(path)
    try:
        with open(path, newline="", errors="ignore") as f:
            rows = list(csv.reader(f, delimiter=delim))
    except Exception as e:
        return {"applicable": False, "dropped": 0, "rows": 0, "reason": type(e).__name__}
    if len(rows) <= 1:
        return {"applicable": False, "dropped": 0, "rows": max(len(rows) - 1, 0), "reason": "no_data_rows"}
    data = rows[1:]
    # Deterministic spread: delete roughly 10% of data rows, preserving the header and columns.
    drop_n = max(1, len(data) // 10)
    drop_idx = {min(len(data) - 1, ((i + 1) * len(data) // drop_n) - 1) for i in range(drop_n)}
    kept = [row for i, row in enumerate(data) if i not in drop_idx]
    with open(path, "w", newline="") as f:
        csv.writer(f, delimiter=delim).writerows([rows[0]] + kept)
    return {"applicable": True, "dropped": len(data) - len(kept), "rows": len(data), "reason": ""}


def _same_num(a, b):
    an, bn = _norm_num(a), _norm_num(b)
    if an is None or bn is None:
        return False
    return abs(an - bn) <= max(TOL_REL * abs(an), 1e-9)


def _counterfactual_run(trace, domain, mode, timeout=180):
    tmp = tempfile.mkdtemp(prefix=f"ledger_cf_{mode}_")
    try:
        mirror = os.path.join(tmp, "data")
        copied = _copy_cited_mirror(trace, domain, mirror)
        effects = []
        for rel in copied:
            path = os.path.join(mirror, rel)
            if mode == "numeric_scale":
                eff = _perturb_numeric_file(path)
            elif mode == "row_drop":
                eff = _drop_rows_file(path)
            else:
                eff = {"applicable": False, "reason": "unknown_mode"}
            eff["file"] = rel
            effects.append(eff)
        applicable = any(e.get("applicable") for e in effects)
        if not applicable:
            return {"mode": mode, "applicable": False, "value": None, "err": None,
                    "unchanged": None, "effects": effects}
        try:
            value, err, detail = _run_code_in_mirror(trace, domain, mirror, tmp, timeout)
        except subprocess.TimeoutExpired:
            value, err, detail = None, "timeout", ""
        return {"mode": mode, "applicable": True, "value": value, "err": err,
                "detail": detail[:160], "unchanged": None, "effects": effects}
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def counterfactual_check(trace, domain, recomputed):
    """Detect vacuous reads after hermetic support: perturb cited operands and re-run deterministically.

    Only numeric recomputed answers are checked. A trace is flagged vacuous only when both a numeric-cell
    perturbation and a structural 10% row deletion leave the recomputed value unchanged within tolerance.
    """
    if _norm_num(recomputed) is None:
        return {"status": "not_applicable", "reason": "non_numeric_recomputed", "steps": []}
    steps = []
    for mode in ("numeric_scale", "row_drop"):
        step = _counterfactual_run(trace, domain, mode)
        if step.get("applicable") and step.get("err") is None:
            step["unchanged"] = _same_num(recomputed, step.get("value"))
        elif step.get("applicable"):
            step["unchanged"] = False
        steps.append(step)
    applicable = [s for s in steps if s.get("applicable")]
    if len(applicable) < 2:
        return {"status": "not_applicable", "reason": "insufficient_counterfactuals", "steps": steps}
    if all(s.get("unchanged") is True for s in applicable):
        return {"status": "vacuous", "reason": "counterfactual_insensitive", "steps": steps}
    return {"status": "sensitive", "reason": "counterfactual_changed_or_failed", "steps": steps}


def _cf_effect_summary(step):
    if not step.get("applicable"):
        reasons = sorted(set(e.get("reason") for e in step.get("effects", []) if e.get("reason")))
        return f"{step['mode']}:na({','.join(reasons[:3])})"
    if step["mode"] == "numeric_scale":
        cells = sum(int(e.get("cells") or 0) for e in step.get("effects", []))
        eff = f"cells={cells}"
    else:
        dropped = sum(int(e.get("dropped") or 0) for e in step.get("effects", []))
        eff = f"dropped={dropped}"
    if step.get("err"):
        return f"{step['mode']}:{eff},err={step.get('err')}"
    return f"{step['mode']}:{eff},value={_norm_num(step.get('value'))},unchanged={step.get('unchanged')}"


def _cf_detail(cf):
    if not cf:
        return "cf=not_run"
    steps = "; ".join(_cf_effect_summary(s) for s in cf.get("steps", []))
    return f"cf={cf.get('status')}({cf.get('reason')}); {steps}"


def _norm_text(x):
    s = unicodedata.normalize("NFKC", str(x)).casefold().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def boundary_contains(container, needle):
    """Token-boundary contains over normalized text; prevents e.g. 'no' matching 'north'."""
    c, n = _norm_text(container), _norm_text(needle)
    if not c or not n:
        return False
    if c == n:
        return True
    return re.search(r"(?<![0-9a-z_])" + re.escape(n) + r"(?![0-9a-z_])", c) is not None


def support_label(stated, trace, domain):
    """-> (label, cls, subtype, detail).
    label in {supported, unsupported, unknown}; cls in {a,b,c,d,None}.
    subtype refines unsupported only while preserving the coarse class for compatibility."""
    exec_failed = bool(trace["stderr"].strip())
    if stated in (None, "") or (isinstance(stated, str) and stated.startswith("Pipeline not successful")):
        return "unknown", None, None, "no stated answer (abstained/failed)"
    if exec_failed:
        return "unsupported", "b", "b", "final-try exec had errors but an answer was stated"
    if not trace["cited"]:
        return "unsupported", "c", "c2", "no resolvable cited file reference in the executed program"
    read_not_cited = sorted(set(trace.get("files_read") or []) - set(trace.get("files_cited") or trace["cited"]))
    if read_not_cited:
        return "unsupported", "c", "c1", "program read files outside the cited set: " + ", ".join(read_not_cited[:5])
    recomputed, err, detail = reexecute_restricted(trace, domain)
    if err == "uncited_read":
        return "unsupported", "c", "c1", detail
    if err == "missing_operand":
        return "unsupported", "d", "d", detail
    if err in ("exec_error", "timeout", "no_output"):
        return "unknown", None, None, f"restricted re-exec {err}: {detail[:200]}"
    if recomputed is None:
        return "unsupported", "a", "a2", "restricted re-exec produced no comparable main-task value"
    sn, rn = _norm_num(stated), _norm_num(recomputed)
    if sn is not None and rn is not None:
        ok = abs(rn - sn) <= max(TOL_REL * abs(sn), 1e-9)
        detail = f"stated={sn} recomputed={rn}"
        if not ok:
            return "unsupported", "a", "a1", detail
        cf = counterfactual_check(trace, domain, recomputed)
        detail = f"{detail}; {_cf_detail(cf)}"
        if cf.get("status") == "vacuous":
            return "unsupported", "a", "a3_vacuous", detail
        return "supported", None, None, detail
    ss, rs = str(stated).strip().lower(), str(recomputed).strip().lower()
    if ss and rs:
        ok = boundary_contains(rs, ss) or boundary_contains(ss, rs)
        detail = f"stated={ss[:60]!r} recomputed={rs[:60]!r}; cf=not_applicable(non_numeric_answer)"
        return ("supported" if ok else "unsupported"), (None if ok else "a"), (None if ok else "a1"), detail
    return "unsupported", "a", "a2", "non-comparable stated/recomputed values"


# ---------------- driver ----------------
def gold_map():
    rows = {}
    for l in open(os.path.join(REPO, "datasets", "normalized", "kramabench_tasks.jsonl")):
        t = json.loads(l); rows[t["taskid"]] = t
    return rows


def run(sut, setting, domains):
    gm = gold_map()
    outdir = os.path.join(REPO, "records"); os.makedirs(outdir, exist_ok=True)
    for domain in domains:
        out = os.path.join(outdir, f"records_ledger_n2_kb_{domain}_{setting}.jsonl")
        W = CA.RecordWriter(out)
        stats = {"n": 0, "supported": 0, "unsupported": 0, "unknown": 0,
                 "a": 0, "b": 0, "c": 0, "d": 0,
                 "subtypes": {"a1": 0, "a2": 0, "a3_vacuous": 0, "b": 0, "c1": 0, "c2": 0, "d": 0}}
        for qdir in sorted(glob.glob(os.path.join(SCRATCH, sut, f"{domain}-*"))):
            qid = os.path.basename(qdir)
            trace = capture_trace(qdir, domain)
            if trace is None:
                continue
            stated = stated_answer(trace)
            label, cls, subtype, detail = support_label(stated, trace, domain)
            gold = gm.get(qid, {})
            m = CA.CostMeter()
            m.add_llm((trace["tokens_in"], trace["tokens_out"])); m.turn(trace["turns"]); m.tool(1)
            rec = CA.build_record(
                taskid=qid, domain=domain, system=f"{sut}|{setting}",
                pred=stated, gold=gold.get("answer"), score=-1,   # official score joined from measures CSV
                sources_selected=trace["cited"], gold_sources=gold.get("gold_sources", []),
                meter=m, trace_path=trace["code_path"],
                provenance=[{"source_file": c, "columns": trace["columns"], "ops": trace["ops"]}
                            for c in trace["cited"]],
                grounding_label=label, unsupported_class=cls, unsupported_subtype=subtype,
                support_detail=detail[:500],
                offered_files=trace["offered"][:40], uncited_refs=trace["uncited_refs"][:10],
                files_read=trace.get("files_read", []), files_cited=trace.get("files_cited", trace["cited"]),
                exec_failed=bool(trace["stderr"].strip()))
            rec["latency_s"] = trace["latency_s"]
            W.write(rec)
            stats["n"] += 1; stats[label] += 1
            if cls: stats[cls] += 1
            if subtype: stats["subtypes"][subtype] += 1
        print(f"[{domain}/{setting}] {json.dumps(stats)} -> {out}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sut", required=True)
    ap.add_argument("--setting", required=True, choices=["oracle", "discovery"])
    ap.add_argument("--domains", default="environment,wildfire,legal,archeology")
    a = ap.parse_args()
    run(a.sut, a.setting, [d for d in a.domains.split(",") if d])
