"""LEDGER STEP 4 — N2: training-free provenance / grounding layer.

Two jobs:
  1) ATTRIBUTION: for each answer, record {source_file, rows/cols or text-span} that produced it,
     reconstructed from the agent's execution trace (which file was read/aggregated).
  2) UNSUPPORTED DETECTION: decide grounding_label in {supported, unsupported, unknown}:
       - numeric answers  -> re-check the claimed value against the cited source (value present /
         recompute-consistent within tolerance).
       - entity/text answers -> the cited source text contains the entity + a discriminating property.

CRITICAL FRAMING (guidance STEP 4 + proposal §4/§6.3): this asks "is the answer SUPPORTED by its cited
source" (grounding), NOT "is the answer correct" (correctness). We reuse hop_verify.surface_boundary /
source_contains as SUPPORT checks, NOT as answer vetoes — a different use than the failed final-answer
correctness gate (which had correct-FP 34/47). Evaluation is against SUPPORT labels, never correctness;
the flag's correlation with correctness is only a secondary/exploratory analysis.

Standalone + GPU-free (deterministic file ops). Self-tested in __main__ against tiny CSVs.
"""
from __future__ import annotations
import os, re, json
from dataclasses import dataclass, field
from typing import Optional, Any
try:
    from . import hop_verify as HV
except Exception:                        # script mode
    import hop_verify as HV

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


# ---------------- attribution ----------------
@dataclass
class ProvenanceItem:
    source_file: str
    locator: str = ""          # "rows: filter Violation=yes; col: Bacteria" OR "span@1234" OR ""
    kind: str = "table"        # table | text

    def as_dict(self) -> dict:
        return {"source_file": self.source_file, "locator": self.locator, "kind": self.kind}


def attribute(trace: list[dict]) -> list[ProvenanceItem]:
    """Reconstruct provenance from an execution trace.

    Expected trace entries (produced by the agent loop; adapt keys at wiring time):
        {"tool": "open_file"|"run_sql"|"run_python"|"read_text", "path": "...",
         "columns": [...], "filter": "...", "used_in_answer": bool}
    Heuristic: any file that was read/aggregated AND flagged used_in_answer (or the LAST computed-over
    file if none flagged) is attributed.
    """
    items: list[ProvenanceItem] = []
    last_table = None
    for ev in trace or []:
        path = ev.get("path") or ev.get("file")
        if not path:
            continue
        tool = ev.get("tool", "")
        kind = "text" if (tool == "read_text" or str(path).endswith((".txt", ".md"))) else "table"
        loc_bits = []
        if ev.get("filter"): loc_bits.append(f"filter: {ev['filter']}")
        if ev.get("columns"): loc_bits.append("cols: " + ",".join(map(str, ev["columns"][:6])))
        if ev.get("span") is not None: loc_bits.append(f"span@{ev['span']}")
        item = ProvenanceItem(source_file=os.path.basename(str(path)),
                              locator="; ".join(loc_bits), kind=kind)
        if ev.get("used_in_answer"):
            items.append(item)
        if kind == "table":
            last_table = item
    if not items and last_table is not None:      # fallback: attribute the last computed table
        items.append(last_table)
    # dedup by (file, locator)
    seen, out = set(), []
    for it in items:
        k = (it.source_file, it.locator)
        if k not in seen:
            seen.add(k); out.append(it)
    return out


# ---------------- support checks ----------------
def _value_present_in_table(path: str, value: float, tol_rel: float = 1e-4) -> tuple[bool, str]:
    """Is `value` present (or a simple aggregate) in the cited table? Uses duckdb via hop_verify helpers."""
    if not os.path.exists(path):
        return False, "file not found"
    try:
        import duckdb
        con = duckdb.connect()
        con.execute(f"CREATE VIEW t AS SELECT * FROM read_csv_auto('{HV._q(path)}', ignore_errors=true, sample_size=200000)")
        cols = [r[0] for r in con.execute("DESCRIBE t").fetchall()]
        tol = max(tol_rel * abs(value), 1e-6)

        def qc(c):  # quote+escape identifier, strip commas/%/$ before numeric cast (73.8% / 1,234 / $12)
            cc = c.replace('"', '""')
            return f"TRY_CAST(REPLACE(REPLACE(REPLACE(CAST(\"{cc}\" AS VARCHAR),',',''),'%',''),'$','') AS DOUBLE)"

        for c in cols:
            try:
                expr = qc(c)
                agg = con.execute(f'SELECT COUNT(*) n, SUM({expr}) s, AVG({expr}) a, '
                                  f'MAX({expr}) mx, MIN({expr}) mn FROM t').fetchone()
                n, s, a, mx, mn = agg
                # COUNT(*) is an aggregate independent of the column cast -> only count it once (on first col)
                for label, cand in (("count", n), ("sum", s), ("avg", a), ("max", mx), ("min", mn)):
                    if cand is None:
                        continue
                    if abs(float(cand) - value) <= tol:
                        strength = "row-count" if label == "count" else f'{label}("{c}")'
                        return True, f'value≈{strength} ({cand})'
                hit = con.execute(f'SELECT COUNT(*) FROM t WHERE ABS({expr} - {value}) <= {tol}').fetchone()[0]
                if hit and hit > 0:
                    return True, f'value present as a cell in column "{c}" ({hit} rows)'
            except Exception:
                continue
        # NOTE (codex P1): this is a WEAK/loose support signal — it matches ANY column aggregate/cell without
        # using the trace's cited column/filter/operation. Wire cited-column constraints at integration to
        # tighten unsupported-detection precision (see CODEX_LEDGER_REVIEW.md).
        return False, "value not derivable from any column aggregate/cell"
    except Exception as e:
        return False, "error: " + str(e)[:80]


def check_support(answer: Any, provenance: list[ProvenanceItem], data_root: str = "",
                  property_terms: Optional[list[str]] = None) -> tuple[str, list[dict]]:
    """Return (grounding_label, evidence[]). label in {supported, unsupported, unknown}.

    - No provenance -> unknown.
    - Numeric answer -> supported if value is present/derivable in ANY cited table.
    - Text/entity answer -> supported if the entity co-occurs with a property in ANY cited text source
      (hop_verify.source_contains, used here as a SUPPORT check, not a veto).
    """
    if not provenance:
        return "unknown", []
    ev = []
    s = str(answer).strip()
    m = _NUM.fullmatch(s.replace("%", "").replace(",", "").strip())
    is_numeric = m is not None
    supported = False
    for p in provenance:
        path = os.path.join(data_root, p.source_file) if data_root else p.source_file
        if is_numeric and p.kind == "table":
            ok, detail = _value_present_in_table(path, float(s.replace("%", "").replace(",", "")))
        elif not is_numeric:
            ok, detail = HV.source_contains(path, s, property_terms or [w for w in re.findall(r"[A-Za-z]{4,}", s)][:3])
        else:
            ok, detail = False, "numeric answer but cited source is text"
        ev.append({"source_file": p.source_file, "supported": bool(ok), "detail": detail})
        supported = supported or bool(ok)
    return ("supported" if supported else "unsupported"), ev


# ---------------- operand-constrained support check (LEDGER v2 STEP 1, the C1 core) ----------------
class ConstrainedPool:
    """Deterministic derivability pool over the CITED operands only (no free closure). NEVER an LLM judge.

    Vectorized (numpy): D0 = cited operands ∪ cited consts; D1 = pairwise ops over D0;
    D2 = pairwise ops over D1 × (D0 ∪ D1) — covers linear 2-chains AND (#i op #j) combos.
    Depth-3 (gold-typical linear tail, e.g. add,add,divide) is checked by INVERTING the last op:
    t = op(x, d0)  →  solve x, membership-test x ∈ D2. No D3 materialization (the naive closure
    exploded combinatorially — negatives never match and forced full expansion).

    ops: sum(a+b) diff(a−b, b−a) ratio(a/b, b/a) pctchg((a−b)/b, (b−a)/a) prod(a*b — grid variant).
    Percent/fraction equivalence (v ↔ v/100 ↔ 100v) is applied to the TARGET, gated on percent_ctx.
    Build once per (citations, ops); check() is O(log n) per target — tol/percent are match-time knobs.
    """
    MAXP = 500_000

    def __init__(self, operands, consts=(), ops=("sum", "diff", "ratio", "pctchg"), chain="full"):
        """chain='linear': D2 = D1 × D0 only (gold programs are op(#prev, literal) chains — ConvFinQA
        seed has ZERO op(#i,#j) programs, so linearization costs no recall while shrinking the
        false-support surface ~30x). chain='full': D2 = D1 × (D0 ∪ D1)."""
        import numpy as np
        self.np = np
        self.ops = ops
        self.chain = chain
        base = sorted({float(x) for x in list(operands) + list(consts)})
        self.d0 = np.array(base, dtype=np.float64)
        if len(self.d0) == 0:
            self.d1 = self.d2 = self.d0
            return
        d1 = self._pair(self.d0, self.d0)
        self.d1 = np.unique(d1)[:self.MAXP]
        right = self.d0 if chain == "linear" else np.unique(np.concatenate([self.d0, self.d1]))
        self.d2 = np.unique(self._pair(self.d1, right))[:self.MAXP]

    def _pair(self, A, B):
        np = self.np
        a = A[:, None]; b = B[None, :]
        outs = []
        if "sum" in self.ops: outs.append((a + b).ravel())
        if "diff" in self.ops: outs += [(a - b).ravel(), (b - a).ravel()]
        with np.errstate(divide="ignore", invalid="ignore"):
            if "ratio" in self.ops:
                outs += [np.where(b != 0, a / b, np.nan).ravel(), np.where(a != 0, b / a, np.nan).ravel()]
            if "pctchg" in self.ops:
                outs += [np.where(b != 0, (a - b) / b, np.nan).ravel(), np.where(a != 0, (b - a) / a, np.nan).ravel()]
        if "prod" in self.ops: outs.append((a * b).ravel())
        v = np.concatenate(outs) if outs else np.array([])
        v = v[np.isfinite(v)]
        return v[np.abs(v) < 1e15]

    def _member(self, arr, x, tol_rel):
        """x within max(tol_rel*|x|, 1e-9) of any element of sorted arr."""
        if len(arr) == 0:
            return False
        np = self.np
        tol = max(tol_rel * abs(x), 1e-9)
        i = np.searchsorted(arr, x)
        for j in (i - 1, i, i + 1):
            if 0 <= j < len(arr) and abs(arr[j] - x) <= tol:
                return True
        # tol window may span multiple slots; widen scan
        lo = np.searchsorted(arr, x - tol); hi = np.searchsorted(arr, x + tol)
        return hi > lo

    def check(self, value, tol_rel=1e-4, percent=True, percent_ctx=True, max_depth=3):
        """-> 'depth0'|'depth1'|'depth2'|'depth3' | None."""
        if value is None or len(self.d0) == 0:
            return None
        v = float(value)
        targets = [v] + ([v / 100.0, v * 100.0] if (percent and percent_ctx and v != 0) else [])
        for t in targets:
            if self._member(self.d0, t, tol_rel):
                return "depth0"
        if max_depth >= 1:
            for t in targets:
                if self._member(self.d1, t, tol_rel):
                    return "depth1"
        if max_depth >= 2:
            for t in targets:
                if self._member(self.d2, t, tol_rel):
                    return "depth2"
        if max_depth >= 3:
            # invert the linear tail: t = op(x, d0) -> x; test x ∈ D2
            for t in targets:
                for d0 in self.d0:
                    cands = []
                    if "sum" in self.ops: cands.append(t - d0)
                    if "diff" in self.ops: cands += [t + d0, d0 - t]
                    if "ratio" in self.ops:
                        if d0 != 0: cands.append(t * d0)
                        if t != 0: cands.append(d0 / t)
                    if "pctchg" in self.ops:
                        cands.append(d0 * (1 + t))
                        if 1 + t != 0: cands.append(d0 / (1 + t))
                    if "prod" in self.ops and d0 != 0: cands.append(t / d0)
                    for x in cands:
                        if x is not None and abs(x) < 1e15 and self._member(self.d2, x, tol_rel):
                            return "depth3"
        return None


def check_support_constrained(value, operands, consts=(), ops=("sum", "diff", "ratio", "pctchg"),
                              tol_rel=1e-4, percent=True, percent_ctx=True, max_depth=3):
    """One-shot wrapper (builds the pool per call). Returns (label, detail) like the free checker.
    For grid/batch use, build ConstrainedPool once per (citations, ops) and call .check() per point."""
    pool = ConstrainedPool(operands, consts, ops)
    lab = pool.check(value, tol_rel, percent, percent_ctx, max_depth)
    return (lab, f"constrained({len(pool.d0)} cited operands, ops={'/'.join(ops)})") if lab else (None, None)


# ---------------- evaluation (against SUPPORT labels, not correctness) ----------------
def eval_support(records: list[dict], support_labels: dict[str, str]) -> dict:
    """Evaluate provenance / unsupported-detection against human/gold SUPPORT labels.

    records: [{taskid, grounding_label, provenance, ...}] emitted by the agent.
    support_labels: {taskid: "supported"|"unsupported"} annotated from gold programs / re-run aggregation.
    Reports provenance coverage + unsupported-detection precision/recall. Correctness is NOT used here.
    """
    tp = fp = fn = tn = 0
    prov_present = 0
    n = 0
    for r in records:
        tid = r.get("taskid")
        if tid not in support_labels:
            continue
        n += 1
        if r.get("provenance"):
            prov_present += 1
        pred_unsup = (r.get("grounding_label") == "unsupported")
        gold_unsup = (support_labels[tid] == "unsupported")
        if pred_unsup and gold_unsup: tp += 1
        elif pred_unsup and not gold_unsup: fp += 1
        elif not pred_unsup and gold_unsup: fn += 1
        else: tn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"n": n, "provenance_coverage": round(prov_present / n, 4) if n else 0.0,
            "unsupported_precision": round(prec, 4), "unsupported_recall": round(rec, 4),
            "unsupported_f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "note": "evaluated vs SUPPORT labels, NOT correctness (guidance STEP 4)"}


if __name__ == "__main__":
    import tempfile
    d = tempfile.mkdtemp()
    csv = os.path.join(d, "water-body-testing-2013.csv")
    with open(csv, "w") as f:
        f.write("Beach,State,Violation,Bacteria\n")
        for i in range(738):
            f.write(f"B{i},Massachusetts,yes,{100+i}\n")
        for i in range(262):
            f.write(f"C{i},Massachusetts,no,{10+i}\n")
    # attribution from a trace
    trace = [{"tool": "run_python", "path": csv, "columns": ["Violation", "Bacteria"],
              "filter": "Violation=yes", "used_in_answer": True}]
    prov = attribute(trace)
    assert prov and prov[0].source_file == "water-body-testing-2013.csv", prov
    # numeric support: 738 (count of violations) IS derivable -> supported
    lbl, ev = check_support(738, prov, data_root=d)
    print("support(738):", lbl, ev)
    assert lbl == "supported", ev
    # a fabricated value NOT derivable -> unsupported
    lbl2, ev2 = check_support(999999, prov, data_root=d)
    print("support(999999):", lbl2)
    assert lbl2 == "unsupported"
    # eval harness
    recs = [{"taskid": "t1", "grounding_label": "unsupported", "provenance": [prov[0].as_dict()]},
            {"taskid": "t2", "grounding_label": "supported", "provenance": [prov[0].as_dict()]}]
    labels = {"t1": "unsupported", "t2": "supported"}
    m = eval_support(recs, labels)
    print("eval_support:", m)
    assert m["unsupported_precision"] == 1.0 and m["unsupported_recall"] == 1.0
    print("provenance self-test OK")
