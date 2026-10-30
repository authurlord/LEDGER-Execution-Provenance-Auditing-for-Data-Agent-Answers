"""LEDGER E12 driver: re-derive the 166 provenance labels with the refined kb_trace_provenance
(boundary_contains + unsupported_subtype + files_read/files_cited), diff against the smart01
canonical (git HEAD), emit records/kb_labels_rederived.jsonl + docs/LABEL_STABILITY.md, then
restore the canonical n2 records. Pure CPU, no LLM.

NOTE: jcyj02 re-exec has ~10% env drift vs smart01 (pandas/lib versions). The diff separates
E12-logic flips (text/boundary + subtype refinement) from env-drift flips (numeric re-exec).
"""
import os, re, json, subprocess, collections
from . import kb_trace_provenance as P

REPO = P.REPO
DOMAINS = ["environment", "wildfire", "legal", "archeology"]
SETTINGS = {"oracle": "BaselineLLMSystemLocalQwenFewShot", "discovery": "BaselineLLMSystemLocalQwenDiscovery"}
RECF = lambda d, s: os.path.join(REPO, "records", f"records_ledger_n2_kb_{d}_{s}.jsonl")


def _load_git(fp):
    rel = os.path.relpath(fp, REPO)
    try:
        out = subprocess.run(["git", "-C", REPO, "show", f"HEAD:{rel}"], capture_output=True, text=True).stdout
        return {json.loads(l)["taskid"]: json.loads(l) for l in out.splitlines() if l.strip()}
    except Exception:
        return {}


def _is_num(x):
    try:
        float(str(x)); return True
    except Exception:
        return False


def _cf_status(detail):
    m = re.search(r"cf=([a-z_]+)\(([^)]*)\)", str(detail or ""))
    if not m:
        return None, None
    return m.group(1), m.group(2)


def main():
    old = {(d, s): _load_git(RECF(d, s)) for d in DOMAINS for s in SETTINGS}
    for s, sut in SETTINGS.items():
        print(f"[rederive] re-run {sut} / {s}", flush=True)
        P.run(sut, s, DOMAINS)

    new = {}; rederived = []
    for d in DOMAINS:
        for s in SETTINGS:
            for l in open(RECF(d, s)):
                r = json.loads(l); r["_domain"] = d; r["_setting"] = s
                new[(d, s, r["taskid"])] = r; rederived.append(r)
    with open(os.path.join(REPO, "records", "kb_labels_rederived.jsonl"), "w") as f:
        for r in rederived:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    vacuous_rows = []
    for r in rederived:
        status, reason = _cf_status(r.get("support_detail"))
        if status or r.get("unsupported_subtype") == "a3_vacuous":
            row = {
                "taskid": r.get("taskid"),
                "domain": r.get("_domain"),
                "setting": r.get("_setting"),
                "system": r.get("system"),
                "pred": r.get("pred"),
                "grounding_label": r.get("grounding_label"),
                "unsupported_class": r.get("unsupported_class"),
                "unsupported_subtype": r.get("unsupported_subtype"),
                "counterfactual_status": status,
                "counterfactual_reason": reason,
                "support_detail": r.get("support_detail"),
                "trace_path": r.get("trace_path"),
            }
            vacuous_rows.append(row)
    with open(os.path.join(REPO, "records", "kb_vacuous_check.jsonl"), "w") as f:
        for r in vacuous_rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")

    flips = []; cnt = collections.Counter(); subtype_ref = 0
    for d in DOMAINS:
        for s in SETTINGS:
            for tid, o in old[(d, s)].items():
                n = new.get((d, s, tid))
                if not n:
                    continue
                og, ng = o.get("grounding_label"), n.get("grounding_label")
                if og != ng:
                    kind = "env-drift(numeric)" if _is_num(n.get("pred")) else "boundary/text-logic"
                    flips.append((d, s, tid, og, ng, kind, str(n.get("pred"))[:40]))
                    cnt[kind] += 1
                st = n.get("unsupported_subtype")
                if st and st != o.get("unsupported_class"):
                    subtype_ref += 1

    flags = ["legal-hard-1", "legal-hard-14", "legal-hard-30"]
    flagrows = []
    for tid in flags:
        for s in SETTINGS:
            n = new.get(("legal", s, tid))
            if n:
                flagrows.append((tid, s, n.get("grounding_label"), n.get("unsupported_subtype"), str(n.get("support_detail"))[:60]))

    bc = getattr(P, "boundary_contains", None) or getattr(P, "_boundary_contains", None)
    reg = {}
    if bc:
        reg = {"No in North": bc("North", "No"), "No in Nothing": bc("Nothing", "No"),
               "No == No": bc("No", "No"), "'no' in 'No '": bc("No ", "no")}

    old_dist = collections.Counter(o.get("grounding_label") for m in old.values() for o in m.values())
    new_dist = collections.Counter(r.get("grounding_label") for r in rederived)
    sub_dist = collections.Counter(r.get("unsupported_subtype") for r in rederived if r.get("grounding_label") == "unsupported")

    L = []
    L.append("# LEDGER E12 — Label re-derivation & stability\n")
    L.append("Refined `kb_trace_provenance` (boundary_contains text support + unsupported_subtype "
             "a1/a2/c1/c2/b/d + files_read vs files_cited), re-run over 166 episodes (4 CSV domains × "
             "oracle/discovery). **old** = smart01 canonical (git HEAD); **new** = jcyj02 refined re-run.\n")
    L.append("> jcyj02 re-exec differs from smart01 by ~10% on numeric grounding (pandas/lib versions). "
             "Flips are split into **E12-logic** (text/boundary) and **env-drift** (numeric re-exec) so the "
             "logic change is auditable independent of the environment. smart01 canonical remains authoritative.\n")
    L.append(f"## grounding_label distribution\n- old (smart01): {dict(old_dist)}\n- new (jcyj02):  {dict(new_dist)}\n")
    L.append(f"## unsupported_subtype distribution (new)\n- {dict(sub_dist)}  (subtype refinements vs old coarse class: {subtype_ref})\n")
    L.append(f"## grounding_label flips old→new: {len(flips)}\n- by kind: {dict(cnt)}\n")
    if flips:
        L.append("| domain | setting | taskid | old | new | kind | pred |\n|---|---|---|---|---|---|---|")
        for d, s, tid, og, ng, k, pv in sorted(flips):
            L.append(f"| {d} | {s} | {tid} | {og} | {ng} | {k} | {pv} |")
        L.append("")
    L.append("## Three flags (must stay unsupported with refined subtype: a2, c2, c2)\n")
    L.append("| taskid | setting | grounding | subtype | detail |\n|---|---|---|---|---|")
    for tid, s, g, st, det in flagrows:
        L.append(f"| {tid} | {s} | {g} | {st} | {det} |")
    L.append("")
    L.append(f"## Boundary-containment regression (substring bug fixed)\n- {reg}\n")
    with open(os.path.join(REPO, "docs", "LABEL_STABILITY.md"), "w") as f:
        f.write("\n".join(L) + "\n")

    cf_dist = collections.Counter(r.get("counterfactual_status") for r in vacuous_rows)
    a3 = [r for r in vacuous_rows if r.get("unsupported_subtype") == "a3_vacuous"]
    numeric_checked = [r for r in vacuous_rows if r.get("counterfactual_status") in ("sensitive", "vacuous")]
    passed = [r for r in numeric_checked if r.get("counterfactual_status") == "sensitive"]
    not_app = [r for r in vacuous_rows if r.get("counterfactual_status") == "not_applicable"]
    V = []
    V.append("# LEDGER P0-1 - Vacuous-Read Counterfactual Check\n")
    V.append("Pure CPU, deterministic, zero-LLM re-run over the 166 live KramaBench provenance episodes "
             "(4 CSV domains x oracle/discovery). The canonical per-domain records are restored after the run; "
             "the audit output is `records/kb_vacuous_check.jsonl`.\n")
    V.append("A record is reclassified as `unsupported` / `a3_vacuous` only when the original hermetic "
             "re-execution supports the stated numeric answer and both counterfactuals leave the recomputed value "
             "unchanged within the frozen tolerance: numeric cited cells scaled by 1.37, and 10% of cited data rows deleted.\n")
    V.append("## Summary\n")
    V.append(f"- live episodes rederived: {len(rederived)}")
    V.append(f"- hermetic-supported records entering counterfactual audit: {len(vacuous_rows)}")
    V.append(f"- numeric counterfactual checks: {len(numeric_checked)}")
    V.append(f"- passed (counterfactually sensitive): {len(passed)}")
    V.append(f"- failed / reclassified `a3_vacuous`: {len(a3)}")
    V.append(f"- not applicable or insufficient counterfactuals: {len(not_app)}")
    V.append(f"- counterfactual status distribution: {dict(cf_dist)}\n")
    V.append("## Reclassified A3 Vacuous Reads\n")
    if a3:
        V.append("| domain | setting | taskid | pred | detail |")
        V.append("|---|---|---|---|---|")
        for r in sorted(a3, key=lambda x: (x.get("domain") or "", x.get("setting") or "", x.get("taskid") or "")):
            det = str(r.get("support_detail") or "").replace("|", "\\|")[:240]
            V.append(f"| {r.get('domain')} | {r.get('setting')} | {r.get('taskid')} | {r.get('pred')} | {det} |")
    else:
        V.append("None.\n")
    V.append("## Smoke Sample\n")
    smoke = next((r for r in vacuous_rows if r.get("domain") == "environment" and r.get("setting") == "oracle"), None)
    if smoke:
        V.append("```json")
        V.append(json.dumps(smoke, ensure_ascii=False, indent=2, default=str)[:1600])
        V.append("```")
    with open(os.path.join(REPO, "docs", "VACUOUS_READ_CHECK.md"), "w") as f:
        f.write("\n".join(V) + "\n")

    for d in DOMAINS:
        for s in SETTINGS:
            subprocess.run(["git", "-C", REPO, "restore", os.path.relpath(RECF(d, s), REPO)])

    print("[rederive] DONE", flush=True)
    print(json.dumps({"flips": len(flips), "flip_kinds": dict(cnt), "subtype_refinements": subtype_ref,
                      "old_dist": dict(old_dist), "new_dist": dict(new_dist),
                      "flags": flagrows, "boundary_regression": reg,
                      "vacuous_check": {"rows": len(vacuous_rows), "numeric_checked": len(numeric_checked),
                                        "passed_sensitive": len(passed), "a3_vacuous": len(a3),
                                        "not_applicable": len(not_app), "status_dist": dict(cf_dist)}},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
