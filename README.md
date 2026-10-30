# LEDGER: Execution Provenance Auditing for Data-Agent Answers

**[Full version — complete paper with appendix (PDF)](paper/main.pdf)** · **[LaTeX source](paper/main.tex)**

Research code, frozen experiment records, and the full manuscript for **LEDGER**,
presented at IP&MC2026. LEDGER audits completed data-agent episodes by connecting
declared evidence, observed reads, the final program, and the reported answer.
It checks execution provenance and keeps task semantics as a separate judgement.

The implementation was developed in
[LakeOrch's `ledger` branch](https://github.com/authurlord/LakeOrch/tree/776266a77069df9330303051e4aff22c702fca6c).
This repository brings that implementation together with the July 30 experiment
expansion and the complete paper. The original module names are retained so
experiment records and historical commands remain traceable.

## Quick start

Use Python **3.11 or 3.12**, from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Audit self-tests, frozen result checks, and annotation consistency; no model calls.
python scripts/verify_release.py

# Reproduce the paper's real running example using the packaged source files.
python running_examples/identity_theft/reproduce.py

# Small in-memory SQLite replay and arithmetic examples.
python examples/audit_demo.py
```

The running example verifies file hashes, reproduces the original reported value
using the observed-read view, reruns the same program with only declared evidence,
and independently recomputes the benchmark reference. No original server paths,
model server, or full benchmark checkout are needed for these commands.

## Code map

| Part | Entry point |
| --- | --- |
| Operand-constrained arithmetic closure | [`kramabench_integration/provenance.py`](kramabench_integration/provenance.py) |
| Python/pandas trace audit and restricted replay | [`kramabench_integration/kb_trace_provenance.py`](kramabench_integration/kb_trace_provenance.py) |
| SQLite read capture and restricted replay | [`experiments/sqlledger/runtime/`](experiments/sqlledger/runtime/) |
| ConvFinQA and MultiHiertt evaluation | [`convfinqa_checker_fulln.py`](kramabench_integration/convfinqa_checker_fulln.py), [`multihiertt_checker.py`](kramabench_integration/multihiertt_checker.py) |
| Final numeric baselines and ablations | [`code/run_numeric_support_baselines.py`](code/run_numeric_support_baselines.py), [`code/run_closure_ablation_grid_rev2.py`](code/run_closure_ablation_grid_rev2.py) |
| WikiSQL semantic perturbations and validators | [`code/build_wikisql_semantic_perturbations.py`](code/build_wikisql_semantic_perturbations.py), [`code/eval_wikisql_semantic_validators.py`](code/eval_wikisql_semantic_validators.py) |
| Selective delivery and frozen-record reanalysis | [`code/eval_delivery_and_escalation_policies.py`](code/eval_delivery_and_escalation_policies.py), [`code/analyze_positive_claims.py`](code/analyze_positive_claims.py) |
| Real running example | [`running_examples/identity_theft/`](running_examples/identity_theft/) |
| Per-example records / paper-ready tables | [`records/`](records/), [`artifacts/ipmc_rev2/`](artifacts/ipmc_rev2/) |
| Full paper, including appendix | [`paper/`](paper/) |

## Reproducing results

The repository includes the frozen per-example records and human annotations.
You can check reported counts and recompute the numeric baselines without a GPU:

```bash
python code/submission_assertions_rev2.py
python code/check_record_to_paper_consistency.py
PYTHONPATH=.:code python code/run_numeric_support_baselines.py
PYTHONPATH=.:code python code/eval_delivery_and_escalation_policies.py
PYTHONPATH=.:code python code/analyze_positive_claims.py
```

| Evaluation | Reported result |
| --- | --- |
| ConvFinQA | 360/386 reference derivations certified; 371/386 cross-example substitutions rejected |
| MultiHiertt, frozen transfer | 5,648/6,301 certified; 6,125/6,301 substitutions rejected |
| WikiSQL semantic changes | 2,715 perturbations from 500 references; output comparison misses 466; normalized SQL detects all perturbations |
| Human-labelled KramaBench | Delivered support: 11/39 → 11/21; all 11 supported episodes retained |
| Officially scored KramaBench | Mean score: 0.307 → 0.429, retaining 95/148 episodes |

These are different metrics and populations. Replay-based selection changes which
completed answers are delivered. Semantic correctness is assessed separately.
Historical record labels `supported` / `unsupported` / `unknown` refer to the
corresponding experiment's audit outcome; they are not interchangeable with human
correctness labels.

See **[Reproduction guide](docs/REPRODUCING.md)** for dataset placement, optional
model-based baselines, and commands to regenerate outputs.

## Build the full paper

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

For Overleaf, upload the contents of `paper/`, set `main.tex` as the main document,
and use pdfLaTeX. The full manuscript includes the appendix and references;
the four-page Procedia manuscript and poster are separate presentation artifacts.

## Sources

- Core implementation: LakeOrch `ledger`, commit
  [`776266a`](https://github.com/authurlord/LakeOrch/commit/776266a77069df9330303051e4aff22c702fca6c).
- Experiment expansion: `LEDGER_IPMC_POSITIVE_EVIDENCE_PACKAGE_20260730.zip`.
- Full-paper source: this repository's paper-only commit
  [`6bf1f5b`](https://github.com/authurlord/LEDGER-Execution-Provenance-Auditing-for-Data-Agent-Answers/commit/6bf1f5b60e72dafe1833d89889e822bff581e7b4).

Original source notices and benchmark attribution are retained in the code and
dataset references.
