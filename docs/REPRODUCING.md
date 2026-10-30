# Reproducing LEDGER

Run commands from the repository root after installing `requirements.txt` in a
Python 3.11 or 3.12 virtual environment.

## Run without downloading a benchmark

```bash
python scripts/verify_release.py
python examples/audit_demo.py
python running_examples/identity_theft/reproduce.py
```

The first command runs the component self-tests and verifies the frozen result
counts and annotation consistency. The SQLite demo uses synthetic sales data.
The identity-theft example uses the data already packaged beside its script.

## Recompute results from saved records

```bash
export PYTHONPATH=.:code
python code/run_numeric_support_baselines.py
python code/eval_delivery_and_escalation_policies.py
python code/analyze_positive_claims.py
python code/eval_evidence_aware_judge.py
```

These commands read the saved per-example records and regenerate summary files
under `records/` and tables under `artifacts/ipmc_rev2/`. The numeric script runs
the recovered arithmetic checker again on the recorded operands. Judge evaluation
uses saved responses and makes no new model calls.

The human annotations are in `docs/annotation/KB_SUPPORT_ANNOTATION.csv`.
The main result sources are:

| Result | Input records |
| --- | --- |
| ConvFinQA | `records/records_ledger_checker_convfinqa_full.jsonl` |
| MultiHiertt | `records/records_ledger_checker_multihiertt.jsonl` |
| Live KramaBench | `records/records_ledger_n2_kb_*.jsonl` |
| DABstep | `records/records_ledger_dabstep_support.jsonl` |
| WikiSQL semantics | `records/wikisql_semantic_perturbations.jsonl`, `records/wikisql_semantic_validation.jsonl` |
| Model judge | `records/evidence_aware_judge_records.jsonl` |

## Regenerate benchmark inputs and executions

Place the original benchmark files at the paths expected by the corresponding
loader. Large benchmark databases and model weights are downloaded separately.

| Benchmark | Expected path |
| --- | --- |
| ConvFinQA original release | `datasets/convfinqa_original/data/dev.json` |
| MultiHiertt | `datasets/multihiertt/multihiertt_data/train.json` |
| WikiSQL | `external/raw/wikisql/data/dev.jsonl`, `dev.tables.jsonl`, `dev.db` |
| KramaBench | `KramaBench/data/` and the preserved episodes in `KramaBench/system_scratch/` |
| DABstep | `datasets/dabstep/data/context/` |
| Spider SQL experiments | `external/raw/spider/spider/dev.json` and `database/<db_id>/<db_id>.sqlite` |
| BIRD SQL experiments | `external/raw/bird/minidev/MINIDEV/mini_dev_sqlite.json` and `dev_databases/` |

Dataset loader docstrings retain the original dataset identifiers and formats.
`datasets/pull_ledger_datasets.py` downloads the DABstep context and the FLARE
ConvFinQA export using `huggingface_hub`; the original ConvFinQA gold-program
release is used for the arithmetic experiments. Install optional download support
with `python -m pip install huggingface_hub pyarrow`.

With the corresponding raw files available:

```bash
export PYTHONPATH=.:code
python -m kramabench_integration.convfinqa_checker_fulln
python -m kramabench_integration.multihiertt_checker
python code/build_wikisql_semantic_perturbations.py
python code/eval_wikisql_semantic_validators.py
python code/run_closure_ablation_grid_rev2.py
```

The SQLLedger command-line entry point is:

```bash
python -m experiments.sqlledger.cli --help
```

## Optional model baselines

For a new judge run, start an OpenAI-compatible model service and choose a new
output path so the saved responses remain available:

```bash
python code/run_evidence_aware_judge.py \
  --base-url http://127.0.0.1:8000/v1 \
  --model qwen3.6-35b \
  --output output/judge-rerun.jsonl
```

For the DABstep host-generation scripts, configure `LEDGER_MODEL_BASE_URL`,
`LEDGER_MODEL_API_KEY`, and `LEDGER_MODEL_NAME` in the environment. The local-service
defaults are `http://127.0.0.1:8000/v1`, `local-placeholder`, and `qwen3.6-35b`.
Model calls occur in the host/judge scripts; the deterministic provenance checks
do not call a language model.

## Build the complete paper

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The entry point includes the method, evaluation, full appendix, and bibliography.
The generated file is `paper/main.pdf`.
