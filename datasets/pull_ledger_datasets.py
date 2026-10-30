"""Pull the LEDGER second-benchmark + provenance datasets (raw). Proxy via clash. Idempotent."""
import os, sys
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER","0")
from huggingface_hub import snapshot_download
ROOT=os.path.dirname(os.path.abspath(__file__))
def pull(repo, patterns, sub):
    dst=os.path.join(ROOT, sub)
    print(f"[pull] {repo} -> {dst} patterns={patterns}", flush=True)
    p=snapshot_download(repo_id=repo, repo_type="dataset", local_dir=dst,
                        allow_patterns=patterns)
    print(f"[done] {repo} @ {p}", flush=True)
if __name__=="__main__":
    # DABStep: second data-agent benchmark (450+ tasks). Only tasks + the fixed context lake.
    pull("adyen/DABstep", ["data/tasks/*.jsonl","data/context/*"], "dabstep")
    # ConvFinQA: numeric multi-step QA w/ gold programs -> exact provenance support labels (STEP 4).
    pull("TheFinAI/flare-convfinqa", ["data/*.parquet"], "convfinqa")
    print("ALL PULLS COMPLETE", flush=True)
