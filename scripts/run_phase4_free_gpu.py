"""Run a bounded number of resumable Phase 4 experiments on a free CUDA session."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.modern_benchmark import execute_final_run  # noqa: E402
from ml.modern_benchmark_queue import (  # noqa: E402
    MODERN_RESULTS_ROOT,
    RunStatus,
    initialize_queue,
    matrix_audit,
    refresh_queue,
    unblock_gpu_runs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-runs", type=int, default=1, help="Maximum experiments for this free-GPU session.")
    parser.add_argument("--results-root", type=Path, default=MODERN_RESULTS_ROOT)
    parser.add_argument("--status-only", action="store_true")
    args = parser.parse_args()
    if args.max_runs < 1:
        parser.error("--max-runs must be positive")

    initialize_queue(args.results_root)
    if args.status_only:
        print(json.dumps(matrix_audit(args.results_root), indent=2, ensure_ascii=False))
        return 0

    import torch

    if not torch.cuda.is_available():
        print("BLOCKED — GPU UNAVAILABLE. Queue was not falsely executed on CPU.")
        return 2
    unblock_gpu_runs(args.results_root)
    completed_this_session = 0
    while completed_this_session < args.max_runs:
        queue = refresh_queue(args.results_root)
        resumable = [run for run in queue["runs"] if run["status"] == RunStatus.RUNNING.value]
        pending = [run for run in queue["runs"] if run["status"] == RunStatus.PENDING.value]
        candidates = resumable + pending
        if not candidates:
            break
        selected = candidates[0]
        print(f"STARTING {selected['run_id']}", flush=True)
        execute_final_run(selected["dataset"], selected["model"], results_root=args.results_root)
        completed_this_session += 1
        print(f"COMPLETED {selected['run_id']}", flush=True)
    print(json.dumps(matrix_audit(args.results_root), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
