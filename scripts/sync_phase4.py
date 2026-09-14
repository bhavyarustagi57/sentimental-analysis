"""Synchronize Phase 4 evidence without running training or final evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.modern_benchmark_queue import (  # noqa: E402
    MODERN_DATASET_KEYS,
    MODERN_MODEL_KEYS,
    MODERN_RESULTS_ROOT,
    atomic_json,
    promote_staged_result,
    read_json,
    run_paths,
)
from ml.phase4_persistence import (  # noqa: E402
    persist_completed_checkpoint,
    restore_checkpoint,
    restore_lightweight_from_github,
    sync_lightweight_to_github,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("push-lightweight", "pull-lightweight", "upload-checkpoint",
                                             "restore-checkpoint", "persist-promote"))
    parser.add_argument("--dataset", choices=MODERN_DATASET_KEYS)
    parser.add_argument("--model", choices=MODERN_MODEL_KEYS)
    parser.add_argument("--results-root", type=Path, default=MODERN_RESULTS_ROOT)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    if args.command in {"upload-checkpoint", "restore-checkpoint", "persist-promote"} and not (args.dataset and args.model):
        parser.error(f"{args.command} requires --dataset and --model")
    if args.command == "push-lightweight":
        changed = sync_lightweight_to_github(args.project_root, results_root=args.results_root)
        print(json.dumps({"pushed": changed}))
    elif args.command == "pull-lightweight":
        restore_lightweight_from_github(args.project_root)
        print(json.dumps({"restored": True}))
    elif args.command == "restore-checkpoint":
        print(json.dumps(restore_checkpoint(
            args.dataset, args.model, results_root=args.results_root,
        ), indent=2))
    else:
        persistence = persist_completed_checkpoint(
            args.dataset, args.model, results_root=args.results_root,
        )
        if args.command == "persist-promote":
            paths = run_paths(args.dataset, args.model, args.results_root)
            result = dict(read_json(paths.result))
            result["checkpoint_persistence"] = persistence
            atomic_json(paths.result, result)
            promoted = promote_staged_result(args.dataset, args.model, args.results_root)
            print(json.dumps({"persistence": persistence, "promotion": promoted}, indent=2))
        else:
            print(json.dumps(persistence, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
