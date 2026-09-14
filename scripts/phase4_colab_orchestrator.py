"""Run the safe, bounded, idempotent Phase 4 Colab workflow."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.phase4_orchestrator import EngineeringReviewRequired, run_orchestrator  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-runs", type=int, default=1)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results" / "modern")
    args = parser.parse_args()
    try:
        summary = run_orchestrator(
            max_runs=args.max_runs,
            verbose=args.verbose,
            project_root=PROJECT_ROOT,
            results_root=args.results_root,
        )
    except EngineeringReviewRequired as exc:
        print(str(exc), file=sys.stderr)
        return 3
    except BaseException as exc:
        print(f"ORCHESTRATION STOPPED — {exc}", file=sys.stderr)
        return 2
    if args.verbose:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
