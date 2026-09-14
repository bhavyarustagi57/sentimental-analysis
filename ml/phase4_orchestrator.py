"""Idempotent one-command orchestration for zero-cost Phase 4 GPU sessions."""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .modern_benchmark import execute_final_run
from .modern_benchmark_queue import (
    FailureClass,
    MODERN_RESULTS_ROOT,
    RunStatus,
    initialize_queue,
    matrix_audit,
    read_json,
    rebuild_canonical_results,
    refresh_queue,
    run_paths,
    transition_run,
    validate_completed_result,
)
from .phase4_persistence import (
    restore_checkpoint,
    sync_lightweight_to_github,
    validate_git_identity,
    validate_persistence_environment,
)


NEOBERT_XFORMERS = "0.0.35"
GEMMA_ID = "google/gemma-3-270m"
GEMMA_REVISION = "9b0cfec892e2bc2afd938c98eabe4e4a7b1e0ca1"


class EngineeringReviewRequired(RuntimeError):
    pass


@dataclass
class OrchestratorServices:
    runner: Callable[..., Any] = subprocess.run
    execute: Callable[..., Any] = execute_final_run
    restore: Callable[..., Any] = restore_checkpoint
    sync: Callable[..., Any] = sync_lightweight_to_github
    api_factory: Callable[..., Any] | None = None
    torch_module: Any | None = None
    import_module: Callable[[str], Any] = importlib.import_module
    package_version: Callable[[str], str] = importlib.metadata.version


def _run_quiet(services: OrchestratorServices, command: list[str], project_root: Path, *, check: bool = True) -> Any:
    return services.runner(command, cwd=project_root, check=check, capture_output=True, text=True)


def _validate_repository(project_root: Path) -> None:
    required = (project_root / "ml" / "modern_benchmark.py", project_root / "scripts" / "sync_phase4.py")
    if not all(path.is_file() for path in required) or not (project_root / ".git").exists():
        raise RuntimeError("Run the orchestrator from the root of the cloned Phase 4 Git repository.")


def _validate_environment(environment: Mapping[str, str]) -> None:
    validate_persistence_environment(environment)
    if not (environment.get("GH_TOKEN", "").strip() or environment.get("GITHUB_TOKEN", "").strip()):
        raise RuntimeError("GH_TOKEN or GITHUB_TOKEN is required for durable Phase 4 Git synchronization.")
    if not environment.get("PHASE4_GIT_NAME", "").strip() or not environment.get("PHASE4_GIT_EMAIL", "").strip():
        raise RuntimeError("Set PHASE4_GIT_NAME and PHASE4_GIT_EMAIL before GPU work.")


def _configure_and_verify_git(
    project_root: Path, environment: Mapping[str, str], services: OrchestratorServices,
) -> None:
    for key, env_key in (("user.name", "PHASE4_GIT_NAME"), ("user.email", "PHASE4_GIT_EMAIL")):
        _run_quiet(services, ["git", "config", key, environment[env_key]], project_root)
    validate_git_identity(project_root, runner=services.runner)
    remote = _run_quiet(services, ["git", "remote", "get-url", "origin"], project_root)
    remote_url = str(remote.stdout).strip()
    if not remote_url or ("://" in remote_url and "@" in remote_url.split("://", 1)[1].split("/", 1)[0]):
        raise RuntimeError("Git origin is missing or embeds credentials; use the credential helper instead.")
    _run_quiet(services, ["git", "ls-remote", "--exit-code", "origin", "HEAD"], project_root)
    _run_quiet(services, ["git", "pull", "--ff-only", "origin", "main"], project_root)


def _cuda_profile(services: OrchestratorServices) -> tuple[Any, str, str]:
    torch = services.torch_module or importlib.import_module("torch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Select a free Colab GPU runtime before continuing.")
    return torch, str(torch.cuda.get_device_name(0)), str(torch.__version__)


def _verify_hugging_face(environment: Mapping[str, str], services: OrchestratorServices) -> None:
    if services.api_factory is None:
        from huggingface_hub import HfApi

        services.api_factory = HfApi
    token = environment["HF_TOKEN"]
    api = services.api_factory(token=token)
    api.whoami(token=token)
    api.repo_info(environment["PHASE4_HF_REPO"], repo_type="model", token=token)
    api.model_info(GEMMA_ID, revision=GEMMA_REVISION, token=token)


def _local_completed_valid(dataset: str, model: str, results_root: Path) -> bool:
    paths = run_paths(dataset, model, results_root)
    try:
        validate_completed_result(
            read_json(paths.result), paths.predictions, expected_config=read_json(paths.config),
        )
        return RunStatus(read_json(paths.state)["status"]) is RunStatus.COMPLETED
    except (FileNotFoundError, ValueError, KeyError):
        return False


def recover_completed_runs(
    results_root: Path,
    environment: Mapping[str, str],
    services: OrchestratorServices,
) -> int:
    initialize_queue(results_root)
    if services.api_factory is None:
        from huggingface_hub import HfApi

        services.api_factory = HfApi
    token = environment["HF_TOKEN"]
    api = services.api_factory(token=token)
    remote_files = set(api.list_repo_files(
        environment["PHASE4_HF_REPO"], repo_type="model", token=token,
    ))
    restored = 0
    for state in refresh_queue(results_root)["runs"]:
        prefix = f"phase4/{state['run_id']}"
        remote_completed_evidence = all(
            f"{prefix}/{name}" in remote_files
            for name in ("persistence_manifest.json", "result.json", "state.json", "predictions.jsonl")
        )
        if state["status"] != RunStatus.COMPLETED.value and not remote_completed_evidence:
            continue
        if _local_completed_valid(state["dataset"], state["model"], results_root):
            continue
        try:
            services.restore(
                state["dataset"], state["model"], results_root=results_root,
                environment=environment,
            )
        except BaseException as exc:
            raise RuntimeError(f"Completed artifact recovery failed for {state['run_id']}: {exc}") from exc
        if not _local_completed_valid(state["dataset"], state["model"], results_root):
            raise RuntimeError(f"Completed artifact recovery remained invalid for {state['run_id']}.")
        restored += 1
    if any(state["status"] == RunStatus.COMPLETED.value for state in refresh_queue(results_root)["runs"]):
        rebuild_canonical_results(results_root)
    return restored


def _needs_neobert_dependency(states: list[dict[str, Any]]) -> bool:
    return any(
        state["model"] == "neobert" and state["status"] in {
            RunStatus.PENDING.value, RunStatus.RUNNING.value, RunStatus.FAILED.value,
        }
        for state in states
    )


def ensure_neobert_dependency(services: OrchestratorServices, project_root: Path) -> bool:
    """Install the locked wheel without allowing pip to replace torch/CUDA dependencies."""
    torch, gpu_before, torch_before = _cuda_profile(services)
    installed = False
    try:
        version = services.package_version("xformers")
        services.import_module("xformers")
    except (ImportError, importlib.metadata.PackageNotFoundError):
        _run_quiet(
            services,
            [sys.executable, "-m", "pip", "install", "--no-deps", f"xformers=={NEOBERT_XFORMERS}"],
            project_root,
        )
        installed = True
        version = services.package_version("xformers")
        services.import_module("xformers")
    if version != NEOBERT_XFORMERS:
        raise EngineeringReviewRequired(
            f"ENGINEERING REVIEW REQUIRED — expected xformers=={NEOBERT_XFORMERS}, found {version}."
        )
    torch_after, gpu_after, version_after = _cuda_profile(services)
    if torch_after is not torch or version_after != torch_before or gpu_after != gpu_before:
        raise RuntimeError("Dependency installation changed PyTorch or CUDA hardware identity.")
    return installed


def _known_xformers_failure(state: Mapping[str, Any]) -> bool:
    message = str((state.get("failure") or {}).get("message", "")).lower()
    return state.get("model") == "neobert" and "xformers" in message and any(
        phrase in message for phrase in ("no module named", "not installed", "requires", "missing")
    )


def normalize_retryable_states(results_root: Path, *, dependency_ready: bool) -> int:
    retried = 0
    for state in refresh_queue(results_root)["runs"]:
        status = RunStatus(state["status"])
        attempts = int(state.get("test_evaluation_attempts", 0))
        checkpoint = state.get("last_checkpoint")
        if status is RunStatus.RUNNING and checkpoint and not Path(checkpoint).is_dir():
            if attempts:
                raise EngineeringReviewRequired(f"ENGINEERING REVIEW REQUIRED — interrupted post-test run {state['run_id']}.")
            transition_run(
                state["dataset"], state["model"], RunStatus.FAILED,
                updates={"failure": {"classification": FailureClass.TRANSIENT_INFRASTRUCTURE.value,
                                     "message": "Ephemeral in-progress checkpoint was lost after runtime reset."}},
                results_root=results_root,
            )
            transition_run(
                state["dataset"], state["model"], RunStatus.PENDING,
                updates={"last_checkpoint": None, "best_checkpoint": None,
                         "current_epoch": 0, "current_step": 0, "failure": None},
                results_root=results_root,
            )
            retried += 1
        elif status is RunStatus.FAILED:
            failure_class = str((state.get("failure") or {}).get("classification", ""))
            safe = attempts == 0 and (
                (_known_xformers_failure(state) and dependency_ready)
                or (failure_class == FailureClass.TRANSIENT_INFRASTRUCTURE.value and int(state.get("current_step", 0)) == 0)
            )
            if not safe:
                raise EngineeringReviewRequired(f"ENGINEERING REVIEW REQUIRED — {state['run_id']} is not automatically retryable.")
            transition_run(state["dataset"], state["model"], RunStatus.PENDING,
                           updates={"failure": None}, results_root=results_root)
            retried += 1
        elif status is RunStatus.BLOCKED:
            failure_class = str((state.get("failure") or {}).get("classification", ""))
            if attempts == 0 and failure_class in {
                FailureClass.GPU_UNAVAILABLE.value, FailureClass.AUTHENTICATION.value,
            }:
                transition_run(state["dataset"], state["model"], RunStatus.PENDING,
                               updates={"failure": None}, results_root=results_root)
                retried += 1
            else:
                raise EngineeringReviewRequired(
                    f"ENGINEERING REVIEW REQUIRED — blocked run {state['run_id']} is not automatically retryable."
                )
    return retried


def run_orchestrator(
    *,
    max_runs: int = 1,
    verbose: bool = False,
    project_root: Path,
    results_root: Path = MODERN_RESULTS_ROOT,
    environment: Mapping[str, str] | None = None,
    services: OrchestratorServices | None = None,
    emit: Callable[[str], None] = print,
) -> dict[str, Any]:
    if max_runs < 1:
        raise ValueError("max_runs must be positive.")
    services = services or OrchestratorServices()
    environment = os.environ if environment is None else environment
    project_root = Path(project_root).resolve()
    _validate_repository(project_root)
    _validate_environment(environment)
    torch, gpu_name, _ = _cuda_profile(services)
    emit(f"CUDA PASS — {gpu_name}")
    _configure_and_verify_git(project_root, environment, services)
    _verify_hugging_face(environment, services)
    emit("BOOTSTRAP PASS")
    initialize_queue(results_root)
    restored = recover_completed_runs(results_root, environment, services)
    emit(f"RECOVERED {restored} COMPLETED RUNS")
    states = refresh_queue(results_root)["runs"]
    dependency_ready = True
    if _needs_neobert_dependency(states):
        ensure_neobert_dependency(services, project_root)
    emit("DEPENDENCIES PASS")
    normalize_retryable_states(results_root, dependency_ready=dependency_ready)
    completed_now = 0
    while completed_now < max_runs:
        queue = refresh_queue(results_root)
        eligible = [state for state in queue["runs"] if state["status"] in {
            RunStatus.RUNNING.value, RunStatus.PENDING.value,
        }]
        if not eligible:
            break
        selected = sorted(eligible, key=lambda item: (item["status"] != RunStatus.RUNNING.value, queue["runs"].index(item)))[0]
        emit(f"STARTING {selected['run_id']}")
        try:
            services.execute(
                selected["dataset"], selected["model"], results_root=results_root,
                sync_github=True,
            )
        except BaseException as exc:
            failed_state = read_json(run_paths(selected["dataset"], selected["model"], results_root).state)
            if failed_state["status"] == RunStatus.FAILED.value and (
                int(failed_state.get("test_evaluation_attempts", 0)) > 0
                or not _known_xformers_failure(failed_state)
                and str((failed_state.get("failure") or {}).get("classification", ""))
                != FailureClass.TRANSIENT_INFRASTRUCTURE.value
            ):
                raise EngineeringReviewRequired(
                    f"ENGINEERING REVIEW REQUIRED — {selected['run_id']} failed: {exc}"
                ) from exc
            raise
        current = read_json(run_paths(selected["dataset"], selected["model"], results_root).state)
        if current["status"] != RunStatus.COMPLETED.value or not _local_completed_valid(
            selected["dataset"], selected["model"], results_root,
        ):
            raise RuntimeError(f"Persistence verification failed for {selected['run_id']}.")
        services.sync(project_root, results_root=results_root)
        completed_now += 1
        emit(f"COMPLETED {selected['run_id']}")
        emit("PERSISTENCE VERIFIED")
        emit(f"QUEUE {matrix_audit(results_root)['completed']}/24 COMPLETE")
    audit = matrix_audit(results_root)
    pending_runs = [row for row in audit["rows"] if row["status"] == RunStatus.PENDING.value]
    summary = {
        "completed": audit["completed"], "pending": audit["pending"],
        "failed": audit["failed"], "blocked": audit["blocked"],
        "next_run": f"{pending_runs[0]['dataset']}__{pending_runs[0]['model']}" if pending_runs else None,
        "phase_4_pass": audit["phase_4_pass"], "completed_this_invocation": completed_now,
    }
    if verbose:
        summary["audit"] = audit
    emit("SUMMARY " + " ".join(f"{key}={value}" for key, value in summary.items() if key != "audit"))
    return summary
