"""Zero-cost durable storage for Phase 4 lightweight evidence and checkpoints."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping

from .modern_benchmark_queue import (
    MODERN_RESULTS_ROOT,
    RunStatus,
    atomic_json,
    initialize_queue,
    locked_run_config,
    read_json,
    rebuild_canonical_results,
    refresh_queue,
    run_id,
    run_paths,
    utc_now,
)


HF_REPO_ENV = "PHASE4_HF_REPO"
HF_TOKEN_ENV = "HF_TOKEN"
REMOTE_ROOT = "phase4"


def checkpoint_repo_path(dataset: str, model: str) -> str:
    """Return the stable Hub directory for one selected final checkpoint."""
    return str(PurePosixPath(REMOTE_ROOT) / run_id(dataset, model))


def _credentials(environment: Mapping[str, str] | None = None) -> tuple[str, str]:
    values = os.environ if environment is None else environment
    repo_id = values.get(HF_REPO_ENV, "").strip()
    token = values.get(HF_TOKEN_ENV, "").strip()
    if not repo_id:
        raise RuntimeError(f"{HF_REPO_ENV} must name the private checkpoint repository.")
    if not token:
        raise RuntimeError(f"{HF_TOKEN_ENV} is required but is never persisted by this project.")
    return repo_id, token


def validate_persistence_environment(environment: Mapping[str, str] | None = None) -> str:
    """Fail before expensive training when durable private storage is not configured."""
    repo_id, _ = _credentials(environment)
    return repo_id


def _expected_upload_files(root: Path) -> list[Path]:
    selected: list[Path] = []
    for relative in ("checkpoints/best/model", "tokenizer", "config"):
        directory = root / relative
        if not directory.is_dir():
            raise FileNotFoundError(f"Required completed-run directory is missing: {directory}")
        selected.extend(path for path in directory.rglob("*") if path.is_file())
    if not any(path.suffix == ".safetensors" for path in selected):
        raise FileNotFoundError("Selected best checkpoint contains no safetensors weights.")
    for name in ("config.json", "state.json", "result.json"):
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(f"Required completed-run metadata is missing: {path}")
        selected.append(path)
    for name in ("training_history.json", "predictions.jsonl"):
        path = root / name
        if path.is_file():
            selected.append(path)
    return selected


def persist_completed_checkpoint(
    dataset: str,
    model: str,
    *,
    results_root: Path = MODERN_RESULTS_ROOT,
    environment: Mapping[str, str] | None = None,
    api_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Upload and verify only durable final artifacts; never uploads resume state."""
    repo_id, token = _credentials(environment)
    paths = run_paths(dataset, model, results_root)
    files = _expected_upload_files(paths.root)
    prefix = checkpoint_repo_path(dataset, model)
    if api_factory is None:
        from huggingface_hub import HfApi

        api_factory = HfApi
    api = api_factory(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True, token=token)
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=paths.root,
        path_in_repo=prefix,
        allow_patterns=[
            "checkpoints/best/model/**", "tokenizer/**", "config/**",
            "config.json", "state.json", "result.json", "training_history.json", "predictions.jsonl",
        ],
        commit_message=f"Persist Phase 4 {run_id(dataset, model)}",
        token=token,
    )
    remote_files = set(api.list_repo_files(repo_id, repo_type="model", token=token))
    expected_remote = {
        str(PurePosixPath(prefix) / path.relative_to(paths.root).as_posix()) for path in files
    }
    missing = sorted(expected_remote - remote_files)
    if missing:
        raise RuntimeError(f"Hugging Face checkpoint upload verification failed: {missing}")
    metadata = {
        "repo_id": repo_id,
        "repo_type": "model",
        "path_in_repo": prefix,
        "commit": str(getattr(commit, "oid", "") or getattr(commit, "commit_id", "") or "unknown"),
        "verified": True,
        "verified_file_count": len(expected_remote),
        "persisted_at": utc_now(),
    }
    manifest = paths.root / "persistence_manifest.json"
    atomic_json(manifest, metadata)
    api.upload_file(
        path_or_fileobj=manifest,
        path_in_repo=str(PurePosixPath(prefix) / manifest.name),
        repo_id=repo_id,
        repo_type="model",
        token=token,
        commit_message=f"Verify Phase 4 {run_id(dataset, model)} persistence",
    )
    if str(PurePosixPath(prefix) / manifest.name) not in set(
        api.list_repo_files(repo_id, repo_type="model", token=token)
    ):
        raise RuntimeError("Hugging Face persistence manifest verification failed.")
    return metadata


def restore_checkpoint(
    dataset: str,
    model: str,
    *,
    results_root: Path = MODERN_RESULTS_ROOT,
    environment: Mapping[str, str] | None = None,
    api_factory: Callable[..., Any] | None = None,
    download_function: Callable[..., str] | None = None,
) -> dict[str, Any]:
    """Atomically reconstruct and validate one completed run from Hub evidence."""
    repo_id, token = _credentials(environment)
    prefix = checkpoint_repo_path(dataset, model)
    initialize_queue(results_root)
    if api_factory is None or download_function is None:
        from huggingface_hub import HfApi, hf_hub_download

        api_factory = api_factory or HfApi
        download_function = download_function or hf_hub_download
    api = api_factory(token=token)
    remote_files = [
        name for name in api.list_repo_files(repo_id, repo_type="model", token=token)
        if name.startswith(prefix + "/")
    ]
    root_files = {"config.json", "state.json", "result.json", "predictions.jsonl",
                  "training_history.json", "persistence_manifest.json"}
    restorable = [name for name in remote_files if (
        PurePosixPath(name).relative_to(PurePosixPath(prefix)).as_posix() in root_files
        or any(name.startswith(f"{prefix}/{part}/") for part in ("checkpoints/best/model", "tokenizer", "config"))
    )]
    relative_names = {
        PurePosixPath(name).relative_to(PurePosixPath(prefix)).as_posix() for name in restorable
    }
    missing_metadata = sorted(root_files - relative_names)
    if missing_metadata:
        raise FileNotFoundError(f"Persisted completed run is missing required files: {missing_metadata}")
    if not any(name.endswith(".safetensors") for name in restorable):
        raise FileNotFoundError(f"No selected safetensors checkpoint exists at {repo_id}/{prefix}.")
    if not any(name.startswith(f"{prefix}/tokenizer/") for name in restorable):
        raise FileNotFoundError("Persisted completed run has no tokenizer files.")
    if f"{prefix}/config/config.json" not in restorable:
        raise FileNotFoundError("Persisted completed run has no saved model config.")
    paths = run_paths(dataset, model, results_root)
    temporary = paths.root.parent / f".{paths.root.name}.restore-incomplete"
    backup = paths.root.parent / f".{paths.root.name}.restore-previous"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)
    for remote_name in restorable:
        downloaded = Path(download_function(
            repo_id=repo_id, filename=remote_name, repo_type="model", token=token,
        ))
        relative = PurePosixPath(remote_name).relative_to(PurePosixPath(prefix))
        destination = temporary.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloaded, destination)
    expected_config = locked_run_config(dataset, model)
    restored_config = read_json(temporary / "config.json")
    if restored_config != expected_config:
        raise ValueError("Persisted run configuration differs from the locked model/split configuration.")
    state = dict(read_json(temporary / "state.json"))
    result = dict(read_json(temporary / "result.json"))
    manifest = dict(read_json(temporary / "persistence_manifest.json"))
    expected_id = run_id(dataset, model)
    if state.get("run_id") != expected_id or state.get("dataset") != dataset or state.get("model") != model:
        raise ValueError("Persisted state identity does not match the requested run.")
    if result.get("dataset") != dataset or result.get("model") != model or result.get("status") != RunStatus.COMPLETED.value:
        raise ValueError("Persisted result identity/status does not match the requested completed run.")
    if RunStatus(state.get("status")) not in {RunStatus.RUNNING, RunStatus.FAILED, RunStatus.COMPLETED}:
        raise ValueError("Persisted state is not a completed or post-evaluation recoverable state.")
    if int(state.get("test_evaluation_attempts", 0)) != 1:
        raise ValueError("Completed-run restore requires exactly one final-test attempt.")
    if manifest.get("verified") is not True or manifest.get("repo_id") != repo_id or manifest.get("path_in_repo") != prefix:
        raise ValueError("Persisted checkpoint manifest is inconsistent with the requested Hub location.")
    result["checkpoint_persistence"] = manifest
    temporary_checkpoint = temporary / "checkpoints" / "best" / "model"
    result["checkpoint_identifier"] = str(temporary_checkpoint)
    atomic_json(temporary / "result.json", result)
    from .modern_benchmark_queue import validate_completed_result

    validate_completed_result(result, temporary / "predictions.jsonl", expected_config=expected_config)
    if read_json(temporary / "training_history.json") != result["training_history"]:
        raise ValueError("Persisted training history differs from completed result metadata.")
    restored_checkpoint = paths.checkpoints / "best" / "model"
    result["checkpoint_identifier"] = str(restored_checkpoint)
    recovered_failure = state.get("failure") if RunStatus(state["status"]) is RunStatus.FAILED else state.get("recovered_failure")
    state.update({
        "status": RunStatus.COMPLETED.value,
        "best_checkpoint": str(restored_checkpoint),
        "last_checkpoint": None,
        "current_epoch": result["final_epoch"],
        "current_step": result["final_step"],
        "failure": None,
        "recovered_failure": recovered_failure,
        "completed_at": result["completed_at"],
    })
    atomic_json(temporary / "result.json", result)
    atomic_json(temporary / "state.json", state)
    if backup.exists():
        shutil.rmtree(backup)
    paths.root.rename(backup)
    temporary.rename(paths.root)
    try:
        queue = refresh_queue(results_root)
        completed_states = [item for item in queue["runs"] if item["status"] == RunStatus.COMPLETED.value]
        all_completed_evidence_local = all(
            run_paths(item["dataset"], item["model"], results_root).result.is_file()
            and run_paths(item["dataset"], item["model"], results_root).predictions.is_file()
            and (run_paths(item["dataset"], item["model"], results_root).checkpoints / "best" / "model").is_dir()
            for item in completed_states
        )
        if all_completed_evidence_local:
            rebuild_canonical_results(results_root)
    except BaseException:
        if paths.root.exists():
            shutil.rmtree(paths.root)
        backup.rename(paths.root)
        refresh_queue(results_root)
        raise
    shutil.rmtree(backup)
    return {"repo_id": repo_id, "path_in_repo": prefix, "restored_files": len(restorable), "verified": True}


def lightweight_files(project_root: Path, results_root: Path = MODERN_RESULTS_ROOT) -> list[Path]:
    """Return the exact Git-safe evidence allowlist; model/tokenizer files are excluded."""
    root = Path(results_root)
    candidates = [root / "queue.json", root / "results.json", root / "results.csv",
                  root / "comparison.json", root / "frontend_data.json"]
    for run_root in (root / "staging").glob("*") if (root / "staging").is_dir() else ():
        candidates.extend(run_root / name for name in (
            "state.json", "config.json", "result.json", "training_history.json",
            "predictions.jsonl", "persistence_manifest.json",
        ))
    project = Path(project_root).resolve()
    return sorted(path for path in candidates if path.is_file() and path.resolve().is_relative_to(project))


def sync_lightweight_to_github(
    project_root: Path,
    *,
    results_root: Path = MODERN_RESULTS_ROOT,
    runner: Callable[..., Any] = subprocess.run,
) -> bool:
    """Commit and push only allowlisted lightweight evidence using existing Git credentials."""
    project = Path(project_root).resolve()
    validate_git_identity(project, runner=runner)
    files = lightweight_files(project, results_root)
    if not files:
        raise FileNotFoundError("No lightweight Phase 4 evidence exists to sync.")
    relative = [str(path.resolve().relative_to(project)) for path in files]
    runner(["git", "add", "--", *relative], cwd=project, check=True)
    changed = runner(["git", "diff", "--cached", "--quiet", "--", *relative], cwd=project).returncode != 0
    if not changed:
        return False
    runner(["git", "commit", "-m", "Persist Phase 4 benchmark state", "--only", "--", *relative], cwd=project, check=True)
    runner(["git", "push"], cwd=project, check=True)
    return True


def restore_lightweight_from_github(
    project_root: Path, *, runner: Callable[..., Any] = subprocess.run,
) -> None:
    """Fast-forward the Git checkout using its preconfigured credential helper."""
    runner(["git", "pull", "--ff-only"], cwd=Path(project_root).resolve(), check=True)


def validate_git_identity(
    project_root: Path, *, runner: Callable[..., Any] = subprocess.run,
) -> dict[str, str]:
    """Fail before GPU work when a fresh runtime cannot create persistence commits."""
    project = Path(project_root).resolve()
    identity: dict[str, str] = {}
    for key in ("user.name", "user.email"):
        completed = runner(
            ["git", "config", "--get", key], cwd=project, check=False,
            capture_output=True, text=True,
        )
        value = str(getattr(completed, "stdout", "")).strip()
        if completed.returncode != 0 or not value:
            raise RuntimeError(
                f"Git {key} is not configured. Before GPU work run: "
                f"git config {key} \"YOUR VALUE\" (use your own non-secret identity)."
            )
        identity[key] = value
    return identity
