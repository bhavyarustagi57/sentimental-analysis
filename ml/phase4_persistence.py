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
    """Restore selected checkpoint/tokenizer/config without restoring optimizer state."""
    repo_id, token = _credentials(environment)
    prefix = checkpoint_repo_path(dataset, model)
    if api_factory is None or download_function is None:
        from huggingface_hub import HfApi, hf_hub_download

        api_factory = api_factory or HfApi
        download_function = download_function or hf_hub_download
    api = api_factory(token=token)
    remote_files = [
        name for name in api.list_repo_files(repo_id, repo_type="model", token=token)
        if name.startswith(prefix + "/")
    ]
    restorable = [
        name for name in remote_files
        if any(name.startswith(f"{prefix}/{part}/") for part in ("checkpoints/best/model", "tokenizer", "config"))
    ]
    if not restorable or not any(name.endswith(".safetensors") for name in restorable):
        raise FileNotFoundError(f"No verified selected checkpoint exists at {repo_id}/{prefix}.")
    root = run_paths(dataset, model, results_root).root
    for remote_name in restorable:
        downloaded = Path(download_function(
            repo_id=repo_id, filename=remote_name, repo_type="model", token=token,
        ))
        relative = PurePosixPath(remote_name).relative_to(PurePosixPath(prefix))
        destination = root.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(downloaded, destination)
    paths = run_paths(dataset, model, results_root)
    restored_checkpoint = paths.checkpoints / "best" / "model"
    state = dict(read_json(paths.state))
    state["best_checkpoint"] = str(restored_checkpoint)
    atomic_json(paths.state, state)
    if paths.result.is_file():
        result = dict(read_json(paths.result))
        result["checkpoint_identifier"] = str(restored_checkpoint)
        atomic_json(paths.result, result)
    refresh_queue(results_root)
    if RunStatus(state["status"]) is RunStatus.COMPLETED and paths.result.is_file():
        rebuild_canonical_results(results_root)
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
