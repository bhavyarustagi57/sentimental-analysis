from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any

import pytest

from ml.modern_benchmark_queue import (
    atomic_json,
    initialize_queue,
    matrix_audit,
    prediction_sha256,
    read_json,
    run_paths,
)
from ml.phase4_persistence import (
    checkpoint_repo_path,
    lightweight_files,
    persist_completed_checkpoint,
    restore_checkpoint,
    restore_lightweight_from_github,
    validate_git_identity,
)


TOKEN = "hf_test_secret_that_must_never_reach_disk"
ENVIRONMENT = {"PHASE4_HF_REPO": "owner/private-phase4", "HF_TOKEN": TOKEN}


class FakeHub:
    def __init__(self, **kwargs: Any) -> None:
        self.constructor_token = kwargs.get("token")
        self.files: set[str] = set()

    def create_repo(self, **_: Any) -> None:
        return None

    def upload_folder(self, *, folder_path: str | Path, path_in_repo: str, **_: Any) -> Any:
        root = Path(folder_path)
        allowed_roots = ("checkpoints/best/model/", "tokenizer/", "config/")
        allowed_files = {"config.json", "state.json", "result.json", "training_history.json", "predictions.jsonl"}
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if relative in allowed_files or relative.startswith(allowed_roots):
                self.files.add(f"{path_in_repo}/{relative}")
        return SimpleNamespace(oid="fake-commit")

    def upload_file(self, *, path_in_repo: str, **_: Any) -> Any:
        self.files.add(path_in_repo)
        return SimpleNamespace(oid="fake-manifest-commit")

    def list_repo_files(self, *_: Any, **__: Any) -> list[str]:
        return sorted(self.files)


def _completed_fixture(root: Path) -> tuple[Any, FakeHub]:
    initialize_queue(root)
    paths = run_paths("imdb", "eurobert_210m", root)
    (paths.checkpoints / "best" / "model").mkdir(parents=True)
    (paths.checkpoints / "best" / "model" / "model.safetensors").write_bytes(b"weights")
    (paths.root / "tokenizer").mkdir()
    (paths.root / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
    (paths.root / "config").mkdir()
    (paths.root / "config" / "config.json").write_text("{}", encoding="utf-8")
    lines = [
        json.dumps({"row_id": f"row-{index}", "label": index % 2, "prediction": index % 2,
                    "logits": [1.0, 0.0] if index % 2 == 0 else [0.0, 1.0]})
        for index in range(2_000)
    ]
    paths.predictions.write_text("\n".join(lines) + "\n", encoding="utf-8")
    history = [{"epoch": 3, "global_step": 675}]
    atomic_json(paths.history, history)
    config = read_json(paths.config)
    persistence = {"repo_id": ENVIRONMENT["PHASE4_HF_REPO"], "repo_type": "model",
                   "path_in_repo": checkpoint_repo_path("imdb", "eurobert_210m"),
                   "commit": "fake-commit", "verified": True, "verified_file_count": 10,
                   "persisted_at": "2026-01-01T00:00:00+00:00"}
    atomic_json(paths.root / "persistence_manifest.json", persistence)
    stale = "/content/old-runtime/results/modern/staging/imdb__eurobert_210m/checkpoints/best/model"
    result = {
        "schema_version": 1, "dataset": "imdb", "model": "eurobert_210m", "status": "COMPLETED",
        "accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1_score": 1.0,
        "validation_f1": 0.9, "validation_accuracy": 0.9, "validation_loss": 0.2,
        "training_time_seconds": 10.0, "inference_time_seconds": 2.0,
        "examples_per_second": 1_000.0, "trainable_parameters": 10, "total_parameters": 100,
        "adaptation_mode": config["adaptation_mode"], "final_epoch": 3, "final_step": 675,
        "device": "cuda", "device_name": "Tesla T4", "numeric_precision": "bf16",
        "checkpoint_identifier": stale, "model_id": config["model_id"],
        "model_revision": config["model_revision"], "tokenizer_revision": config["tokenizer_revision"],
        "split_manifest_hash": config["split_manifest_hash"], "prediction_count": 2_000,
        "prediction_sha256": prediction_sha256(paths.predictions), "dataset_fingerprint": "fixture",
        "preprocessing_version": "1.0", "seed": 42,
        "labels": {"0": "negative", "1": "positive"}, "max_sequence_length": 256,
        "truncation_rate": {"train": 0.0, "validation": 0.0, "test": 0.0},
        "checkpoint_reload_verified": True, "training_history": history,
        "training_configuration": {"effective_batch_size": 32}, "oom_adjustments": [],
        "hardware": {"python": "3.12", "torch": "fixture", "transformers": "fixture",
                     "accelerate": "fixture", "peft": "fixture", "safetensors": "fixture",
                     "device_type": "cuda", "device_name": "Tesla T4", "precision": "bf16"},
        "training_timing_definition": "fixture", "inference_timing_definition": "fixture",
        "checkpoint_persistence": persistence, "completed_at": "2026-01-01T00:00:00+00:00",
    }
    atomic_json(paths.result, result)
    state = dict(read_json(paths.state))
    state.update({"status": "COMPLETED", "test_evaluation_attempts": 1,
                  "best_checkpoint": stale, "last_checkpoint": stale,
                  "current_epoch": 3, "current_step": 675,
                  "completed_at": result["completed_at"]})
    atomic_json(paths.state, state)
    hub = FakeHub(token=TOKEN)
    return paths, hub


def test_completed_result_upload_is_verified_and_excludes_resume_checkpoint(tmp_path: Path) -> None:
    paths, hub = _completed_fixture(tmp_path)
    resume = paths.checkpoints / "checkpoint-step-000100"
    resume.mkdir()
    (resume / "optimizer.pt").write_bytes(b"large optimizer")

    metadata = persist_completed_checkpoint(
        "imdb", "eurobert_210m", results_root=tmp_path,
        environment=ENVIRONMENT, api_factory=lambda **_: hub,
    )

    assert metadata["verified"] is True
    assert metadata["commit"] == "fake-commit"
    assert any(name.endswith("model.safetensors") for name in hub.files)
    assert not any("checkpoint-step" in name or "optimizer.pt" in name for name in hub.files)
    assert (paths.root / "persistence_manifest.json").is_file()


def test_checkpoint_upload_path_is_stable_and_run_specific() -> None:
    assert checkpoint_repo_path("imdb", "eurobert_210m") == "phase4/imdb__eurobert_210m"
    assert checkpoint_repo_path("yelp_polarity", "qwen3_0_6b") == "phase4/yelp_polarity__qwen3_0_6b"


def _hub_from_completed_source(source: Path) -> tuple[FakeHub, Any]:
    paths, hub = _completed_fixture(source)
    prefix = checkpoint_repo_path("imdb", "eurobert_210m")
    for path in paths.root.rglob("*"):
        if path.is_file() and "checkpoint-step" not in path.as_posix():
            hub.files.add(f"{prefix}/{path.relative_to(paths.root).as_posix()}")

    def download(*, filename: str, **_: Any) -> str:
        relative = PurePosixPath(filename).relative_to(PurePosixPath(prefix))
        return str(paths.root.joinpath(*relative.parts))

    return hub, download


def test_full_completed_run_restore_repairs_paths_and_rebuilds_queue_and_canonical(tmp_path: Path, monkeypatch: Any) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    hub, download = _hub_from_completed_source(source)
    monkeypatch.setattr("ml.modern_benchmark.execute_final_run", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("training called")))

    restored = restore_checkpoint(
        "imdb", "eurobert_210m", results_root=destination,
        environment=ENVIRONMENT, api_factory=lambda **_: hub, download_function=download,
    )
    paths = run_paths("imdb", "eurobert_210m", destination)
    assert restored["verified"] is True
    assert (paths.checkpoints / "best" / "model" / "model.safetensors").is_file()
    assert (paths.root / "tokenizer" / "tokenizer.json").is_file()
    assert paths.predictions.is_file() and paths.result.is_file() and paths.history.is_file()
    assert (paths.root / "persistence_manifest.json").is_file()
    state, result = read_json(paths.state), read_json(paths.result)
    assert state["status"] == "COMPLETED" and state["test_evaluation_attempts"] == 1
    assert state["best_checkpoint"] == result["checkpoint_identifier"] == str(paths.checkpoints / "best" / "model")
    audit = matrix_audit(destination)
    assert audit["completed"] == 1 and audit["pending"] == 23
    assert len(read_json(destination / "results.json")) == 1


@pytest.mark.parametrize("missing", ["predictions.jsonl", "result.json"])
def test_full_restore_rejects_missing_required_evidence_without_changing_queue(tmp_path: Path, missing: str) -> None:
    source, destination = tmp_path / "source", tmp_path / "destination"
    hub, download = _hub_from_completed_source(source)
    prefix = checkpoint_repo_path("imdb", "eurobert_210m")
    hub.files.remove(f"{prefix}/{missing}")
    with pytest.raises(FileNotFoundError, match="missing required files"):
        restore_checkpoint("imdb", "eurobert_210m", results_root=destination,
                           environment=ENVIRONMENT, api_factory=lambda **_: hub, download_function=download)
    assert matrix_audit(destination)["completed"] == 0


def test_full_restore_rejects_corrupted_prediction_hash_without_marking_completed(tmp_path: Path) -> None:
    source, destination = tmp_path / "source", tmp_path / "destination"
    hub, download = _hub_from_completed_source(source)
    source_result = run_paths("imdb", "eurobert_210m", source).result
    result = dict(read_json(source_result))
    result["prediction_sha256"] = "0" * 64
    atomic_json(source_result, result)
    with pytest.raises(ValueError, match="hash"):
        restore_checkpoint("imdb", "eurobert_210m", results_root=destination,
                           environment=ENVIRONMENT, api_factory=lambda **_: hub, download_function=download)
    assert matrix_audit(destination)["completed"] == 0


def test_full_restore_rejects_duplicate_final_test_attempt(tmp_path: Path) -> None:
    source, destination = tmp_path / "source", tmp_path / "destination"
    hub, download = _hub_from_completed_source(source)
    source_state = run_paths("imdb", "eurobert_210m", source).state
    state = dict(read_json(source_state))
    state["test_evaluation_attempts"] = 2
    atomic_json(source_state, state)
    with pytest.raises(ValueError, match="exactly one final-test attempt"):
        restore_checkpoint("imdb", "eurobert_210m", results_root=destination,
                           environment=ENVIRONMENT, api_factory=lambda **_: hub, download_function=download)
    assert matrix_audit(destination)["completed"] == 0


def test_git_identity_preflight_accepts_configured_identity_and_rejects_missing(tmp_path: Path) -> None:
    values = {"user.name": "Academic Runner", "user.email": "runner@example.invalid"}

    def configured(command: list[str], **_: Any) -> Any:
        return SimpleNamespace(returncode=0, stdout=values[command[-1]] + "\n")

    assert validate_git_identity(tmp_path, runner=configured) == values

    def missing(_command: list[str], **_: Any) -> Any:
        return SimpleNamespace(returncode=1, stdout="")

    with pytest.raises(RuntimeError, match="Git user.name is not configured"):
        validate_git_identity(tmp_path, runner=missing)


def test_git_allowlist_and_written_metadata_never_contain_secret(tmp_path: Path) -> None:
    project = tmp_path / "project"
    results = project / "results" / "modern"
    paths, hub = _completed_fixture(results)
    persist_completed_checkpoint(
        "imdb", "eurobert_210m", results_root=results,
        environment=ENVIRONMENT, api_factory=lambda **_: hub,
    )
    allowed = lightweight_files(project, results)
    assert paths.result in allowed and paths.state in allowed and paths.config in allowed
    assert not any("checkpoints" in path.parts or path.name.endswith(".safetensors") for path in allowed)
    for path in project.rglob("*"):
        if path.is_file():
            assert TOKEN.encode() not in path.read_bytes()
    ignore = (Path(__file__).resolve().parents[1] / ".gitignore").read_text(encoding="utf-8")
    assert "*.safetensors" in ignore and "*.pt" in ignore
    assert "results/modern/staging/*/checkpoints/" in ignore


def test_lightweight_restore_uses_fast_forward_git_pull_without_credentials_in_arguments(tmp_path: Path) -> None:
    calls: list[tuple[list[str], Path, bool]] = []

    def runner(command: list[str], *, cwd: Path, check: bool) -> Any:
        calls.append((command, cwd, check))
        return SimpleNamespace(returncode=0)

    restore_lightweight_from_github(tmp_path, runner=runner)
    assert calls == [(["git", "pull", "--ff-only"], tmp_path.resolve(), True)]
    assert TOKEN not in json.dumps(calls, default=str)
