from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ml.modern_benchmark_queue import atomic_json, initialize_queue, read_json, run_paths
from ml.phase4_persistence import (
    checkpoint_repo_path,
    lightweight_files,
    persist_completed_checkpoint,
    restore_checkpoint,
    restore_lightweight_from_github,
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
    atomic_json(paths.result, {"status": "COMPLETED"})
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


def test_restore_downloads_only_selected_checkpoint_tokenizer_and_config(tmp_path: Path) -> None:
    initialize_queue(tmp_path / "results")
    hub = FakeHub(token=TOKEN)
    prefix = checkpoint_repo_path("imdb", "eurobert_210m")
    hub.files = {
        f"{prefix}/checkpoints/best/model/model.safetensors",
        f"{prefix}/tokenizer/tokenizer.json",
        f"{prefix}/config/config.json",
        f"{prefix}/result.json",
    }
    cache = tmp_path / "cache"

    def download(*, filename: str, **_: Any) -> str:
        path = cache / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(filename, encoding="utf-8")
        return str(path)

    restored = restore_checkpoint(
        "imdb", "eurobert_210m", results_root=tmp_path / "results",
        environment=ENVIRONMENT, api_factory=lambda **_: hub, download_function=download,
    )
    paths = run_paths("imdb", "eurobert_210m", tmp_path / "results")
    assert restored["verified"] is True and restored["restored_files"] == 3
    assert (paths.checkpoints / "best" / "model" / "model.safetensors").is_file()
    assert (paths.root / "tokenizer" / "tokenizer.json").is_file()
    assert not paths.result.exists()
    assert read_json(paths.state)["best_checkpoint"] == str(paths.checkpoints / "best" / "model")


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
