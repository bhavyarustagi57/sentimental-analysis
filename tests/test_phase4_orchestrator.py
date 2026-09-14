from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from ml.modern_benchmark_queue import RunStatus, atomic_json, initialize_queue, read_json, run_paths, transition_run
from ml.phase4_orchestrator import (
    EngineeringReviewRequired,
    OrchestratorServices,
    ensure_neobert_dependency,
    normalize_retryable_states,
    recover_completed_runs,
    run_orchestrator,
)


ENV = {
    "HF_TOKEN": "hf_fake_not_written",
    "PHASE4_HF_REPO": "owner/private-phase4",
    "GH_TOKEN": "github_fake_not_written",
    "PHASE4_GIT_NAME": "Academic Runner",
    "PHASE4_GIT_EMAIL": "runner@example.invalid",
}


@pytest.fixture(autouse=True)
def _isolate_canonical_rebuild(monkeypatch: Any) -> None:
    monkeypatch.setattr("ml.phase4_orchestrator.rebuild_canonical_results", lambda *_args, **_kwargs: [])


class FakeCuda:
    def __init__(self) -> None:
        self.checks = 0

    def is_available(self) -> bool:
        self.checks += 1
        return True

    def get_device_name(self, _: int) -> str:
        return "Tesla T4"


class FakeTorch:
    __version__ = "2.14.0"

    def __init__(self) -> None:
        self.cuda = FakeCuda()


class FakeApi:
    def __init__(self, files: set[str] | None = None, **_: Any) -> None:
        self.files = files or set()

    def whoami(self, **_: Any) -> dict[str, str]: return {"name": "fixture"}
    def repo_info(self, *_: Any, **__: Any) -> object: return object()
    def model_info(self, *_: Any, **__: Any) -> object: return object()
    def list_repo_files(self, *_: Any, **__: Any) -> list[str]: return sorted(self.files)


def _project(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    (project / ".git").mkdir(parents=True)
    (project / "ml").mkdir()
    (project / "scripts").mkdir()
    (project / "ml" / "modern_benchmark.py").write_text("", encoding="utf-8")
    (project / "scripts" / "sync_phase4.py").write_text("", encoding="utf-8")
    return project


def _runner(events: list[str], *, install_hook: Any | None = None) -> Any:
    def run(command: list[str], **_: Any) -> Any:
        events.append("command:" + " ".join(command))
        if command[:3] == ["git", "config", "--get"]:
            return SimpleNamespace(returncode=0, stdout="configured\n")
        if command[:3] == ["git", "remote", "get-url"]:
            return SimpleNamespace(returncode=0, stdout="https://github.com/owner/repo.git\n")
        if "pip" in command and install_hook:
            install_hook()
        return SimpleNamespace(returncode=0, stdout="")
    return run


def _services(events: list[str], execute: Any | None = None, api: FakeApi | None = None) -> OrchestratorServices:
    return OrchestratorServices(
        runner=_runner(events),
        execute=execute or (lambda *_args, **_kwargs: None),
        restore=lambda *_args, **_kwargs: None,
        sync=lambda *_args, **_kwargs: events.append("sync"),
        api_factory=lambda **_: api or FakeApi(),
        torch_module=FakeTorch(),
        import_module=lambda _name: object(),
        package_version=lambda _name: "0.0.35",
    )


def _mark_all_completed(results: Path) -> None:
    initialize_queue(results)
    for state in read_json(results / "queue.json")["runs"]:
        path = run_paths(state["dataset"], state["model"], results).state
        value = dict(read_json(path))
        value["status"] = "COMPLETED"
        atomic_json(path, value)


def test_fresh_runtime_bootstrap_skips_completed_and_is_secret_safe(tmp_path: Path, monkeypatch: Any) -> None:
    project, events = _project(tmp_path), []
    results = project / "results" / "modern"
    _mark_all_completed(results)
    monkeypatch.setattr("ml.phase4_orchestrator._local_completed_valid", lambda *_: True)
    summary = run_orchestrator(project_root=project, results_root=results, environment=ENV,
                               services=_services(events), emit=lambda message: events.append(message))
    assert summary["completed"] == 24 and summary["completed_this_invocation"] == 0
    assert not any(ENV["HF_TOKEN"] in event or ENV["GH_TOKEN"] in event for event in events)


def test_missing_hf_or_github_secret_fails_before_execution(tmp_path: Path) -> None:
    project, events = _project(tmp_path), []
    services = _services(events, execute=lambda *_a, **_k: pytest.fail("execution must not start"))
    with pytest.raises(RuntimeError, match="HF_TOKEN"):
        run_orchestrator(project_root=project, results_root=project / "results" / "modern",
                         environment={**ENV, "HF_TOKEN": ""}, services=services)
    with pytest.raises(RuntimeError, match="GH_TOKEN or GITHUB_TOKEN"):
        run_orchestrator(project_root=project, results_root=project / "results2" / "modern",
                         environment={key: value for key, value in ENV.items() if key != "GH_TOKEN"}, services=services)


def test_dependency_decision_installs_locked_xformers_without_dependencies_and_revalidates_cuda(tmp_path: Path) -> None:
    events: list[str] = []
    installed = {"value": False}
    services = _services(events)
    services.runner = _runner(events, install_hook=lambda: installed.update(value=True))
    services.package_version = lambda _name: "0.0.35" if installed["value"] else (_ for _ in ()).throw(ImportError())
    services.import_module = lambda _name: object() if installed["value"] else (_ for _ in ()).throw(ImportError())
    assert ensure_neobert_dependency(services, tmp_path) is True
    assert any("--no-deps xformers==0.0.35" in event for event in events)
    assert services.torch_module.cuda.checks >= 2


def test_neobert_missing_xformers_failure_becomes_retryable_only_before_test(tmp_path: Path) -> None:
    initialize_queue(tmp_path)
    transition_run("imdb", "neobert", RunStatus.RUNNING, results_root=tmp_path)
    transition_run("imdb", "neobert", RunStatus.FAILED,
                   updates={"failure": {"classification": "model_runtime_defect",
                                        "message": "No module named xformers"}}, results_root=tmp_path)
    assert normalize_retryable_states(tmp_path, dependency_ready=True) == 1
    assert read_json(run_paths("imdb", "neobert", tmp_path).state)["status"] == "PENDING"


def test_unknown_or_post_test_failure_stops_for_engineering_review(tmp_path: Path) -> None:
    initialize_queue(tmp_path)
    transition_run("imdb", "neobert", RunStatus.RUNNING,
                   updates={"test_evaluation_attempts": 1}, results_root=tmp_path)
    transition_run("imdb", "neobert", RunStatus.FAILED,
                   updates={"failure": {"classification": "unknown", "message": "unknown"}}, results_root=tmp_path)
    with pytest.raises(EngineeringReviewRequired, match="ENGINEERING REVIEW REQUIRED"):
        normalize_retryable_states(tmp_path, dependency_ready=True)


def test_remote_completed_evidence_is_restored_even_when_github_state_is_pending(tmp_path: Path, monkeypatch: Any) -> None:
    results = tmp_path / "results"
    initialize_queue(results)
    run = "imdb__eurobert_210m"
    files = {f"phase4/{run}/{name}" for name in
             ("persistence_manifest.json", "result.json", "state.json", "predictions.jsonl")}

    def restore(dataset: str, model: str, **_: Any) -> None:
        state_path = run_paths(dataset, model, results).state
        state = dict(read_json(state_path)); state["status"] = "COMPLETED"; atomic_json(state_path, state)

    services = _services([], api=FakeApi(files))
    services.restore = restore
    monkeypatch.setattr("ml.phase4_orchestrator._local_completed_valid",
                        lambda dataset, model, _root: read_json(run_paths(dataset, model, results).state)["status"] == "COMPLETED")
    assert recover_completed_runs(results, ENV, services) == 1


def test_bounded_loop_persists_each_success_before_next_and_is_idempotent(tmp_path: Path, monkeypatch: Any) -> None:
    project, events = _project(tmp_path), []
    results = project / "results" / "modern"

    def execute(dataset: str, model: str, **_: Any) -> None:
        events.append("execute:" + f"{dataset}__{model}")
        transition_run(dataset, model, RunStatus.RUNNING, results_root=results)
        transition_run(dataset, model, RunStatus.COMPLETED, results_root=results)

    services = _services(events, execute=execute)
    monkeypatch.setattr("ml.phase4_orchestrator._local_completed_valid",
                        lambda dataset, model, _root: read_json(run_paths(dataset, model, results).state)["status"] == "COMPLETED")
    first = run_orchestrator(max_runs=2, project_root=project, results_root=results,
                             environment=ENV, services=services, emit=lambda _message: None)
    assert first["completed_this_invocation"] == 2
    actions = [event for event in events if event.startswith("execute:") or event == "sync"]
    assert actions[:4] == ["execute:imdb__eurobert_210m", "sync", "execute:imdb__ettin_encoder_150m", "sync"]
    second = run_orchestrator(max_runs=1, project_root=project, results_root=results,
                              environment=ENV, services=services, emit=lambda _message: None)
    assert second["completed_this_invocation"] == 1
    assert events.count("execute:imdb__eurobert_210m") == 1


def test_failure_in_second_run_preserves_first_completed_run(tmp_path: Path, monkeypatch: Any) -> None:
    project, events = _project(tmp_path), []
    results = project / "results" / "modern"
    calls = {"count": 0}

    def execute(dataset: str, model: str, **_: Any) -> None:
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("second run failed")
        transition_run(dataset, model, RunStatus.RUNNING, results_root=results)
        transition_run(dataset, model, RunStatus.COMPLETED, results_root=results)

    monkeypatch.setattr("ml.phase4_orchestrator._local_completed_valid",
                        lambda dataset, model, _root: read_json(run_paths(dataset, model, results).state)["status"] == "COMPLETED")
    with pytest.raises(RuntimeError, match="second run failed"):
        run_orchestrator(max_runs=2, project_root=project, results_root=results,
                         environment=ENV, services=_services(events, execute=execute), emit=lambda _message: None)
    assert read_json(run_paths("imdb", "eurobert_210m", results).state)["status"] == "COMPLETED"
    assert "sync" in events
