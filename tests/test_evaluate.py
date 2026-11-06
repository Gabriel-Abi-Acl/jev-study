"""Contrato do harness: as 45 tarefas com fakes, sem rede e sem número de estudo no stdout fixo.

O eco de gabarito casa com o rótulo por construção. Isso não mede o Jev nem o baseline.
"""

import json
import subprocess
import sys

import pytest

from jev_study.backends.fake import FakeBackend, echo_labels
from jev_study.dataset import load_eval_decisions
from jev_study.errors import BackendError
from jev_study.evaluate import BackendRun, evaluate_backends, main
from jev_study.schema import (
    NoulAnswer,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
    dump_request,
)

STUDY_ENV = (
    "TYPESAFE_API_KEY",
    "JEV_API_KEY",
    "TYPESAFE_BASE_URL",
    "JEV_MODEL",
    "JEV_INPUT_USD_PER_MTOK",
    "JEV_OUTPUT_USD_PER_MTOK",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_INPUT_USD_PER_MTOK",
    "LLM_OUTPUT_USD_PER_MTOK",
)


def _isolate(monkeypatch):
    for name in STUDY_ENV:
        monkeypatch.delenv(name, raising=False)


def _clock(values):
    iterator = iter(values)

    def clock():
        return next(iterator)

    return clock


def test_label_echo_scores_every_v1_task_without_sending_the_label():
    decisions = load_eval_decisions()
    assert len(decisions) == 45
    tasks = [row.as_decision_task() for row in decisions]
    backend = echo_labels(tasks)
    report = evaluate_backends([BackendRun(backend)], decisions)
    assert len(backend.calls) == 45
    assert len(report.items) == 45
    assert {item.question_type for item in report.items} == {"choice", "score", "noul"}
    assert [item.question_type for item in report.items].count("choice") == 15
    assert [item.question_type for item in report.items].count("score") == 15
    assert [item.question_type for item in report.items].count("noul") == 15
    assert sum(item.expectation == "low_confidence" for item in report.items) == 6
    assert all(item.schema_valid is True and item.matched is True for item in report.items)
    assert all(item.estimated_cost_usd is None for item in report.items)
    assert all(item.input_tokens is None and item.output_tokens is None for item in report.items)
    assert all(item.latency_ms is not None and item.latency_ms >= 0 for item in report.items)
    assert all(item.error is None for item in report.items)
    for item in report.items:
        if item.question_type == "noul":
            assert item.declared_confidence is None
        else:
            assert item.declared_confidence == 1
    metrics = report.metrics[0]
    assert metrics.backend == "fake"
    assert metrics.scored == 45
    assert metrics.agreement == 1
    assert metrics.invalid == 0
    assert metrics.transport_errors == 0
    assert metrics.low_confidence_mean_is_lower is False
    assert metrics.by_expectation["low_confidence"].count == 6
    assert metrics.by_expectation["clear"].count == 39
    assert metrics.by_question_type["noul"].calibration.signal == "declared_probability"
    assert metrics.by_question_type["choice"].calibration.signal == "declared_confidence"
    dumped = dump_request(backend.calls[0])
    assert set(dumped) == {"model", "state", "questions"}
    raw = json.dumps(dumped)
    assert "expected" not in raw
    assert "low_confidence" not in raw
    assert "notes" not in raw


def test_two_fakes_stay_paired_and_are_not_pooled():
    decisions = load_eval_decisions()
    tasks = [row.as_decision_task() for row in decisions]
    jev = echo_labels(tasks, name="jev", model="jev-fake")
    llm = echo_labels(tasks, name="llm", model="llm-fake")
    report = evaluate_backends([BackendRun(jev), BackendRun(llm)], decisions)
    assert len(report.items) == 90
    assert [metrics.backend for metrics in report.metrics] == ["jev", "llm"]
    assert all(metrics.count == 45 for metrics in report.metrics)
    ids = {item.task_id for item in report.items if item.backend == "jev"}
    assert ids == {item.task_id for item in report.items if item.backend == "llm"}


def test_usage_latency_and_cost_are_copied_when_present():
    decisions = load_eval_decisions()[:1]
    task = decisions[0].as_decision_task()
    sample = echo_labels([task], name="jev", model="jev-1.14.0").decide(
        SystemOneRequest(model="jev-1.14.0", state=task.state, questions=task.questions)
    )
    response = sample.model_copy(update={"usage": Usage(input_tokens=1_000_000, output_tokens=0)})
    backend = FakeBackend(lambda request: response, name="jev", model="jev-1.14.0")
    report = evaluate_backends(
        [BackendRun(backend, input_usd_per_mtok=0.042, output_usd_per_mtok=0.0)],
        decisions,
        clock=_clock([1.0, 1.01]),
    )
    item = report.items[0]
    assert item.latency_ms == pytest.approx(10)
    assert item.input_tokens == 1_000_000
    assert item.output_tokens == 0
    assert item.estimated_cost_usd == pytest.approx(0.042)
    assert item.matched is True


def test_mismatched_type_stays_unscored_through_the_harness():
    decision = next(row for row in load_eval_decisions() if row.category == "score")

    def respond(request):
        name = next(iter(request.questions))
        return SystemOneResponse(
            model="fake-1",
            answers={name: NoulAnswer(type="noul", noul=0.8)},
        )

    report = evaluate_backends(
        [BackendRun(FakeBackend(respond, name="fake", model="fake-1"))],
        [decision],
    )
    item = report.items[0]
    assert item.question_type == "score"
    assert item.schema_valid is False
    assert item.matched is None
    assert item.predicted_decision is None
    assert item.expected_decision == {"score_level": decision.expected[item.question].score_level}
    assert report.metrics[0].agreement is None
    assert report.metrics[0].invalid == 1
    assert report.metrics[0].by_question_type["score"].calibration.counted == 0


def test_transport_error_is_not_turned_into_a_score():
    decision = next(row for row in load_eval_decisions() if row.category == "score")

    def respond(request):
        raise BackendError("HTTP 503: down", status=503)

    report = evaluate_backends(
        [BackendRun(FakeBackend(respond, name="jev", model="jev-1.14.0"))],
        [decision],
        clock=_clock([2.0, 2.25]),
    )
    item = report.items[0]
    assert item.matched is None
    assert item.schema_valid is None
    assert item.predicted_decision is None
    assert item.expected_decision["score_level"] == decision.expected[item.question].score_level
    assert item.error == "HTTP 503: down"
    assert item.latency_ms == pytest.approx(250)
    assert item.estimated_cost_usd is None
    assert report.metrics[0].transport_errors == 1
    assert report.metrics[0].scored == 0
    assert report.metrics[0].agreement is None


def test_fake_command_prints_items_and_metrics_without_network(monkeypatch, capsys):
    _isolate(monkeypatch)

    def boom(*args, **kwargs):
        raise AssertionError("rede")

    monkeypatch.setattr("urllib.request.OpenerDirector.open", boom)
    sys.modules.pop("openai", None)
    assert main([]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload) == {"items", "metrics"}
    assert len(payload["items"]) == 45
    item = payload["items"][0]
    for key in (
        "backend",
        "question_type",
        "predicted_decision",
        "expected_decision",
        "matched",
        "schema_valid",
        "declared_confidence",
        "declared_probability",
        "latency_ms",
        "input_tokens",
        "output_tokens",
        "estimated_cost_usd",
    ):
        assert key in item
    assert payload["metrics"][0]["backend"] == "fake"
    assert payload["metrics"][0]["low_confidence_mean_is_lower"] is False
    assert "openai" not in sys.modules


def test_live_backends_without_keys_exit_before_the_network(monkeypatch, capsys):
    _isolate(monkeypatch)
    sys.modules.pop("openai", None)
    assert main(["--backend", "jev"]) == 2
    assert main(["--backend", "llm"]) == 2
    assert main(["--backend", "all"]) == 2
    error = capsys.readouterr().err
    assert "TYPESAFE_API_KEY" in error
    assert "LLM_API_KEY" in error
    assert "openai" not in sys.modules


def test_bins_below_one_exits_before_loading_work(monkeypatch, capsys):
    _isolate(monkeypatch)
    assert main(["--bins", "0"]) == 2
    assert "bins" in capsys.readouterr().err


def test_module_entry_runs_the_fake_harness_offline(monkeypatch):
    _isolate(monkeypatch)
    completed = subprocess.run(
        [sys.executable, "-m", "jev_study.evaluate", "--backend", "fake"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert len(payload["items"]) == 45
    assert payload["metrics"][0]["backend"] == "fake"
