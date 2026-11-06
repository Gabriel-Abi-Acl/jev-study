"""Contrato do runner: registra decisão, confiança, probabilidade, latência e custo da request."""

import pytest

from jev_study.backends.fake import FakeBackend, echo_labels
from jev_study.dataset import load_synthetic_tasks
from jev_study.errors import BackendError
from jev_study.records import RunRecord
from jev_study.runner import run_tasks
from jev_study.schema import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    ScoreAnswer,
    SystemOneResponse,
    Usage,
)
from jev_study.tasks import DecisionTask, ExpectedLabel


def _clock(values):
    iterator = iter(values)

    def clock():
        return next(iterator)

    return clock


def _choice(name: str, label: str) -> tuple[ChoiceQuestion, ExpectedLabel]:
    question = ChoiceQuestion(
        instructions="Tom?",
        criteria={"calm": "neutro", "angry": "hostil"},
    )
    return question, ExpectedLabel(choice=label)


def test_label_echo_runs_the_placeholder_without_calling_skipped_box():
    tasks = load_synthetic_tasks()
    backend = echo_labels(tasks)
    records = run_tasks(backend, tasks)
    assert len(backend.calls) == sum(not task.skip for task in tasks)
    assert len(records) == len(tasks)
    by_id = {record.task_id: record for record in records}
    billing = by_id["route_duplicate_charge"]
    assert billing.backend == "fake"
    assert billing.requested_model == "fake-label-echo"
    assert billing.decision["choice"] == "billing"
    assert billing.declared_confidence == 1
    assert billing.declared_probability == 1
    assert billing.estimated_cost_usd is None
    assert billing.input_tokens is None
    refund = by_id["refund_requested_no"]
    assert refund.decision["noul"] == 0
    assert refund.declared_confidence is None
    assert refund.declared_probability == 0
    skipped = by_id["box_bumper_skipped"]
    assert skipped.skipped is True
    assert skipped.question_type == "box"
    assert skipped.decision is None
    assert skipped.error is None
    assert skipped.model is None
    assert skipped.latency_ms is None


def test_request_cost_is_shared_and_resolved_model_is_kept():
    calm, calm_label = _choice("tone", "calm")
    angry, angry_label = _choice("mood", "angry")
    task = DecisionTask(
        id="dual",
        state="Olá.",
        questions={"tone": calm, "mood": angry},
        expected={"tone": calm_label, "mood": angry_label},
        note="duas perguntas na mesma request",
    )
    response = SystemOneResponse(
        model="jev-1.14.0",
        answers={
            "tone": ChoiceAnswer(
                type="choice",
                choice="calm",
                confidence=0.7,
                probabilities={"calm": 0.8, "angry": 0.2},
            ),
            "mood": ChoiceAnswer(
                type="choice",
                choice="angry",
                confidence=0.4,
                probabilities={"calm": 0.3, "angry": 0.7},
            ),
        },
        usage=Usage(input_tokens=1_000_000, output_tokens=500_000),
    )
    backend = FakeBackend(lambda request: response, name="jev", model="jev-latest")
    records = run_tasks(
        backend,
        [task],
        input_usd_per_mtok=0.042,
        output_usd_per_mtok=0.0,
        clock=_clock([5.0, 5.04]),
    )
    assert backend.calls[0].model == "jev-latest"
    assert [record.question for record in records] == ["tone", "mood"]
    assert records[0].declared_confidence == 0.7
    assert records[0].declared_probability == 0.8
    assert records[0].latency_ms == pytest.approx(40)
    assert records[1].latency_ms == pytest.approx(40)
    assert records[0].estimated_cost_usd == pytest.approx(0.042)
    assert records[1].estimated_cost_usd == pytest.approx(records[0].estimated_cost_usd)
    assert records[0].cost_scope == "request"
    assert records[0].model == "jev-1.14.0"
    assert records[0].requested_model == "jev-latest"
    assert records[0].input_tokens == 1_000_000


def test_backend_error_records_latency_without_inventing_a_decision():
    question, label = _choice("tone", "calm")
    task = DecisionTask(
        id="boom",
        state="Olá.",
        questions={"tone": question, "mood": question},
        expected={"tone": label, "mood": label},
        note="falha",
    )

    def explode(request):
        raise BackendError("HTTP 422: no", status=422)

    backend = FakeBackend(explode, name="jev", model="jev-1.14.0")
    records = run_tasks(backend, [task], clock=_clock([1.0, 1.25]))
    assert len(records) == 2
    assert all(record.error == "HTTP 422: no" for record in records)
    assert all(record.decision is None for record in records)
    assert all(record.estimated_cost_usd is None for record in records)
    assert records[0].latency_ms == pytest.approx(250)
    assert records[1].latency_ms == pytest.approx(250)


def test_missing_mismatched_and_extra_answers():
    question, label = _choice("tone", "calm")
    task = DecisionTask(
        id="partial",
        state="Olá.",
        questions={"tone": question},
        expected={"tone": label},
        note="parcial",
    )
    response = SystemOneResponse(
        model="jev-1.14.0",
        answers={
            "tone": NoulAnswer(type="noul", noul=0.2),
            "extra": ScoreAnswer(
                type="score",
                score=0,
                confidence=0.3,
                legend={"0": "baixo"},
                probabilities={"0": 1.0},
            ),
        },
        usage=Usage(input_tokens=100, output_tokens=1),
    )
    records = run_tasks(
        FakeBackend(lambda request: response, name="jev", model="jev-1.14.0"),
        [task],
        input_usd_per_mtok=0.042,
        output_usd_per_mtok=0.0,
    )
    mismatched = records[0]
    assert mismatched.question == "tone"
    assert mismatched.error is not None
    assert "noul" in mismatched.error
    assert mismatched.decision["noul"] == 0.2
    assert mismatched.declared_confidence is None
    assert mismatched.declared_probability == 0.2
    assert mismatched.estimated_cost_usd == pytest.approx(100 / 1_000_000 * 0.042)
    extra = records[1]
    assert extra.question == "extra"
    assert extra.error == "resposta sem pergunta correspondente"
    assert extra.question_type == "score"

    missing_response = SystemOneResponse(
        model="jev-1.14.0",
        answers={
            "other": ChoiceAnswer(
                type="choice",
                choice="calm",
                confidence=1,
                probabilities={"calm": 1, "angry": 0},
            )
        },
        usage=Usage(input_tokens=100, output_tokens=0),
    )
    missing = run_tasks(
        FakeBackend(lambda request: missing_response, name="jev", model="jev-1.14.0"),
        [task],
        input_usd_per_mtok=0.042,
        output_usd_per_mtok=0.0,
    )
    assert any(record.error == "resposta ausente" and record.decision is None for record in missing)
    assert isinstance(missing[0], RunRecord)


def test_skip_does_not_call_the_backend_or_the_clock():
    task = next(item for item in load_synthetic_tasks() if item.skip)

    def clock():
        raise AssertionError("clock")

    def decide(request):
        raise AssertionError("decide")

    records = run_tasks(
        FakeBackend(decide, name="fake", model="m"),
        [task],
        clock=clock,
    )
    assert len(records) == 1
    assert records[0].skipped is True
    assert records[0].requested_model == "m"
