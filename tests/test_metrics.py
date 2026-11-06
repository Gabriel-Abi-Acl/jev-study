"""Contrato das métricas: parse, concordância, tipo inválido, transporte e calibração.

Os números daqui são contas de exemplo. Não são resultado de modelo.
"""

import pytest

from jev_study.metrics import (
    EvalItem,
    bin_index,
    build_report,
    low_confidence_mean_is_lower,
    parse_prediction,
    reliability_diagram,
    score_record,
)
from jev_study.records import RunRecord
from jev_study.schema import ChoiceQuestion, NoulCriteria, NoulQuestion, ScoreQuestion
from jev_study.tasks import DecisionTask, ExpectedLabel


def _choice_task(label: str = "billing") -> DecisionTask:
    return DecisionTask(
        id="t1",
        source="eval-v1",
        state="Quero o estorno.",
        questions={
            "team": ChoiceQuestion(
                instructions="Qual equipe?",
                criteria={"billing": "cobrança", "technical": "defeito"},
            )
        },
        expected={"team": ExpectedLabel(choice=label)},
        note="estorno",
    )


def _score_task(level: int = 2) -> DecisionTask:
    return DecisionTask(
        id="s1",
        source="eval-v1",
        state="Precisa ficar pronto hoje.",
        questions={
            "urgency": ScoreQuestion(
                instructions="Urgência?",
                criteria=["pode esperar", "nesta semana", "hoje"],
            )
        },
        expected={"urgency": ExpectedLabel(score_level=level)},
        note="hoje",
    )


def _noul_task(label: bool = True) -> DecisionTask:
    return DecisionTask(
        id="n1",
        source="eval-v1",
        state="Peço o estorno.",
        questions={
            "refund_requested": NoulQuestion(
                instructions="Pede estorno?",
                criteria=NoulCriteria.model_validate(
                    {"true": "pede devolução", "false": "não pede"}
                ),
            )
        },
        expected={"refund_requested": ExpectedLabel(noul=label)},
        note="verbo pede",
    )


def _record(task: DecisionTask, **overrides) -> RunRecord:
    question = next(iter(task.questions))
    fields = {
        "task_id": task.id,
        "source": task.source,
        "backend": "fake",
        "requested_model": "fake-1",
        "model": "fake-1",
        "question": question,
        "question_type": task.questions[question].type,
        "latency_ms": 12.5,
        "estimated_cost_usd": None,
        "input_tokens": None,
        "output_tokens": None,
    }
    fields.update(overrides)
    return RunRecord(**fields)


def test_parse_choice_score_tie_and_noul_threshold():
    assert parse_prediction(
        "choice",
        {"type": "choice", "choice": "billing", "confidence": 0.7},
    ) == {"choice": "billing"}
    assert parse_prediction("score", {"type": "score", "score": 1.5}) == {"score_level": 2}
    assert parse_prediction("score", {"type": "score", "score": 1.49}) == {"score_level": 1}
    assert parse_prediction("noul", {"type": "noul", "noul": 0.5}) == {"noul": True}
    assert parse_prediction("noul", {"type": "noul", "noul": 0.0}) == {"noul": False}
    assert parse_prediction("noul", {"type": "noul", "noul": 0.499}) == {"noul": False}


def test_parse_rejects_wrong_type_bool_and_blank_choice():
    assert parse_prediction("score", {"type": "noul", "noul": 0.8}) is None
    assert parse_prediction("noul", {"type": "score", "score": 0.5}) is None
    assert parse_prediction("score", {"type": "score", "score": True}) is None
    assert parse_prediction("score", {"type": "score", "score": "2"}) is None
    assert parse_prediction("choice", {"type": "choice", "choice": "  "}) is None
    assert parse_prediction("choice", None) is None
    assert parse_prediction("box", {"type": "box", "box": {}}) is None


def test_score_record_keeps_operational_fields_and_separates_confidence():
    task = _choice_task()
    record = _record(
        task,
        decision={
            "type": "choice",
            "choice": "billing",
            "confidence": 0.7,
            "probabilities": {"billing": 0.8, "technical": 0.2},
        },
        declared_confidence=0.7,
        declared_probability=0.8,
        estimated_cost_usd=0.042,
        input_tokens=1_000_000,
        output_tokens=0,
        input_usd_per_mtok=0.042,
        output_usd_per_mtok=0.0,
    )
    item = score_record(record, task, expectation="clear")
    assert item.matched is True
    assert item.schema_valid is True
    assert item.predicted_decision == {"choice": "billing"}
    assert item.expected_decision == {"choice": "billing"}
    assert item.declared_confidence == 0.7
    assert item.declared_probability == 0.8
    assert item.latency_ms == 12.5
    assert item.input_tokens == 1_000_000
    assert item.output_tokens == 0
    assert item.estimated_cost_usd == 0.042
    assert item.backend == "fake"
    assert item.question_type == "choice"


def test_choice_mismatch_stays_in_the_agreement_denominator():
    task = _choice_task("billing")
    record = _record(
        task,
        decision={"type": "choice", "choice": "technical", "confidence": 0.4},
        declared_confidence=0.4,
        declared_probability=0.6,
    )
    item = score_record(record, task, expectation="clear")
    assert item.schema_valid is True
    assert item.matched is False
    assert item.predicted_decision == {"choice": "technical"}


def test_wrong_type_is_invalid_and_not_coerced_into_a_score():
    task = _score_task(2)
    record = _record(
        task,
        decision={"type": "noul", "noul": 0.8},
        declared_confidence=None,
        declared_probability=0.8,
        error="tipo da resposta (noul) difere da pergunta (score)",
    )
    item = score_record(record, task, expectation="clear")
    assert item.schema_valid is False
    assert item.matched is None
    assert item.predicted_decision is None
    assert item.expected_decision == {"score_level": 2}
    assert item.declared_probability == 0.8


def test_transport_error_leaves_correctness_null():
    task = _score_task(2)
    record = _record(
        task,
        model=None,
        decision=None,
        declared_confidence=None,
        declared_probability=None,
        latency_ms=250.0,
        estimated_cost_usd=None,
        input_tokens=None,
        output_tokens=None,
        error="HTTP 503: down",
    )
    item = score_record(record, task, expectation="low_confidence")
    assert item.schema_valid is None
    assert item.matched is None
    assert item.predicted_decision is None
    assert item.expected_decision == {"score_level": 2}
    assert item.latency_ms == 250.0
    assert item.estimated_cost_usd is None
    assert item.input_tokens is None
    report = build_report([item])
    metrics = report.metrics[0]
    assert metrics.agreement is None
    assert metrics.scored == 0
    assert metrics.transport_errors == 1
    assert metrics.by_question_type["score"].calibration.counted == 0


def test_missing_answer_is_invalid_not_a_transport_error():
    task = _choice_task()
    record = _record(
        task,
        decision=None,
        error="resposta ausente",
    )
    item = score_record(record, task, expectation="clear")
    assert item.schema_valid is False
    assert item.matched is None
    report = build_report([item])
    assert report.metrics[0].invalid == 1
    assert report.metrics[0].transport_errors == 0
    assert report.metrics[0].agreement is None


def test_invalid_and_transport_stay_out_of_agreement_and_ece():
    valid = _item("choice", True, 0.9, expectation="clear")
    mismatch = _item("choice", False, 0.4, expectation="clear")
    invalid = _item("choice", None, 0.1, expectation="low_confidence", schema_valid=False)
    transport = _item(
        "choice",
        None,
        None,
        expectation="low_confidence",
        schema_valid=None,
        error="HTTP 422: no",
    )
    report = build_report([valid, mismatch, invalid, transport])
    metrics = report.metrics[0]
    assert metrics.scored == 2
    assert metrics.matched == 1
    assert metrics.agreement == 0.5
    assert metrics.invalid == 1
    assert metrics.unscored == 2
    assert metrics.transport_errors == 1
    curve = metrics.by_question_type["choice"].calibration
    assert curve.counted == 2
    assert curve.signal == "declared_confidence"
    assert curve.outcome == "correctness"
    # A confidence 0,1 do item inválido não entra na faixa baixa.
    assert curve.buckets[1].count == 0
    assert low_confidence_mean_is_lower(report.items) is None


def test_choice_curve_uses_confidence_not_the_selected_probability():
    item = _item("choice", True, 0.7, probability=0.8)
    curve = build_report([item]).metrics[0].by_question_type["choice"].calibration
    assert curve.buckets[7].count == 1
    assert curve.buckets[8].count == 0
    assert curve.buckets[7].outcome_rate == 1


def test_noul_curve_bins_raw_probability_against_the_yes_label():
    yes = _item("noul", True, None, probability=0.9, expected={"noul": True})
    no = _item("noul", True, None, probability=0.1, expected={"noul": False})
    coerced = EvalItem(
        task_id="bad",
        source="eval-v1",
        backend="fake",
        requested_model="fake-1",
        question="refund_requested",
        question_type="noul",
        expectation="clear",
        predicted_decision=None,
        expected_decision={"noul": True},
        matched=None,
        schema_valid=False,
        declared_confidence=0.2,
        declared_probability=0.8,
        error="tipo da resposta (score) difere da pergunta (noul)",
    )
    curve = build_report([yes, no, coerced]).metrics[0].by_question_type["noul"].calibration
    assert curve.signal == "declared_probability"
    assert curve.outcome == "positive_label"
    assert curve.counted == 2
    assert curve.buckets[9].count == 1
    assert curve.buckets[9].outcome_rate == 1
    assert curve.buckets[1].count == 1
    assert curve.buckets[1].mean_confidence == pytest.approx(0.1)
    assert curve.buckets[1].outcome_rate == 0
    assert curve.ece == pytest.approx(0.1)


def test_reliability_diagram_ece_and_endpoint_bins():
    curve = reliability_diagram(
        [(0.95, True), (0.95, True), (0.55, False), (0.25, False)],
        signal="declared_confidence",
        outcome="correctness",
    )
    assert curve.buckets[9].count == 2
    assert curve.buckets[9].mean_confidence == pytest.approx(0.95)
    assert curve.buckets[9].outcome_rate == 1
    assert curve.buckets[5].count == 1
    assert curve.buckets[2].count == 1
    assert curve.ece == pytest.approx(0.225)
    assert bin_index(0.0) == 0
    assert bin_index(0.1) == 1
    assert bin_index(0.3) == 3
    assert bin_index(1.0) == 9
    empty = reliability_diagram([], signal="declared_confidence", outcome="correctness")
    assert empty.ece is None
    assert empty.counted == 0
    assert len(empty.buckets) == 10


def test_low_confidence_slice_reads_only_declared_confidence():
    clear = _item("choice", True, 0.9, expectation="clear")
    low = _item("choice", False, 0.3, expectation="low_confidence")
    tied = _item("score", True, 0.3, expectation="low_confidence", question="urgency")
    same = _item("choice", True, 0.9, expectation="low_confidence")
    assert low_confidence_mean_is_lower([clear, low]) is True
    assert low_confidence_mean_is_lower([clear, same]) is False
    noul_clear = _item(
        "noul",
        True,
        None,
        probability=0.99,
        expectation="clear",
        expected={"noul": True},
    )
    noul_low = _item(
        "noul",
        True,
        None,
        probability=0.51,
        expectation="low_confidence",
        expected={"noul": True},
    )
    assert low_confidence_mean_is_lower([noul_clear, noul_low]) is None
    report = build_report([clear, low, tied])
    slices = report.metrics[0].by_expectation
    assert slices["clear"].confidence_count == 1
    assert slices["clear"].mean_declared_confidence == pytest.approx(0.9)
    assert slices["low_confidence"].confidence_count == 2
    assert slices["low_confidence"].mean_declared_confidence == pytest.approx(0.3)
    assert report.metrics[0].low_confidence_mean_is_lower is True


def test_empty_report_has_no_agreement():
    report = build_report([])
    assert report.items == []
    assert report.metrics == []


def _item(
    question_type: str,
    matched: bool | None,
    confidence: float | None,
    *,
    probability: float | None = None,
    expectation: str = "clear",
    schema_valid: bool | None = True,
    expected: dict | None = None,
    question: str = "team",
    error: str | None = None,
) -> EvalItem:
    if expected is None and question_type == "choice":
        expected = {"choice": "billing"}
    if matched is None:
        predicted = None
    elif matched:
        predicted = expected
    elif question_type == "choice":
        predicted = {"choice": "technical"}
    else:
        predicted = None
    return EvalItem(
        task_id=f"{question_type}-{expectation}-{confidence}",
        source="eval-v1",
        backend="fake",
        requested_model="fake-1",
        question=question,
        question_type=question_type,
        expectation=expectation,
        predicted_decision=predicted,
        expected_decision=expected,
        matched=matched,
        schema_valid=schema_valid,
        declared_confidence=confidence,
        declared_probability=probability if probability is not None else confidence,
        error=error,
    )
