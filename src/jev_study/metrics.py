"""Concordância com o gabarito e curva de confiabilidade.

O runner guarda a decisão crua. Aqui um tipo trocado não vira score, e falha de
transporte deixa o acerto nulo. A fatia low_confidence só lê confidence
declarada em resposta válida; não completa o que o backend omitiu.
"""

import math
from collections.abc import Iterable, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from jev_study.records import RunRecord, nearest_score_level
from jev_study.tasks import DecisionTask, ExpectedLabel

# Limiar do estudo: P(sim) >= 0,5 conta como sim na concordância.
# A curva de Noul não usa este corte; ela compara a probabilidade crua ao rótulo.
NOUL_YES_THRESHOLD = 0.5
DEFAULT_BINS = 10
_TYPE_ORDER = ("choice", "score", "noul", "box")
_EXPECTATIONS = ("clear", "low_confidence")


class EvalItem(BaseModel):
    """Uma pergunta já comparada ao gabarito. matched nulo não é acerto nem erro.

    schema_valid falso cobre tipo trocado e resposta ilegível para a pergunta.
    schema_valid nulo cobre transporte e tarefa pulada, onde não houve corpo.
    """

    model_config = ConfigDict(extra="forbid")

    task_id: str
    source: str
    backend: str
    requested_model: str
    model: str | None = None
    question: str
    question_type: Literal["choice", "score", "noul", "box"]
    expectation: Literal["clear", "low_confidence"] | None = None
    predicted_decision: dict[str, Any] | None = None
    expected_decision: dict[str, Any] | None = None
    matched: bool | None = None
    schema_valid: bool | None = None
    declared_confidence: float | None = None
    declared_probability: float | None = None
    latency_ms: float | None = None
    estimated_cost_usd: float | None = None
    cost_scope: Literal["request"] = "request"
    input_usd_per_mtok: float | None = None
    output_usd_per_mtok: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    skipped: bool = False
    skip_reason: str | None = None
    error: str | None = None


class ReliabilityBucket(BaseModel):
    """Faixa [low, high). A última faixa inclui 1."""

    model_config = ConfigDict(extra="forbid")

    low: float
    high: float
    count: int
    mean_confidence: float | None = None
    outcome_rate: float | None = None


class CalibrationCurve(BaseModel):
    """ECE das faixas não vazias. counted é o que entrou; o resto fica de fora."""

    model_config = ConfigDict(extra="forbid")

    signal: Literal["declared_confidence", "declared_probability"]
    outcome: Literal["correctness", "positive_label"]
    counted: int
    ece: float | None
    buckets: list[ReliabilityBucket]


class TypeMetrics(BaseModel):
    """Concordância e curva de um tipo. Noul não mistura o campo confidence."""

    model_config = ConfigDict(extra="forbid")

    question_type: Literal["choice", "score", "noul", "box"]
    count: int
    scored: int
    matched: int
    agreement: float | None
    invalid: int
    unscored: int
    mean_declared_confidence: float | None
    confidence_count: int
    mean_declared_probability: float | None
    probability_count: int
    calibration: CalibrationCurve


class SliceMetrics(BaseModel):
    """Fatia clear ou low_confidence. A média ignora resposta inválida."""

    model_config = ConfigDict(extra="forbid")

    expectation: Literal["clear", "low_confidence"]
    count: int
    scored: int
    matched: int
    agreement: float | None
    invalid: int
    unscored: int
    mean_declared_confidence: float | None
    confidence_count: int


class BackendMetrics(BaseModel):
    """Métricas de um adapter. Não junta Jev com o baseline na mesma curva."""

    model_config = ConfigDict(extra="forbid")

    backend: str
    count: int
    scored: int
    matched: int
    agreement: float | None
    invalid: int
    unscored: int
    transport_errors: int
    by_question_type: dict[str, TypeMetrics]
    by_expectation: dict[str, SliceMetrics]
    low_confidence_mean_is_lower: bool | None


class EvaluationReport(BaseModel):
    """Itens pareados e as contas por backend. Não é o relatório do study-PR4."""

    model_config = ConfigDict(extra="forbid")

    items: list[EvalItem]
    metrics: list[BackendMetrics]


def parse_prediction(
    question_type: str,
    decision: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Lê a decisão comparável ao gabarito.

    O type do JSON tem de ser o da pergunta. Sem isso o retorno é nulo:
    um noul não é reinterpretado como nível de score.
    """
    if not isinstance(decision, dict) or decision.get("type") != question_type:
        return None
    if question_type == "choice":
        choice = decision.get("choice")
        if not isinstance(choice, str) or not choice.strip():
            return None
        return _compact(ExpectedLabel(choice=choice))
    if question_type == "score":
        score = decision.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            return None
        if not math.isfinite(float(score)):
            return None
        level = nearest_score_level(float(score))
        if not 0 <= level <= 9:
            return None
        return _compact(ExpectedLabel(score_level=level))
    if question_type == "noul":
        noul = decision.get("noul")
        if isinstance(noul, bool) or not isinstance(noul, (int, float)):
            return None
        number = float(noul)
        if not math.isfinite(number) or not 0 <= number <= 1:
            return None
        return _compact(ExpectedLabel(noul=number >= NOUL_YES_THRESHOLD))
    return None


def expected_decision(question_type: str, label: ExpectedLabel | None) -> dict[str, Any] | None:
    """Copia o rótulo no mesmo formato da previsão. Campo de outro tipo não entra."""
    if label is None:
        return None
    if question_type == "choice" and label.choice is not None:
        return _compact(ExpectedLabel(choice=label.choice))
    if question_type == "score" and label.score_level is not None:
        return _compact(ExpectedLabel(score_level=label.score_level))
    if question_type == "noul" and label.noul is not None:
        return _compact(ExpectedLabel(noul=label.noul))
    return None


def score_record(
    record: RunRecord,
    task: DecisionTask,
    *,
    expectation: Literal["clear", "low_confidence"] | None = None,
) -> EvalItem:
    """Monta o item. Copia latência, tokens e custo mesmo quando o acerto fica nulo."""
    expected = expected_decision(record.question_type, task.expected.get(record.question))
    parsed = parse_prediction(record.question_type, record.decision)
    schema_valid = _schema_valid(record, parsed)
    predicted = parsed if schema_valid else None
    # Sem previsão ou sem rótulo o casamento fica nulo: não vira False.
    if schema_valid and predicted is not None and expected is not None:
        matched: bool | None = predicted == expected
    else:
        matched = None
    return EvalItem(
        task_id=record.task_id,
        source=record.source,
        backend=record.backend,
        requested_model=record.requested_model,
        model=record.model,
        question=record.question,
        question_type=record.question_type,
        expectation=expectation,
        predicted_decision=predicted,
        expected_decision=expected,
        matched=matched,
        schema_valid=schema_valid,
        declared_confidence=record.declared_confidence,
        declared_probability=record.declared_probability,
        latency_ms=record.latency_ms,
        estimated_cost_usd=record.estimated_cost_usd,
        cost_scope=record.cost_scope,
        input_usd_per_mtok=record.input_usd_per_mtok,
        output_usd_per_mtok=record.output_usd_per_mtok,
        input_tokens=record.input_tokens,
        output_tokens=record.output_tokens,
        skipped=record.skipped,
        skip_reason=record.skip_reason,
        error=record.error,
    )


def bin_index(confidence: float, bins: int = DEFAULT_BINS) -> int:
    """Índice da faixa. 1 cai na última. O epsilon segura décimos que o float encolhe."""
    if bins < 1:
        raise ValueError("bins < 1")
    if not math.isfinite(confidence) or confidence < 0 or confidence > 1:
        raise ValueError("confiança fora de [0, 1]")
    if confidence == 1:
        return bins - 1
    index = math.floor(confidence * bins + 1e-9)
    return min(bins - 1, max(0, index))


def reliability_diagram(
    pairs: Sequence[tuple[float, bool]],
    *,
    bins: int = DEFAULT_BINS,
    signal: Literal["declared_confidence", "declared_probability"],
    outcome: Literal["correctness", "positive_label"],
) -> CalibrationCurve:
    """Agrupa (número declarado, desfecho) e devolve o ECE ponderado pelo tamanho."""
    if bins < 1:
        raise ValueError("bins < 1")
    groups: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    for confidence, flag in pairs:
        groups[bin_index(confidence, bins)].append((confidence, flag))
    buckets: list[ReliabilityBucket] = []
    counted = 0
    ece_mass = 0.0
    width = 1.0 / bins
    for index, group in enumerate(groups):
        low = index * width
        high = 1.0 if index == bins - 1 else (index + 1) * width
        if not group:
            buckets.append(ReliabilityBucket(low=low, high=high, count=0))
            continue
        mean_confidence = sum(value for value, _ in group) / len(group)
        outcome_rate = sum(1 for _, flag in group if flag) / len(group)
        buckets.append(
            ReliabilityBucket(
                low=low,
                high=high,
                count=len(group),
                mean_confidence=mean_confidence,
                outcome_rate=outcome_rate,
            )
        )
        counted += len(group)
        ece_mass += len(group) * abs(outcome_rate - mean_confidence)
    ece = ece_mass / counted if counted else None
    return CalibrationCurve(
        signal=signal,
        outcome=outcome,
        counted=counted,
        ece=ece,
        buckets=buckets,
    )


def low_confidence_mean_is_lower(items: Sequence[EvalItem]) -> bool | None:
    """True só se a média de confidence na fatia ambígua é estritamente menor.

    Usa o campo confidence de respostas válidas. Noul não entra, porque a API
    não publica esse campo. Empate ou fatia vazia devolve None ou False, sem
    preencher confiança que o backend não declarou.
    """
    clear = _confidence_values(item for item in items if item.expectation == "clear")
    low = _confidence_values(item for item in items if item.expectation == "low_confidence")
    clear_mean = _mean(clear)
    low_mean = _mean(low)
    if clear_mean is None or low_mean is None:
        return None
    return low_mean < clear_mean


def build_report(items: Sequence[EvalItem], *, bins: int = DEFAULT_BINS) -> EvaluationReport:
    """Agrega por backend, na ordem em que os itens aparecem."""
    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 1:
        raise ValueError("bins < 1")
    grouped: dict[str, list[EvalItem]] = {}
    order: list[str] = []
    for item in items:
        if item.backend not in grouped:
            order.append(item.backend)
            grouped[item.backend] = []
        grouped[item.backend].append(item)
    metrics = [_backend_metrics(name, grouped[name], bins) for name in order]
    return EvaluationReport(items=list(items), metrics=metrics)


def _schema_valid(record: RunRecord, parsed: dict[str, Any] | None) -> bool | None:
    # Sem model e sem decisão: a chamada não entregou corpo. Não é tipo inválido.
    if record.skipped or (record.decision is None and record.model is None):
        return None
    if record.error or parsed is None:
        return False
    return True


def _compact(label: ExpectedLabel) -> dict[str, Any]:
    return label.model_dump(mode="json", exclude_none=True)


def _confidence_values(items: Iterable[EvalItem]) -> list[float]:
    return [
        item.declared_confidence
        for item in items
        if item.schema_valid is True and item.declared_confidence is not None
    ]


def _probability_values(items: Sequence[EvalItem]) -> list[float]:
    return [
        item.declared_probability
        for item in items
        if item.schema_valid is True and item.declared_probability is not None
    ]


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _counts(items: Sequence[EvalItem]) -> dict[str, Any]:
    scored = [item for item in items if item.matched is not None]
    matched = sum(1 for item in scored if item.matched)
    return {
        "count": len(items),
        "scored": len(scored),
        "matched": matched,
        "agreement": (matched / len(scored)) if scored else None,
        "invalid": sum(1 for item in items if item.schema_valid is False),
        "unscored": sum(1 for item in items if item.matched is None),
        "transport_errors": sum(
            1
            for item in items
            if item.schema_valid is None and not item.skipped and item.error is not None
        ),
    }


def _curve_for_type(question_type: str, items: Sequence[EvalItem], bins: int) -> CalibrationCurve:
    # Noul: P(sim) contra a frequência do rótulo sim. Choice e Score: confidence contra acerto.
    if question_type == "noul":
        pairs: list[tuple[float, bool]] = []
        for item in items:
            if item.schema_valid is not True or item.declared_probability is None:
                continue
            expected = item.expected_decision or {}
            label = expected.get("noul")
            if not isinstance(label, bool):
                continue
            pairs.append((item.declared_probability, label))
        return reliability_diagram(
            pairs,
            bins=bins,
            signal="declared_probability",
            outcome="positive_label",
        )
    pairs = []
    for item in items:
        if (
            item.schema_valid is not True
            or item.matched is None
            or item.declared_confidence is None
        ):
            continue
        pairs.append((item.declared_confidence, item.matched))
    return reliability_diagram(
        pairs,
        bins=bins,
        signal="declared_confidence",
        outcome="correctness",
    )


def _type_metrics(question_type: str, items: Sequence[EvalItem], bins: int) -> TypeMetrics:
    typed = [item for item in items if item.question_type == question_type]
    counts = _counts(typed)
    confidences = _confidence_values(typed)
    probabilities = _probability_values(typed)
    return TypeMetrics(
        question_type=question_type,
        count=counts["count"],
        scored=counts["scored"],
        matched=counts["matched"],
        agreement=counts["agreement"],
        invalid=counts["invalid"],
        unscored=counts["unscored"],
        mean_declared_confidence=_mean(confidences),
        confidence_count=len(confidences),
        mean_declared_probability=_mean(probabilities),
        probability_count=len(probabilities),
        calibration=_curve_for_type(question_type, typed, bins),
    )


def _slice_metrics(expectation: str, items: Sequence[EvalItem]) -> SliceMetrics:
    selected = [item for item in items if item.expectation == expectation]
    counts = _counts(selected)
    confidences = _confidence_values(selected)
    return SliceMetrics(
        expectation=expectation,
        count=counts["count"],
        scored=counts["scored"],
        matched=counts["matched"],
        agreement=counts["agreement"],
        invalid=counts["invalid"],
        unscored=counts["unscored"],
        mean_declared_confidence=_mean(confidences),
        confidence_count=len(confidences),
    )


def _backend_metrics(name: str, items: Sequence[EvalItem], bins: int) -> BackendMetrics:
    counts = _counts(items)
    present = [kind for kind in _TYPE_ORDER if any(item.question_type == kind for item in items)]
    return BackendMetrics(
        backend=name,
        count=counts["count"],
        scored=counts["scored"],
        matched=counts["matched"],
        agreement=counts["agreement"],
        invalid=counts["invalid"],
        unscored=counts["unscored"],
        transport_errors=counts["transport_errors"],
        by_question_type={kind: _type_metrics(kind, items, bins) for kind in present},
        by_expectation={
            expectation: _slice_metrics(expectation, items) for expectation in _EXPECTATIONS
        },
        low_confidence_mean_is_lower=low_confidence_mean_is_lower(items),
    )
