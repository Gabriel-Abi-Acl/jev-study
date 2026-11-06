"""Campos crus de uma decisão. Concordância e calibração ficam em metrics.py.

confidence é só o campo que a API declara. Noul não tem esse campo, e este módulo
não fabrica um a partir da probabilidade. A probabilidade de score é a do nível
inteiro mais próximo da esperança.
"""

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from jev_study.schema import (
    Answer,
    BoxAnswer,
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
)


class RunRecord(BaseModel):
    """Uma pergunta de uma request. O custo é da request inteira, repetido na linha."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    source: str
    backend: str
    requested_model: str
    model: str | None = None
    question: str
    question_type: Literal["choice", "score", "noul", "box"]
    decision: dict[str, Any] | None = None
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


def declared_confidence(answer: Answer) -> float | None:
    """Campo confidence da API. Noul devolve None de propósito."""
    if isinstance(answer, (ChoiceAnswer, ScoreAnswer, BoxAnswer)):
        return answer.confidence
    return None


def declared_probability(answer: Answer) -> float | None:
    """Probabilidade da decisão pontual, quando o tipo publica distribuição ou P(sim)."""
    if isinstance(answer, NoulAnswer):
        return answer.noul
    if isinstance(answer, ChoiceAnswer):
        return answer.probabilities.get(answer.choice)
    if isinstance(answer, ScoreAnswer):
        return answer.probabilities.get(str(nearest_score_level(answer.score)))
    return None


def nearest_score_level(score: float) -> int:
    """Nível inteiro mais próximo. Empate em .5 vai para o nível de cima."""
    return math.floor(score + 0.5)


def decision_payload(answer: Answer) -> dict[str, Any]:
    """JSON da resposta, inclusive confidence nula de box."""
    return answer.model_dump(mode="json")
