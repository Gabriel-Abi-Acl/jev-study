"""Schema do POST /v1/systemone e da resposta tipada.

Os dois backends devolvem o mesmo objeto. Validade aqui é campo obrigatório,
faixa e tipo, no formato de https://docs.typesafe.ai/api/system-one.
"""

import math
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from jev_study.errors import SchemaError


class ApiModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")


class LooseModel(BaseModel):
    """Resposta da API: campo extra não derruba o contrato já publicado."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


def _unit_interval(value: float, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} não é número")
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{label} fora de [0, 1]")
    return number


class NoulCriteria(ApiModel):
    """Critério de sim/não. No JSON as chaves são os literais true e false."""

    true_case: str | dict[str, Any] | list[Any] | None = Field(default=None, alias="true")
    false_case: str | dict[str, Any] | list[Any] | None = Field(default=None, alias="false")


class NoulQuestion(ApiModel):
    type: Literal["noul"] = "noul"
    instructions: str | dict[str, Any] | list[Any] | None = None
    criteria: NoulCriteria | None = None


class ChoiceQuestion(ApiModel):
    type: Literal["choice"] = "choice"
    instructions: str | dict[str, Any] | list[Any] | None = None
    criteria: dict[str, Any] = Field(min_length=1, max_length=255)

    @model_validator(mode="after")
    def choice_names(self) -> Self:
        if any(not str(name).strip() for name in self.criteria):
            raise ValueError("nome de choice vazio")
        return self


class ScoreQuestion(ApiModel):
    type: Literal["score"] = "score"
    instructions: str | dict[str, Any] | list[Any] | None = None
    criteria: list[Any] = Field(min_length=1, max_length=10)


class BoxQuestion(ApiModel):
    type: Literal["box"] = "box"
    instructions: str | dict[str, Any] | list[Any] | None = None


Question = Annotated[
    NoulQuestion | ChoiceQuestion | ScoreQuestion | BoxQuestion,
    Field(discriminator="type"),
]


class Usage(LooseModel):
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class NoulAnswer(LooseModel):
    type: Literal["noul"]
    noul: float

    @model_validator(mode="after")
    def probability_in_range(self) -> Self:
        self.noul = _unit_interval(self.noul, "noul")
        return self


class ChoiceAnswer(LooseModel):
    type: Literal["choice"]
    choice: str
    confidence: float
    probabilities: dict[str, float] = Field(min_length=1)

    @model_validator(mode="after")
    def selected_is_listed(self) -> Self:
        self.confidence = _unit_interval(self.confidence, "confidence")
        cleaned: dict[str, float] = {}
        for name, probability in self.probabilities.items():
            cleaned[name] = _unit_interval(probability, f"probabilities.{name}")
        self.probabilities = cleaned
        if self.choice not in self.probabilities:
            raise ValueError("choice ausente em probabilities")
        return self


class ScoreAnswer(LooseModel):
    type: Literal["score"]
    score: float
    confidence: float
    legend: dict[str, Any] = Field(min_length=1)
    probabilities: dict[str, float] = Field(min_length=1)

    @model_validator(mode="after")
    def score_inside_legend(self) -> Self:
        self.confidence = _unit_interval(self.confidence, "confidence")
        if not math.isfinite(self.score):
            raise ValueError("score não é finito")
        levels: list[int] = []
        for key in self.legend:
            if not str(key).isdigit():
                raise ValueError("chave de legend não é nível inteiro")
            levels.append(int(key))
        low, high = min(levels), max(levels)
        if self.score < low or self.score > high:
            raise ValueError("score fora da legend")
        cleaned: dict[str, float] = {}
        for name, probability in self.probabilities.items():
            cleaned[str(name)] = _unit_interval(probability, f"probabilities.{name}")
        self.probabilities = cleaned
        return self


class BoxCoordinates(LooseModel):
    top: float
    bottom: float
    left: float
    right: float

    @model_validator(mode="after")
    def edges(self) -> Self:
        self.top = _unit_interval(self.top, "top")
        self.bottom = _unit_interval(self.bottom, "bottom")
        self.left = _unit_interval(self.left, "left")
        self.right = _unit_interval(self.right, "right")
        if self.top > self.bottom or self.left > self.right:
            raise ValueError("arestas da box invertidas")
        return self


class BoxAnswer(LooseModel):
    type: Literal["box"]
    box: BoxCoordinates
    confidence: float | None

    @model_validator(mode="after")
    def optional_confidence(self) -> Self:
        if self.confidence is not None:
            self.confidence = _unit_interval(self.confidence, "confidence")
        return self


Answer = Annotated[
    NoulAnswer | ChoiceAnswer | ScoreAnswer | BoxAnswer,
    Field(discriminator="type"),
]


class SystemOneRequest(ApiModel):
    """Corpo do POST /v1/systemone. state é o material; questions são independentes."""

    model: str = Field(min_length=1)
    state: str | dict[str, Any] | list[Any]
    questions: dict[str, Question] = Field(min_length=1)
    assets: dict[str, Any] | None = None

    @model_validator(mode="after")
    def question_names(self) -> Self:
        if any(not name.strip() for name in self.questions):
            raise ValueError("nome de pergunta vazio")
        return self


class SystemOneResponse(LooseModel):
    """Resposta publicada. usage pode faltar no baseline; o cliente Jev exige o campo."""

    model: str = Field(min_length=1)
    answers: dict[str, Answer] = Field(min_length=1)
    usage: Usage | None = None


def dump_request(request: SystemOneRequest) -> dict[str, Any]:
    """JSON do POST, com alias true/false e sem campos nulos."""
    return request.model_dump(mode="json", by_alias=True, exclude_none=True)


def parse_response(payload: bytes | str | dict[str, Any]) -> SystemOneResponse:
    """Interpreta o JSON tipado. Qualquer desvio vira SchemaError."""
    try:
        if isinstance(payload, dict):
            return SystemOneResponse.model_validate(payload)
        return SystemOneResponse.model_validate_json(payload)
    except ValidationError as exc:
        raise SchemaError(str(exc)) from exc
