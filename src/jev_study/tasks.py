"""Tarefa rotulada do estudo. O request da API não carrega o gabarito."""

import json
from collections.abc import Mapping
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from jev_study.schema import (
    ChoiceQuestion,
    NoulQuestion,
    Question,
    ScoreQuestion,
)


class ExpectedLabel(BaseModel):
    """Um rótulo por pergunta. Noul guarda o booleano, não uma probabilidade.

    A regra planejada para o PR3 é tratar noul >= 0,5 como sim. O PR1 não aplica
    esse limiar; ele só conserva o rótulo e a probabilidade que o backend declarar.
    """

    model_config = ConfigDict(extra="forbid")

    choice: str | None = None
    score_level: int | None = Field(default=None, ge=0, le=9)
    noul: bool | None = None

    @model_validator(mode="after")
    def exactly_one(self) -> Self:
        present = [
            self.choice is not None,
            self.score_level is not None,
            self.noul is not None,
        ]
        if sum(present) != 1:
            raise ValueError("gabarito deve preencher um único campo")
        return self


class DecisionTask(BaseModel):
    """Item pareado: o mesmo state e as mesmas questions vão para os dois backends."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    source: str = Field(min_length=1, default="synthetic")
    state: str | dict[str, Any] | list[Any]
    questions: dict[str, Question] = Field(min_length=1)
    expected: dict[str, ExpectedLabel] = Field(default_factory=dict)
    skip: bool = False
    skip_reason: str | None = None
    note: str = ""

    @model_validator(mode="after")
    def labels_follow_questions(self) -> Self:
        if not self.id.strip():
            raise ValueError("id vazio")
        if self.skip:
            if not (self.skip_reason and self.skip_reason.strip()):
                raise ValueError("skip exige motivo")
            if self.expected:
                raise ValueError("tarefa pulada não leva gabarito")
            return self
        if self.skip_reason:
            raise ValueError("motivo de skip só existe quando a tarefa é pulada")
        for name, question in self.questions.items():
            if not name.strip():
                raise ValueError("nome de pergunta vazio")
            label = self.expected.get(name)
            if label is None:
                raise ValueError(f"gabarito ausente: {name}")
            _check_label(question, label)
        extra = set(self.expected) - set(self.questions)
        if extra:
            raise ValueError(f"gabarito sem pergunta: {sorted(extra)}")
        return self

    def fingerprint(self) -> str:
        """Chave estável de state + questions, sem o campo model."""
        return request_fingerprint(self.state, self.questions)


def request_fingerprint(state: object, questions: Mapping[str, Question]) -> str:
    """Identifica o material da decisão. O nome do modelo não entra."""
    payload = {
        "state": state,
        "questions": {
            name: question.model_dump(mode="json", by_alias=True, exclude_none=True)
            for name, question in questions.items()
        },
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _check_label(question: Question, label: ExpectedLabel) -> None:
    if isinstance(question, ChoiceQuestion):
        if label.choice is None or label.choice not in question.criteria:
            raise ValueError("gabarito choice incompatível com os critérios")
        return
    if isinstance(question, ScoreQuestion):
        if label.score_level is None or not 0 <= label.score_level < len(question.criteria):
            raise ValueError("score_level fora da rubrica")
        return
    if isinstance(question, NoulQuestion):
        if label.noul is None:
            raise ValueError("gabarito noul incompatível")
        return
    raise ValueError("box não tem gabarito enquanto a tarefa não traz imagem")
