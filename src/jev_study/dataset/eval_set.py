"""Gabarito versionado das decisões sintéticas.

O JSONL é a fonte. O loader só aceita linha que caiba no schema do adapter
e num rótulo compatível com a pergunta. `low_confidence` é desenho do item,
não probabilidade medida e não entra no request.
"""

import json
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from jev_study.schema import Question
from jev_study.tasks import DecisionTask, ExpectedLabel

EVAL_SOURCE = "eval-v1"
SCHEMA_VERSION = 1
_RELATIVE_PATH = Path("data") / "eval" / "decisions.v1.jsonl"


class DatasetError(ValueError):
    """Linha do gabarito que não entra no schema ou repete outra linha."""


class EvalDecision(BaseModel):
    """Item do arquivo v1. A categoria é o tipo da pergunta, para o conjunto equilibrar.

    O rótulo fica em `expected`. A expectativa de confiança não é saída de modelo.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    id: str = Field(min_length=1)
    category: Literal["choice", "score", "noul"]
    state: str | dict[str, Any] | list[Any]
    questions: dict[str, Question] = Field(min_length=1)
    expected: dict[str, ExpectedLabel] = Field(min_length=1)
    notes: str = ""
    expectation: Literal["clear", "low_confidence"]

    @model_validator(mode="after")
    def labels_match_category(self) -> Self:
        if not self.id.strip():
            raise ValueError("id vazio")
        if _blank_state(self.state):
            raise ValueError("state vazio")
        kinds = {question.type for question in self.questions.values()}
        if kinds != {self.category}:
            raise ValueError("categoria difere do tipo das perguntas")
        if self.expectation == "low_confidence" and not self.notes.strip():
            raise ValueError("expectativa low_confidence exige nota")
        # DecisionTask recusa gabarito fora da rubrica e pergunta sem rótulo.
        try:
            self.as_decision_task()
        except ValidationError as exc:
            raise ValueError(_format_errors(exc)) from exc
        return self

    def as_decision_task(self) -> DecisionTask:
        """Tarefa que o runner já aceita. Expectativa e nota não vão no POST."""
        return DecisionTask(
            id=self.id,
            source=EVAL_SOURCE,
            state=self.state,
            questions=self.questions,
            expected=self.expected,
            note=self.notes,
        )


def default_eval_path() -> Path:
    """Acha data/eval/decisions.v1.jsonl subindo a partir deste módulo."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / _RELATIVE_PATH
        if candidate.is_file():
            return candidate
    raise DatasetError("decisions.v1.jsonl não encontrado a partir do pacote")


def load_eval_decisions(path: str | Path | None = None) -> list[EvalDecision]:
    """Lê o v1 e recusa linha quebrada, id repetido ou o mesmo estado e perguntas.

    Sem path, usa o arquivo versionado do repositório. O gabarito não entra
    no request: quem chama o backend usa as_decision_task.
    """
    source = Path(path) if path is not None else default_eval_path()
    if not source.is_file():
        raise DatasetError(f"arquivo ausente: {source.name}")
    raw = source.read_text(encoding="utf-8")
    if raw.startswith("\ufeff"):
        raise DatasetError("BOM não é aceito no JSONL")
    if not raw.strip():
        raise DatasetError("arquivo vazio")

    rows: list[EvalDecision] = []
    seen_ids: set[str] = set()
    seen_fingerprints: set[str] = set()
    for number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            raise DatasetError(f"linha {number} vazia")
        row = _parse_line(number, line)
        if row.id in seen_ids:
            raise DatasetError(f"linha {number}: id duplicado ({row.id})")
        fingerprint = row.as_decision_task().fingerprint()
        if fingerprint in seen_fingerprints:
            raise DatasetError(f"linha {number}: estado e perguntas repetidos ({row.id})")
        seen_ids.add(row.id)
        seen_fingerprints.add(fingerprint)
        rows.append(row)
    return rows


def _parse_line(number: int, line: str) -> EvalDecision:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as exc:
        raise DatasetError(f"linha {number}: JSON inválido ({exc.msg})") from exc
    if not isinstance(payload, dict):
        raise DatasetError(f"linha {number}: a linha não é um objeto")
    try:
        return EvalDecision.model_validate(payload)
    except ValidationError as exc:
        raise DatasetError(f"linha {number}: {_format_errors(exc)}") from exc


def _format_errors(exc: ValidationError) -> str:
    parts: list[str] = []
    for error in exc.errors():
        location = ".".join(str(piece) for piece in error["loc"])
        message = str(error["msg"])
        parts.append(f"{location}: {message}" if location else message)
    return "; ".join(parts) or "schema inválido"


def _blank_state(state: str | dict[str, Any] | list[Any]) -> bool:
    if isinstance(state, str):
        return not state.strip()
    return len(state) == 0
