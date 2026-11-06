"""Backend falso para a CI. O eco devolve o gabarito, não uma previsão.

Usage fica vazio de propósito: token inventado viraria custo inventado.
"""

from collections.abc import Callable, Sequence

from jev_study.errors import BackendError
from jev_study.schema import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreAnswer,
    ScoreQuestion,
    SystemOneRequest,
    SystemOneResponse,
)
from jev_study.tasks import DecisionTask, ExpectedLabel, request_fingerprint

Responder = Callable[[SystemOneRequest], SystemOneResponse]


class FakeBackend:
    """Grava cada request e devolve o que o responder programar."""

    def __init__(
        self,
        responder: Responder,
        *,
        name: str = "fake",
        model: str = "fake-1",
    ) -> None:
        self.name = name
        self.model = model
        self.calls: list[SystemOneRequest] = []
        self._responder = responder

    def decide(self, request: SystemOneRequest) -> SystemOneResponse:
        self.calls.append(request)
        return self._responder(request)


def echo_labels(
    tasks: Sequence[DecisionTask],
    *,
    name: str = "fake",
    model: str = "fake-label-echo",
) -> FakeBackend:
    """Ecoa o gabarito com confiança 1. Não serve como resultado de modelo."""
    index: dict[str, DecisionTask] = {}
    for task in tasks:
        if task.skip:
            continue
        key = task.fingerprint()
        if key in index:
            raise ValueError(f"fingerprint repetido: {task.id}")
        index[key] = task

    def respond(request: SystemOneRequest) -> SystemOneResponse:
        task = index.get(request_fingerprint(request.state, request.questions))
        if task is None:
            raise BackendError("request sem gabarito sintético")
        answers = {
            question_name: _echo_answer(question, task.expected[question_name])
            for question_name, question in task.questions.items()
        }
        return SystemOneResponse(model=model, answers=answers, usage=None)

    return FakeBackend(respond, name=name, model=model)


def _echo_answer(question: Question, label: ExpectedLabel):
    if isinstance(question, ChoiceQuestion):
        if label.choice is None:
            raise BackendError("gabarito choice ausente")
        probabilities = {name: 1.0 if name == label.choice else 0.0 for name in question.criteria}
        return ChoiceAnswer(
            type="choice",
            choice=label.choice,
            confidence=1.0,
            probabilities=probabilities,
        )
    if isinstance(question, ScoreQuestion):
        if label.score_level is None:
            raise BackendError("gabarito score ausente")
        legend = {str(index): text for index, text in enumerate(question.criteria)}
        probabilities = {key: 1.0 if int(key) == label.score_level else 0.0 for key in legend}
        return ScoreAnswer(
            type="score",
            score=float(label.score_level),
            confidence=1.0,
            legend=legend,
            probabilities=probabilities,
        )
    if isinstance(question, NoulQuestion):
        if label.noul is None:
            raise BackendError("gabarito noul ausente")
        return NoulAnswer(type="noul", noul=1.0 if label.noul else 0.0)
    raise BackendError("eco de gabarito não inventa coordenada de box")
