"""Executa tarefas e grava a decisão crua. Não calcula concordância nem calibração.

O custo estimado é o da request inteira e se repete em cada pergunta da mesma
task. Somar as linhas conta a mesma chamada mais de uma vez.
"""

import time
from collections.abc import Callable, Sequence

from jev_study.backends.base import DecisionBackend
from jev_study.pricing import estimate_cost_usd
from jev_study.records import (
    RunRecord,
    decision_payload,
    declared_confidence,
    declared_probability,
)
from jev_study.schema import SystemOneRequest, SystemOneResponse
from jev_study.tasks import DecisionTask


def run_tasks(
    backend: DecisionBackend,
    tasks: Sequence[DecisionTask],
    *,
    input_usd_per_mtok: float | None = None,
    output_usd_per_mtok: float | None = None,
    clock: Callable[[], float] | None = None,
) -> list[RunRecord]:
    """Uma linha por pergunta. Tarefa pulada não chama o backend."""
    now = clock or time.perf_counter
    records: list[RunRecord] = []
    for task in tasks:
        if task.skip:
            records.extend(_skipped(task, backend, input_usd_per_mtok, output_usd_per_mtok))
            continue
        request = SystemOneRequest(
            model=backend.model,
            state=task.state,
            questions=task.questions,
        )
        started = now()
        try:
            response = backend.decide(request)
        except Exception as exc:
            elapsed = (now() - started) * 1000
            records.extend(
                _failed(
                    task,
                    backend,
                    elapsed,
                    str(exc),
                    input_usd_per_mtok,
                    output_usd_per_mtok,
                )
            )
            continue
        elapsed = (now() - started) * 1000
        records.extend(
            _answered(
                task,
                backend,
                response,
                elapsed,
                input_usd_per_mtok,
                output_usd_per_mtok,
            )
        )
    return records


def _skipped(
    task: DecisionTask,
    backend: DecisionBackend,
    input_usd_per_mtok: float | None,
    output_usd_per_mtok: float | None,
) -> list[RunRecord]:
    return [
        _row(
            task,
            backend,
            name,
            question.type,
            input_usd_per_mtok,
            output_usd_per_mtok,
            skipped=True,
            skip_reason=task.skip_reason,
        )
        for name, question in task.questions.items()
    ]


def _failed(
    task: DecisionTask,
    backend: DecisionBackend,
    latency_ms: float,
    error: str,
    input_usd_per_mtok: float | None,
    output_usd_per_mtok: float | None,
) -> list[RunRecord]:
    return [
        _row(
            task,
            backend,
            name,
            question.type,
            input_usd_per_mtok,
            output_usd_per_mtok,
            latency_ms=latency_ms,
            error=error,
        )
        for name, question in task.questions.items()
    ]


def _answered(
    task: DecisionTask,
    backend: DecisionBackend,
    response: SystemOneResponse,
    latency_ms: float,
    input_usd_per_mtok: float | None,
    output_usd_per_mtok: float | None,
) -> list[RunRecord]:
    usage = response.usage
    input_tokens = usage.input_tokens if usage is not None else None
    output_tokens = usage.output_tokens if usage is not None else None
    cost = estimate_cost_usd(
        input_tokens,
        output_tokens,
        input_usd_per_mtok=input_usd_per_mtok,
        output_usd_per_mtok=output_usd_per_mtok,
    )
    rows: list[RunRecord] = []
    for name, question in task.questions.items():
        answer = response.answers.get(name)
        if answer is None:
            rows.append(
                _row(
                    task,
                    backend,
                    name,
                    question.type,
                    input_usd_per_mtok,
                    output_usd_per_mtok,
                    model=response.model,
                    latency_ms=latency_ms,
                    estimated_cost_usd=cost,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    error="resposta ausente",
                )
            )
            continue
        error = None
        if answer.type != question.type:
            error = f"tipo da resposta ({answer.type}) difere da pergunta ({question.type})"
        rows.append(
            _row(
                task,
                backend,
                name,
                question.type,
                input_usd_per_mtok,
                output_usd_per_mtok,
                model=response.model,
                decision=decision_payload(answer),
                declared_confidence=declared_confidence(answer),
                declared_probability=declared_probability(answer),
                latency_ms=latency_ms,
                estimated_cost_usd=cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                error=error,
            )
        )
    for name, answer in response.answers.items():
        if name in task.questions:
            continue
        rows.append(
            _row(
                task,
                backend,
                name,
                answer.type,
                input_usd_per_mtok,
                output_usd_per_mtok,
                model=response.model,
                decision=decision_payload(answer),
                declared_confidence=declared_confidence(answer),
                declared_probability=declared_probability(answer),
                latency_ms=latency_ms,
                estimated_cost_usd=cost,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                error="resposta sem pergunta correspondente",
            )
        )
    return rows


def _row(
    task: DecisionTask,
    backend: DecisionBackend,
    question: str,
    question_type: str,
    input_usd_per_mtok: float | None,
    output_usd_per_mtok: float | None,
    **fields,
) -> RunRecord:
    return RunRecord(
        task_id=task.id,
        source=task.source,
        backend=backend.name,
        requested_model=backend.model,
        question=question,
        question_type=question_type,
        input_usd_per_mtok=input_usd_per_mtok,
        output_usd_per_mtok=output_usd_per_mtok,
        **fields,
    )
