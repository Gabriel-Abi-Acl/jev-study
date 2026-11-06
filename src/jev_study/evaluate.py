"""Harness do gabarito v1. O modo fake não abre rede e não é resultado de modelo.

Cada item passa pelo adapter e volta com decisão, gabarito, casamento,
confiança, probabilidade, latência, tokens e custo quando o backend os entrega.
"""

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from jev_study.backends.base import DecisionBackend
from jev_study.backends.fake import echo_labels
from jev_study.backends.jev import JevHttpBackend
from jev_study.backends.llm import LlmStructuredBackend
from jev_study.config import StudyConfig, load_config
from jev_study.dataset import DatasetError, load_eval_decisions
from jev_study.dataset.eval_set import EvalDecision
from jev_study.errors import BackendError, ConfigError
from jev_study.metrics import DEFAULT_BINS, EvalItem, EvaluationReport, build_report, score_record
from jev_study.runner import run_tasks


@dataclass(frozen=True)
class BackendRun:
    """Adapter e as tarifas da estimativa. Sem usage, o custo do item continua nulo."""

    backend: DecisionBackend
    input_usd_per_mtok: float | None = None
    output_usd_per_mtok: float | None = None


def evaluate_backends(
    runs: Sequence[BackendRun],
    decisions: Sequence[EvalDecision],
    *,
    bins: int = DEFAULT_BINS,
    clock: Callable[[], float] | None = None,
) -> EvaluationReport:
    """Roda as mesmas tarefas em cada adapter. O request não leva gabarito nem expectation."""
    tasks = [decision.as_decision_task() for decision in decisions]
    expectation = {decision.id: decision.expectation for decision in decisions}
    by_id = {task.id: task for task in tasks}
    items: list[EvalItem] = []
    for run in runs:
        records = run_tasks(
            run.backend,
            tasks,
            input_usd_per_mtok=run.input_usd_per_mtok,
            output_usd_per_mtok=run.output_usd_per_mtok,
            clock=clock,
        )
        for record in records:
            items.append(
                score_record(
                    record,
                    by_id[record.task_id],
                    expectation=expectation[record.task_id],
                )
            )
    return build_report(items, bins=bins)


def main(argv: list[str] | None = None) -> int:
    """Imprime itens e métricas do v1.

    Sem chave, jev, llm e all saem com código 2. Código 1 avisa que alguma
    linha tem error. Casamento falso é resultado da comparação, não falha.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Avalia o gabarito v1 nos adapters e imprime itens e métricas. "
            "O default fake ecoa o rótulo e não abre rede."
        )
    )
    parser.add_argument(
        "--backend",
        choices=("fake", "jev", "llm", "all"),
        default="fake",
        help="fake é offline; jev, llm e all chamam a rede e exigem chave.",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=DEFAULT_BINS,
        help="Faixas da curva de confiabilidade. Default 10.",
    )
    args = parser.parse_args(argv)
    if args.bins < 1:
        print("bins < 1", file=sys.stderr)
        return 2
    try:
        config = load_config()
        decisions = load_eval_decisions()
        runs = _runs(args.backend, config, decisions)
    except (ConfigError, BackendError, DatasetError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    report = evaluate_backends(runs, decisions, bins=args.bins)
    json.dump(report.model_dump(mode="json"), sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    if any(item.error for item in report.items):
        return 1
    return 0


def _runs(kind: str, config: StudyConfig, decisions: Sequence[EvalDecision]) -> list[BackendRun]:
    tasks = [decision.as_decision_task() for decision in decisions]
    if kind == "fake":
        # Sem tarifa: o eco não tem usage, e um preço aqui pareceria medição.
        return [BackendRun(echo_labels(tasks))]
    if kind == "all":
        # As duas chaves são exigidas antes de qualquer chamada, para não rodar só um lado.
        failures: list[str] = []
        runs: list[BackendRun] = []
        for name in ("jev", "llm"):
            try:
                runs.append(_live(name, config))
            except BackendError as exc:
                failures.append(str(exc))
        if failures:
            raise BackendError("; ".join(failures))
        return runs
    return [_live(kind, config)]


def _live(kind: str, config: StudyConfig) -> BackendRun:
    if kind == "jev":
        if not config.typesafe_api_key:
            raise BackendError("TYPESAFE_API_KEY ou JEV_API_KEY ausente")
        backend = JevHttpBackend(
            api_key=config.typesafe_api_key,
            model=config.jev_model,
            base_url=config.typesafe_base_url,
        )
        return BackendRun(
            backend,
            input_usd_per_mtok=config.jev_input_usd_per_mtok,
            output_usd_per_mtok=config.jev_output_usd_per_mtok,
        )
    if kind == "llm":
        if not config.llm_api_key or not config.llm_model:
            raise BackendError("LLM_API_KEY e LLM_MODEL são obrigatórios para o backend llm")
        backend = LlmStructuredBackend(
            model=config.llm_model,
            api_key=config.llm_api_key,
            base_url=config.llm_base_url,
        )
        return BackendRun(
            backend,
            input_usd_per_mtok=config.llm_input_usd_per_mtok,
            output_usd_per_mtok=config.llm_output_usd_per_mtok,
        )
    raise BackendError(f"backend desconhecido: {kind}")


if __name__ == "__main__":
    raise SystemExit(main())
