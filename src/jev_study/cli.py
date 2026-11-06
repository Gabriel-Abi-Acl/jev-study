"""Ponto de entrada do placeholder. O default ecoa o gabarito e não abre rede.

A saída JSON desse modo não é resultado do Jev nem do baseline.
"""

import argparse
import json
import sys

from jev_study.backends.fake import echo_labels
from jev_study.backends.jev import JevHttpBackend
from jev_study.backends.llm import LlmStructuredBackend
from jev_study.config import load_config
from jev_study.dataset import load_synthetic_tasks
from jev_study.errors import BackendError, ConfigError
from jev_study.runner import run_tasks


def main(argv: list[str] | None = None) -> int:
    """Roda o conjunto sintético. Sem chave, jev e llm saem com código 2."""
    parser = argparse.ArgumentParser(
        description="Roda o conjunto sintético de decisões e imprime o registro JSON."
    )
    parser.add_argument(
        "--backend",
        choices=("fake", "jev", "llm"),
        default="fake",
        help="fake ecoa o gabarito; jev e llm chamam a rede e exigem chave.",
    )
    args = parser.parse_args(argv)
    try:
        config = load_config()
        tasks = load_synthetic_tasks()
        backend, input_price, output_price = _backend(args.backend, config, tasks)
    except (ConfigError, BackendError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    records = run_tasks(
        backend,
        tasks,
        input_usd_per_mtok=input_price,
        output_usd_per_mtok=output_price,
    )
    json.dump(
        [record.model_dump(mode="json") for record in records],
        sys.stdout,
        ensure_ascii=False,
        indent=2,
    )
    sys.stdout.write("\n")
    if any(record.error for record in records):
        return 1
    return 0


def _backend(kind: str, config, tasks):
    if kind == "fake":
        # Sem tarifa: o eco não tem usage, e um zero aqui pareceria medição.
        return echo_labels(tasks), None, None
    if kind == "jev":
        if not config.typesafe_api_key:
            raise BackendError("TYPESAFE_API_KEY ou JEV_API_KEY ausente")
        backend = JevHttpBackend(
            api_key=config.typesafe_api_key,
            model=config.jev_model,
            base_url=config.typesafe_base_url,
        )
        return backend, config.jev_input_usd_per_mtok, config.jev_output_usd_per_mtok
    if not config.llm_api_key or not config.llm_model:
        raise BackendError("LLM_API_KEY e LLM_MODEL são obrigatórios para o backend llm")
    backend = LlmStructuredBackend(
        model=config.llm_model,
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
    )
    return backend, config.llm_input_usd_per_mtok, config.llm_output_usd_per_mtok
