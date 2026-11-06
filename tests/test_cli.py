"""Contrato do comando: o default ecoa o gabarito e não abre rede nem importa SDK."""

import json
import sys

from jev_study.cli import main

STUDY_ENV = (
    "TYPESAFE_API_KEY",
    "JEV_API_KEY",
    "TYPESAFE_BASE_URL",
    "JEV_MODEL",
    "JEV_INPUT_USD_PER_MTOK",
    "JEV_OUTPUT_USD_PER_MTOK",
    "LLM_API_KEY",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_INPUT_USD_PER_MTOK",
    "LLM_OUTPUT_USD_PER_MTOK",
)


def _isolate(monkeypatch):
    for name in STUDY_ENV:
        monkeypatch.delenv(name, raising=False)


def test_fake_command_prints_records_without_network(monkeypatch, capsys):
    _isolate(monkeypatch)

    def boom(*args, **kwargs):
        raise AssertionError("rede")

    monkeypatch.setattr("urllib.request.OpenerDirector.open", boom)
    sys.modules.pop("openai", None)
    assert main([]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 15
    assert sum(row["skipped"] for row in payload) == 1
    assert {row["backend"] for row in payload} == {"fake"}
    assert all(row["estimated_cost_usd"] is None for row in payload)
    billing = next(row for row in payload if row["task_id"] == "route_duplicate_charge")
    assert billing["decision"]["choice"] == "billing"
    assert "openai" not in sys.modules


def test_live_backends_without_keys_exit_before_the_sdk(monkeypatch, capsys):
    _isolate(monkeypatch)
    sys.modules.pop("openai", None)
    assert main(["--backend", "jev"]) == 2
    assert main(["--backend", "llm"]) == 2
    error = capsys.readouterr().err
    assert "TYPESAFE_API_KEY" in error
    assert "LLM_API_KEY" in error
    assert "openai" not in sys.modules
