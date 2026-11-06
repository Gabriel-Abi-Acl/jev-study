"""Contrato do baseline: mesmo schema, SDK só na chamada ao vivo, tokens do provedor ganham."""

import json
import sys
from types import SimpleNamespace

import pytest

from jev_study.backends.llm import (
    LlmCompletion,
    LlmStructuredBackend,
    completion_from_chat_response,
    response_from_completion,
)
from jev_study.errors import BackendError, SchemaError
from jev_study.schema import NoulQuestion, SystemOneRequest
from tests.payloads import NOUL_RESPONSE


def _request() -> SystemOneRequest:
    return SystemOneRequest(
        model="gpt-test",
        state="Quero o estorno.",
        questions={"billing": NoulQuestion(instructions="Pede estorno?")},
    )


def test_package_import_does_not_load_vendor_sdks():
    sys.modules.pop("openai", None)
    sys.modules.pop("typesafe_sdk", None)
    import importlib

    import jev_study.backends.llm as llm_mod

    importlib.reload(llm_mod)
    assert "openai" not in sys.modules
    assert "typesafe_sdk" not in sys.modules


def test_completer_sees_the_same_request_and_provider_usage_wins():
    seen = {}

    def completer(messages, schema):
        seen["messages"] = messages
        seen["schema"] = schema
        payload = json.loads(json.dumps(NOUL_RESPONSE))
        payload["usage"] = {"input_tokens": 999, "output_tokens": 999}
        payload["model"] = "hallucinated-model"
        return LlmCompletion(
            text=json.dumps(payload),
            input_tokens=10,
            output_tokens=2,
            model="gpt-test-2026",
        )

    response = LlmStructuredBackend(model="gpt-test", completer=completer).decide(_request())
    user = json.loads(seen["messages"][1]["content"])
    assert user["state"] == "Quero o estorno."
    assert user["questions"]["billing"]["type"] == "noul"
    assert "Quero o estorno." not in seen["messages"][0]["content"]
    assert "answers" in seen["schema"]["properties"]
    assert "not a System One model" in seen["messages"][0]["content"]
    assert response.model == "gpt-test-2026"
    assert response.usage is not None
    assert response.usage.input_tokens == 10
    assert response.usage.output_tokens == 2
    assert response.answers["billing"].noul == 0.98


def test_answer_map_and_json_usage_when_the_provider_omits_tokens():
    completion = LlmCompletion(
        text=json.dumps({"billing": {"type": "noul", "noul": 0.25}}),
    )
    response = response_from_completion(completion, fallback_model="gpt-test")
    assert response.model == "gpt-test"
    assert response.usage is None
    assert response.answers["billing"].noul == 0.25

    with_usage = response_from_completion(
        LlmCompletion(text=json.dumps(NOUL_RESPONSE)),
        fallback_model="gpt-test",
    )
    assert with_usage.usage is not None
    assert with_usage.usage.input_tokens == 120
    assert with_usage.model == "jev-latest"


def test_invalid_json_is_a_schema_error():
    backend = LlmStructuredBackend(
        model="gpt-test",
        completer=lambda messages, schema: LlmCompletion(text="not-json"),
    )
    with pytest.raises(SchemaError):
        backend.decide(_request())


def test_live_path_imports_only_when_called_and_requires_a_key(monkeypatch):
    import jev_study.backends.llm as llm_mod

    def boom():
        raise ImportError("absent")

    monkeypatch.setattr(llm_mod, "_import_openai", boom)
    missing = LlmStructuredBackend(model="gpt-test", api_key=None)
    with pytest.raises(BackendError, match="LLM_API_KEY"):
        missing.decide(_request())
    live = LlmStructuredBackend(model="gpt-test", api_key="test-key")
    with pytest.raises(BackendError, match="openai"):
        live.decide(_request())


def test_chat_response_maps_provider_token_names():
    response = SimpleNamespace(
        model="gpt-test-2026",
        choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=4),
    )
    completion = completion_from_chat_response(response, fallback_model="fallback")
    assert completion.text == '{"ok": true}'
    assert completion.input_tokens == 11
    assert completion.output_tokens == 4
    assert completion.model == "gpt-test-2026"
    empty = SimpleNamespace(model="", choices=[], usage=None)
    with pytest.raises(BackendError):
        completion_from_chat_response(empty, fallback_model="fallback")
