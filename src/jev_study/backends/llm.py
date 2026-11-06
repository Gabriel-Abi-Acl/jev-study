"""Baseline de LLM com saída no mesmo schema do System One.

O pacote openai só entra na chamada ao vivo. Testes injetam um completer e
não instalam o extra `llm`. A contagem de tokens do provedor substitui a que
vier dentro do JSON, para o modelo não inventar a própria fatura.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from jev_study.errors import BackendError, SchemaError, redact
from jev_study.schema import SystemOneRequest, SystemOneResponse, dump_request, parse_response

DEFAULT_TIMEOUT_S = 60.0

_SYSTEM_PROMPT = (
    "You convert one decision request into the System One response JSON schema. "
    "Use only the state and the question instructions and criteria in the user message. "
    "Return JSON, with no prose. "
    "choice: select one criteria key and give a probability for every key. "
    "score: return the expected score on the rubric, a legend, and a probability per level. "
    "noul: return the probability of yes as a number from 0 to 1. "
    "box: return edges top, bottom, left, and right as fractions from 0 to 1. "
    "You are a structured-output baseline in a comparison study, not a System One model."
)


@dataclass(frozen=True)
class LlmCompletion:
    """Texto do modelo e a contagem que o provedor reportou, se houver."""

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    model: str | None = None


Completer = Callable[[list[dict[str, str]], dict[str, Any]], LlmCompletion]


class LlmStructuredBackend:
    """Pede JSON no schema System One. Sem completer, chama a API da OpenAI."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        completer: Completer | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        name: str = "llm",
    ) -> None:
        if not model.strip():
            raise BackendError("modelo do baseline vazio")
        self.name = name
        self.model = model
        self._api_key = api_key
        self._base_url = base_url
        self._completer = completer
        self._timeout_s = timeout_s

    def decide(self, request: SystemOneRequest) -> SystemOneResponse:
        messages = build_llm_messages(request)
        schema = response_schema()
        if self._completer is not None:
            completion = self._completer(messages, schema)
        else:
            completion = self._live_complete(messages, schema)
        return response_from_completion(completion, fallback_model=self.model)

    def _live_complete(
        self,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
    ) -> LlmCompletion:
        if not self._api_key or any(character.isspace() for character in self._api_key):
            raise BackendError("LLM_API_KEY ausente ou inválida")
        try:
            openai = _import_openai()
        except ImportError as exc:
            raise BackendError(
                "Pacote openai ausente. Instale o extra llm ou injete um completer."
            ) from exc
        client_kwargs: dict[str, Any] = {"api_key": self._api_key}
        if self._base_url:
            client_kwargs["base_url"] = self._base_url
        client = openai.OpenAI(**client_kwargs)
        try:
            # oneOf do discriminator não passa no strict mode da OpenAI.
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": "system_one_response",
                        "schema": schema,
                        "strict": False,
                    },
                },
                timeout=self._timeout_s,
            )
        except Exception as exc:
            detail = redact(f"{exc.__class__.__name__}: {exc}", self._api_key)[:500]
            # from None: erros do SDK às vezes ecoam o request, e com ele a chave.
            raise BackendError(f"falha na chamada do baseline LLM: {detail}") from None
        return completion_from_chat_response(response, fallback_model=self.model)


def build_llm_messages(request: SystemOneRequest) -> list[dict[str, str]]:
    """O user message é o mesmo JSON que o Jev receberia, sem dica de gabarito."""
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(dump_request(request), ensure_ascii=False),
        },
    ]


def response_schema() -> dict[str, Any]:
    """JSON Schema do pydantic. O modo strict fica desligado porque há oneOf."""
    return SystemOneResponse.model_json_schema()


def response_from_completion(
    completion: LlmCompletion,
    *,
    fallback_model: str,
) -> SystemOneResponse:
    """Monta a resposta tipada. Tokens do provedor ganham dos tokens escritos no JSON."""
    data = _load_object(completion.text)
    if _looks_like_answer_map(data):
        data = {"answers": data}
    if completion.model:
        data["model"] = completion.model
    elif "model" not in data:
        data["model"] = fallback_model
    if completion.input_tokens is not None and completion.output_tokens is not None:
        data["usage"] = {
            "input_tokens": completion.input_tokens,
            "output_tokens": completion.output_tokens,
        }
    return parse_response(data)


def completion_from_chat_response(response: object, *, fallback_model: str) -> LlmCompletion:
    """Lê o objeto do SDK sem importá-lo. prompt_tokens vira input_tokens."""
    choices = getattr(response, "choices", None)
    if not choices:
        raise BackendError("resposta do baseline sem choices")
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if not isinstance(content, str) or not content.strip():
        raise SchemaError("baseline devolveu conteúdo vazio")
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", None) if usage is not None else None
    output_tokens = getattr(usage, "completion_tokens", None) if usage is not None else None
    if not isinstance(input_tokens, int) or isinstance(input_tokens, bool):
        input_tokens = None
    if not isinstance(output_tokens, int) or isinstance(output_tokens, bool):
        output_tokens = None
    model = getattr(response, "model", None)
    if not isinstance(model, str) or not model.strip():
        model = fallback_model
    return LlmCompletion(
        text=content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=model,
    )


def _import_openai():
    """Import tardio. A suíte não precisa do pacote openai instalado."""
    import openai

    return openai


def _load_object(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SchemaError(f"JSON inválido: {exc}") from exc
    if not isinstance(data, dict):
        raise SchemaError("JSON do baseline não é objeto")
    return data


def _looks_like_answer_map(data: dict[str, Any]) -> bool:
    if "answers" in data or not data:
        return False
    return all(isinstance(value, dict) and "type" in value for value in data.values())
