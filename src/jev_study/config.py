"""Configuração por ambiente. O processo não lê arquivo .env sozinho.

A tarifa default é a de lista publicada para o Jev 1.14. Ela só entra na conta
de custo quando o backend devolve usage; não é resultado do estudo.
"""

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from urllib.parse import urlparse

from jev_study.errors import ConfigError

# https://docs.typesafe.ai/models — Jev 1.14, USD por milhão de tokens de entrada.
# A mesma página diz que o token de saída não é cobrado.
PUBLISHED_JEV_INPUT_USD_PER_MTOK = 0.042
PUBLISHED_JEV_OUTPUT_USD_PER_MTOK = 0.0
DEFAULT_JEV_MODEL = "jev-1.14.0"
DEFAULT_TYPESAFE_BASE_URL = "https://api.typesafe.ai"


@dataclass(frozen=True)
class StudyConfig:
    typesafe_api_key: str | None
    typesafe_base_url: str
    jev_model: str
    jev_input_usd_per_mtok: float
    jev_output_usd_per_mtok: float
    llm_api_key: str | None
    llm_base_url: str | None
    llm_model: str | None
    llm_input_usd_per_mtok: float | None
    llm_output_usd_per_mtok: float | None


def load_config(environ: Mapping[str, str] | None = None) -> StudyConfig:
    """Lê o ambiente explícito. Dict vazio não herda a chave da máquina."""
    env = os.environ if environ is None else environ
    typesafe_key = _optional_secret(env, "TYPESAFE_API_KEY")
    if typesafe_key is None:
        typesafe_key = _optional_secret(env, "JEV_API_KEY")
    base_url = _text(env, "TYPESAFE_BASE_URL", DEFAULT_TYPESAFE_BASE_URL)
    model = _text(env, "JEV_MODEL", DEFAULT_JEV_MODEL)
    return StudyConfig(
        typesafe_api_key=typesafe_key,
        typesafe_base_url=validate_http_base_url(base_url),
        jev_model=model,
        jev_input_usd_per_mtok=_float_or(
            env, "JEV_INPUT_USD_PER_MTOK", PUBLISHED_JEV_INPUT_USD_PER_MTOK
        ),
        jev_output_usd_per_mtok=_float_or(
            env, "JEV_OUTPUT_USD_PER_MTOK", PUBLISHED_JEV_OUTPUT_USD_PER_MTOK
        ),
        llm_api_key=_optional_secret(env, "LLM_API_KEY"),
        llm_base_url=_optional_base_url(env.get("LLM_BASE_URL")),
        llm_model=_blank_to_none(env.get("LLM_MODEL")),
        llm_input_usd_per_mtok=_optional_float(env, "LLM_INPUT_USD_PER_MTOK"),
        llm_output_usd_per_mtok=_optional_float(env, "LLM_OUTPUT_USD_PER_MTOK"),
    )


def validate_http_base_url(base_url: str) -> str:
    """Origem http(s), sem usuário nem senha. Aceita a origem com ou sem /v1/systemone."""
    stripped = base_url.strip().rstrip("/")
    suffix = "/v1/systemone"
    if stripped.endswith(suffix):
        stripped = stripped[: -len(suffix)].rstrip("/")
    parsed = urlparse(stripped)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ConfigError("URL base deve ser http(s) com host")
    if parsed.username or parsed.password:
        raise ConfigError("URL base não deve carregar credencial")
    return stripped


def _optional_secret(env: Mapping[str, str], name: str) -> str | None:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return None
    if any(character.isspace() for character in raw):
        raise ConfigError(f"{name} inválida")
    return raw.strip()


def _optional_base_url(raw: str | None) -> str | None:
    if raw is None or raw.strip() == "":
        return None
    return validate_http_base_url(raw)


def _text(env: Mapping[str, str], name: str, default: str) -> str:
    value = _blank_to_none(env.get(name))
    return default if value is None else value


def _blank_to_none(raw: str | None) -> str | None:
    if raw is None:
        return None
    stripped = raw.strip()
    return stripped or None


def _float_or(env: Mapping[str, str], name: str, default: float) -> float:
    raw = _blank_to_none(env.get(name))
    if raw is None:
        return default
    return _parse_non_negative(name, raw)


def _optional_float(env: Mapping[str, str], name: str) -> float | None:
    raw = _blank_to_none(env.get(name))
    if raw is None:
        return None
    return _parse_non_negative(name, raw)


def _parse_non_negative(name: str, raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} não é número") from exc
    if not math.isfinite(value) or value < 0:
        raise ConfigError(f"{name} inválida")
    return value
