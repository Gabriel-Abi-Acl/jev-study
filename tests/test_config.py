"""Contrato: chave no ambiente, tarifa default de lista, segredo fora da mensagem."""

import pytest

from jev_study.config import (
    DEFAULT_JEV_MODEL,
    DEFAULT_TYPESAFE_BASE_URL,
    PUBLISHED_JEV_INPUT_USD_PER_MTOK,
    PUBLISHED_JEV_OUTPUT_USD_PER_MTOK,
    load_config,
)
from jev_study.errors import ConfigError


def test_published_list_price_is_the_default_and_is_overridable():
    config = load_config({})
    assert PUBLISHED_JEV_INPUT_USD_PER_MTOK == 0.042
    assert PUBLISHED_JEV_OUTPUT_USD_PER_MTOK == 0.0
    assert config.jev_input_usd_per_mtok == 0.042
    assert config.jev_output_usd_per_mtok == 0.0
    assert config.jev_model == DEFAULT_JEV_MODEL == "jev-1.14.0"
    assert config.typesafe_base_url == DEFAULT_TYPESAFE_BASE_URL
    assert config.typesafe_api_key is None
    assert config.llm_input_usd_per_mtok is None
    overridden = load_config({"JEV_INPUT_USD_PER_MTOK": "1.5", "JEV_MODEL": "jev-latest"})
    assert overridden.jev_input_usd_per_mtok == 1.5
    assert overridden.jev_model == "jev-latest"


def test_typesafe_key_wins_and_jev_key_is_the_fallback():
    both = load_config({"TYPESAFE_API_KEY": "primary", "JEV_API_KEY": "secondary"})
    fallback = load_config({"JEV_API_KEY": "secondary", "TYPESAFE_API_KEY": "  "})
    assert both.typesafe_api_key == "primary"
    assert fallback.typesafe_api_key == "secondary"


def test_invalid_secret_is_rejected_without_echoing_it():
    secret = "abc\ninjected"
    with pytest.raises(ConfigError) as exc:
        load_config({"TYPESAFE_API_KEY": secret})
    assert "abc" not in str(exc.value)
    assert "injected" not in str(exc.value)


def test_unused_malformed_fallback_key_does_not_block_the_primary():
    config = load_config({"TYPESAFE_API_KEY": "primary", "JEV_API_KEY": "bad key"})
    assert config.typesafe_api_key == "primary"


def test_bad_numbers_and_urls_fail():
    with pytest.raises(ConfigError):
        load_config({"JEV_INPUT_USD_PER_MTOK": "free"})
    with pytest.raises(ConfigError):
        load_config({"LLM_OUTPUT_USD_PER_MTOK": "-1"})
    with pytest.raises(ConfigError):
        load_config({"TYPESAFE_BASE_URL": "file:///tmp/jev"})
    with pytest.raises(ConfigError) as exc:
        load_config({"TYPESAFE_BASE_URL": "https://user:super-secret@api.typesafe.ai"})
    assert "super-secret" not in str(exc.value)


def test_full_endpoint_is_accepted_as_origin():
    config = load_config({"TYPESAFE_BASE_URL": "https://api.typesafe.ai/v1/systemone/"})
    assert config.typesafe_base_url == "https://api.typesafe.ai"


def test_llm_prices_stay_unset_until_provided():
    config = load_config({"LLM_INPUT_USD_PER_MTOK": "2", "LLM_BASE_URL": ""})
    assert config.llm_input_usd_per_mtok == 2
    assert config.llm_output_usd_per_mtok is None
    assert config.llm_base_url is None
    assert config.llm_model is None
