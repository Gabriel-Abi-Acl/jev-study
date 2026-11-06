"""Contrato: custo só existe com tarifa e contagem. Saída desconhecida não vira zero."""

import pytest

from jev_study.pricing import estimate_cost_usd


def test_input_only_when_output_tokens_are_absent():
    assert estimate_cost_usd(
        1_000_000,
        None,
        input_usd_per_mtok=0.042,
        output_usd_per_mtok=None,
    ) == pytest.approx(0.042)


def test_nonzero_output_without_output_tariff_is_unavailable():
    assert (
        estimate_cost_usd(
            1_000_000,
            10,
            input_usd_per_mtok=0.042,
            output_usd_per_mtok=None,
        )
        is None
    )


def test_published_output_tariff_of_zero_does_not_add_cost():
    assert estimate_cost_usd(
        2_000_000,
        500_000,
        input_usd_per_mtok=0.042,
        output_usd_per_mtok=0.0,
    ) == pytest.approx(0.084)


def test_both_tariffs_add():
    assert estimate_cost_usd(
        2_000_000,
        1_000_000,
        input_usd_per_mtok=2.0,
        output_usd_per_mtok=3.0,
    ) == pytest.approx(7.0)


def test_missing_input_price_or_negative_values():
    assert estimate_cost_usd(10, 10, input_usd_per_mtok=None, output_usd_per_mtok=0) is None
    with pytest.raises(ValueError):
        estimate_cost_usd(10, 0, input_usd_per_mtok=-1, output_usd_per_mtok=0)
