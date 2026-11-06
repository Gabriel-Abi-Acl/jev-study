"""Estimativa de custo a partir de tokens e de uma tarifa passada pelo chamador.

Sem tarifa ou sem contagem, o custo fica ausente. Nada aqui é medição de fatura
nem leitura de página de marketing.
"""


def estimate_cost_usd(
    input_tokens: int | None,
    output_tokens: int | None,
    *,
    input_usd_per_mtok: float | None,
    output_usd_per_mtok: float | None,
) -> float | None:
    """USD da request. Tarifa de saída ausente com tokens de saída > 0 não estima."""
    if input_usd_per_mtok is None or input_tokens is None:
        return None
    _non_negative("tarifa de entrada", input_usd_per_mtok)
    _non_negative("input_tokens", input_tokens)
    cost = (input_tokens / 1_000_000) * input_usd_per_mtok
    if output_usd_per_mtok is None:
        if output_tokens not in (None, 0):
            return None
        if output_tokens is not None:
            _non_negative("output_tokens", output_tokens)
        return cost
    if output_tokens is None:
        return None
    _non_negative("tarifa de saída", output_usd_per_mtok)
    _non_negative("output_tokens", output_tokens)
    return cost + (output_tokens / 1_000_000) * output_usd_per_mtok


def _non_negative(label: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{label} negativo")
