"""Cost estimate from token usage. Prices are per million tokens, first-party API list prices as documented on
2026-10-09; they are an estimate only and a fallback model's tokens are priced at the primary model's rate."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input: float
    output: float
    cache_read: float
    cache_write: float  # 5-minute cache write, 1.25x input


PRICES: dict[str, Price] = {
    "claude-opus-5-5": Price(4.00, 20.00, 0.20, 5.00),
    "claude-sonnet-5-5": Price(2.00, 10.00, 0.20, 2.50),
}


def estimate_cost_usd(
    model: str, input_tokens: int, output_tokens: int, cache_read: int, cache_write: int
) -> float | None:
    """None when the model has no known price."""
    p = PRICES.get(model)
    if p is None:
        return None
    total = (
        input_tokens * p.input
        + output_tokens * p.output
        + cache_read * p.cache_read
        + cache_write * p.cache_write
    )
    return round(total / 1_000_000, 6)
