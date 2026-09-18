from __future__ import annotations


def weighted_opportunity_score(
    demand: float,
    growth: float,
    stability: float,
    anomaly_safety: float,
    *,
    demand_weight: float = 0.35,
    growth_weight: float = 0.35,
    stability_weight: float = 0.15,
    anomaly_weight: float = 0.15,
) -> float:
    """Return a 0-100 configurable decision-support score."""
    weights = (demand_weight, growth_weight, stability_weight, anomaly_weight)
    if abs(sum(weights) - 1.0) > 1e-9:
        raise ValueError("Opportunity weights must sum to 1.0")

    signals = (demand, growth, stability, anomaly_safety)
    if any(value < 0.0 or value > 1.0 for value in signals):
        raise ValueError("All normalized signals must be between 0 and 1")

    raw = (
        demand * demand_weight
        + growth * growth_weight
        + stability * stability_weight
        + anomaly_safety * anomaly_weight
    )
    return round(raw * 100.0, 2)
