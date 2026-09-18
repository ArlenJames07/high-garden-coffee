import pytest

from high_garden.business_rules import weighted_opportunity_score


def test_opportunity_score_range_and_value():
    score = weighted_opportunity_score(1.0, 1.0, 1.0, 1.0)
    assert score == 100.0


def test_opportunity_weights_must_sum_to_one():
    with pytest.raises(ValueError):
        weighted_opportunity_score(
            0.5,
            0.5,
            0.5,
            0.5,
            demand_weight=0.5,
            growth_weight=0.5,
            stability_weight=0.5,
            anomaly_weight=0.5,
        )
