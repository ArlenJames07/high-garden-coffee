import pytest

from high_garden.metrics import wape


def test_wape_perfect_prediction():
    assert wape([100, 200], [100, 200]) == 0.0


def test_wape_known_value():
    assert wape([100, 100], [90, 120]) == pytest.approx(0.15)
