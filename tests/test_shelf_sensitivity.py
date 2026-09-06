import pandas as pd
import pytest

from trade_marketing_tool.shelf_sensitivity import analyze_sensitivity, predict


def test_perfect_linear_relationship_recovers_known_slope():
    x = pd.Series([1, 2, 3, 4, 5])
    y = 3 * x + 2  # y = 2 + 3x exactly
    result = analyze_sensitivity(x, y)
    assert result.slope == pytest.approx(3.0, abs=1e-6)
    assert result.intercept == pytest.approx(2.0, abs=1e-6)
    assert result.r == pytest.approx(1.0, abs=1e-6)
    assert result.r_squared == pytest.approx(1.0, abs=1e-6)


def test_negative_relationship_gives_negative_r():
    x = pd.Series([1, 2, 3, 4, 5])
    y = pd.Series([10, 8, 6, 4, 2])
    result = analyze_sensitivity(x, y)
    assert result.r < 0
    assert result.slope < 0


def test_predict_uses_fitted_line():
    x = pd.Series([1, 2, 3, 4, 5])
    y = 3 * x + 2
    result = analyze_sensitivity(x, y)
    assert predict(result, 10) == pytest.approx(32.0, abs=1e-6)


def test_requires_at_least_three_observations():
    with pytest.raises(ValueError):
        analyze_sensitivity(pd.Series([1, 2]), pd.Series([1, 2]))


def test_requires_matching_lengths():
    with pytest.raises(ValueError):
        analyze_sensitivity(pd.Series([1, 2, 3]), pd.Series([1, 2]))
