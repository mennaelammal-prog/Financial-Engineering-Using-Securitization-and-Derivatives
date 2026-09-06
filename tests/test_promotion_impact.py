import pandas as pd
import pytest

from trade_marketing_tool.promotion_impact import (
    compare_before_after,
    compare_campaign_vs_control,
    compare_multiple_campaigns,
    promotion_cost_efficiency,
)


def test_paired_ttest_detects_clear_uplift():
    before = pd.Series([10, 12, 8, 15, 9, 11, 13, 10])
    after = before + 5  # deterministic uplift
    result = compare_before_after(before, after)
    assert result.uplift == pytest.approx(5.0)
    assert result.significant is True
    assert result.p_value < 0.05


def test_paired_ttest_no_uplift_is_not_significant():
    before = pd.Series([10, 12, 8, 15, 9, 11, 13, 10])
    after = pd.Series([10, 12, 8, 15, 9, 11, 13, 10])
    result = compare_before_after(before, after)
    assert result.uplift == pytest.approx(0.0)
    assert result.significant is False


def test_paired_requires_equal_length():
    with pytest.raises(ValueError):
        compare_before_after(pd.Series([1, 2, 3]), pd.Series([1, 2]))


def test_campaign_vs_control_independent_test():
    control = pd.Series([10, 11, 9, 10, 12, 11])
    treatment = pd.Series([20, 21, 19, 22, 20, 21])
    result = compare_campaign_vs_control(treatment, control)
    assert result.uplift > 0
    assert result.significant is True


def test_anova_identifies_best_group():
    groups = {
        "10% off": pd.Series([10, 12, 11, 9]),
        "BOGO": pd.Series([20, 22, 19, 21]),
        "End-cap": pd.Series([14, 13, 15, 14]),
    }
    result = compare_multiple_campaigns(groups)
    assert result.best_group == "BOGO"
    assert result.significant is True


def test_anova_requires_at_least_two_groups():
    with pytest.raises(ValueError):
        compare_multiple_campaigns({"only": pd.Series([1, 2, 3])})


def test_promotion_cost_efficiency_roi_above_one_when_profitable():
    roi = promotion_cost_efficiency(
        uplift_per_outlet=5.0, n_outlets=10, unit_margin=2.0, campaign_cost=50.0
    )
    assert roi == pytest.approx(2.0)


def test_promotion_cost_efficiency_rejects_nonpositive_cost():
    with pytest.raises(ValueError):
        promotion_cost_efficiency(uplift_per_outlet=5.0, n_outlets=10, unit_margin=2.0, campaign_cost=0)
