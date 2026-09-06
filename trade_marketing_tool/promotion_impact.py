"""
Trade-promotion impact testing: paired/independent t-tests and one-way ANOVA.

This is the applied-statistics counterpart described in *Analytical
Statistics with SPSS Applications* (chapters on mean-difference tests):
quantifying whether a trade-marketing campaign or discount moved sales by
more than chance, and — with more than two campaigns — which one performed
best, before spending is justified against its cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from scipy import stats


@dataclass
class TTestResult:
    label_before: str
    label_after: str
    n_before: int
    n_after: int
    mean_before: float
    mean_after: float
    uplift: float          # mean_after - mean_before
    uplift_pct: float       # uplift / mean_before
    t_statistic: float
    p_value: float
    alpha: float
    significant: bool       # p_value < alpha


def compare_before_after(
    before: pd.Series, after: pd.Series, alpha: float = 0.05
) -> TTestResult:
    """
    Paired t-test: the same outlets/products measured before and after a
    promotion (e.g. weekly sales at 10 stores, pre- vs. post-campaign).
    ``before`` and ``after`` must be the same length and outlet-aligned.
    """
    before = pd.Series(before).astype(float)
    after = pd.Series(after).astype(float)
    if len(before) != len(after):
        raise ValueError("before and after must have the same number of observations")
    if len(before) < 2:
        raise ValueError("need at least 2 paired observations")

    t_stat, p_value = stats.ttest_rel(after, before)
    mean_before = float(before.mean())
    mean_after = float(after.mean())
    uplift = mean_after - mean_before

    return TTestResult(
        label_before="Before",
        label_after="After",
        n_before=len(before),
        n_after=len(after),
        mean_before=mean_before,
        mean_after=mean_after,
        uplift=uplift,
        uplift_pct=(uplift / mean_before) if mean_before else float("nan"),
        t_statistic=float(t_stat),
        p_value=float(p_value),
        alpha=alpha,
        significant=bool(p_value < alpha),
    )


def compare_campaign_vs_control(
    treatment: pd.Series,
    control: pd.Series,
    alpha: float = 0.05,
    equal_var: bool = False,
) -> TTestResult:
    """
    Independent-samples t-test (Welch's by default): outlets that ran a
    promotion vs. a separate control group of outlets that did not.
    """
    treatment = pd.Series(treatment).astype(float).dropna()
    control = pd.Series(control).astype(float).dropna()
    if len(treatment) < 2 or len(control) < 2:
        raise ValueError("need at least 2 observations in each group")

    t_stat, p_value = stats.ttest_ind(treatment, control, equal_var=equal_var)
    mean_control = float(control.mean())
    mean_treatment = float(treatment.mean())
    uplift = mean_treatment - mean_control

    return TTestResult(
        label_before="Control",
        label_after="Campaign",
        n_before=len(control),
        n_after=len(treatment),
        mean_before=mean_control,
        mean_after=mean_treatment,
        uplift=uplift,
        uplift_pct=(uplift / mean_control) if mean_control else float("nan"),
        t_statistic=float(t_stat),
        p_value=float(p_value),
        alpha=alpha,
        significant=bool(p_value < alpha),
    )


@dataclass
class AnovaResult:
    group_stats: pd.DataFrame   # Group, N, Mean, Std
    f_statistic: float
    p_value: float
    alpha: float
    significant: bool
    best_group: str


def compare_multiple_campaigns(
    groups: dict[str, pd.Series], alpha: float = 0.05
) -> AnovaResult:
    """
    One-way ANOVA across three or more campaign/offer variants (e.g. "10%
    off", "Buy-One-Get-One", "End-cap display") to test whether at least one
    variant's mean sales differs from the others.
    """
    if len(groups) < 2:
        raise ValueError("need at least 2 groups to compare")

    clean = {name: pd.Series(vals).astype(float).dropna() for name, vals in groups.items()}
    for name, vals in clean.items():
        if len(vals) < 2:
            raise ValueError(f"group '{name}' needs at least 2 observations")

    f_stat, p_value = stats.f_oneway(*clean.values())

    stats_rows = [
        {"Group": name, "N": len(vals), "Mean": float(vals.mean()), "Std": float(vals.std())}
        for name, vals in clean.items()
    ]
    group_stats = pd.DataFrame(stats_rows)
    best_group = group_stats.loc[group_stats["Mean"].idxmax(), "Group"]

    return AnovaResult(
        group_stats=group_stats,
        f_statistic=float(f_stat),
        p_value=float(p_value),
        alpha=alpha,
        significant=bool(p_value < alpha),
        best_group=str(best_group),
    )


def promotion_cost_efficiency(
    uplift_per_outlet: float,
    n_outlets: int,
    unit_margin: float,
    campaign_cost: float,
) -> float:
    """
    Incremental margin generated per currency unit spent on the campaign:
    ``(uplift_per_outlet * n_outlets * unit_margin) / campaign_cost``.
    A value > 1 means the campaign paid for itself in margin terms alone.
    """
    if campaign_cost <= 0:
        raise ValueError("campaign_cost must be > 0")
    incremental_margin = uplift_per_outlet * n_outlets * unit_margin
    return incremental_margin / campaign_cost


def format_ttest_report(result: TTestResult) -> str:
    lines = []
    lines.append("=" * 56)
    lines.append("        PROMOTION IMPACT — T-TEST")
    lines.append("=" * 56)
    lines.append(f"{result.label_before} (n={result.n_before}): mean = {result.mean_before:.2f}")
    lines.append(f"{result.label_after} (n={result.n_after}): mean = {result.mean_after:.2f}")
    lines.append(f"Uplift: {result.uplift:+.2f} ({result.uplift_pct:+.2%})")
    lines.append(f"t-statistic: {result.t_statistic:.4f}   p-value: {result.p_value:.4f}")
    verdict = "SIGNIFICANT" if result.significant else "NOT significant"
    lines.append(f"Result at alpha={result.alpha}: {verdict}")
    return "\n".join(lines)


def format_anova_report(result: AnovaResult) -> str:
    lines = []
    lines.append("=" * 56)
    lines.append("     PROMOTION COMPARISON — ONE-WAY ANOVA")
    lines.append("=" * 56)
    lines.append(
        result.group_stats.to_string(
            index=False,
            formatters={"Mean": "{:.2f}".format, "Std": "{:.2f}".format},
        )
    )
    lines.append("")
    lines.append(f"F-statistic: {result.f_statistic:.4f}   p-value: {result.p_value:.4f}")
    verdict = "SIGNIFICANT" if result.significant else "NOT significant"
    lines.append(f"Result at alpha={result.alpha}: {verdict}")
    if result.significant:
        lines.append(f"Best-performing group (highest mean): {result.best_group}")
    return "\n".join(lines)
