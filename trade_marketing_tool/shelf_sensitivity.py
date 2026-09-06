"""
Sales sensitivity to shelf space / price: Pearson correlation + simple
linear regression.

The retail-trade-marketing counterpart to ``sensitivity.py``'s Beta/Alpha
regression, applied to a single explanatory variable — shelf space,
price, or any other trade lever — against outlet sales, per the
correlation/regression chapters of *Analytical Statistics with SPSS
Applications*.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import statsmodels.api as sm


@dataclass
class ShelfSensitivityResult:
    x_label: str
    y_label: str
    n: int
    r: float                # Pearson correlation coefficient
    r_squared: float
    slope: float             # b, in Y-units per unit of X
    intercept: float         # a
    p_value: float           # significance of the slope
    std_err: float           # standard error of the slope


def analyze_sensitivity(
    x: pd.Series,
    y: pd.Series,
    x_label: str = "Shelf Space",
    y_label: str = "Sales",
) -> ShelfSensitivityResult:
    """
    Fit ``y = a + b*x`` by OLS and report the Pearson correlation alongside
    it — e.g. shelf space (meters) vs. monthly sales (thousands), or price
    vs. unit volume for a price-elasticity read.
    """
    x = pd.Series(x).astype(float)
    y = pd.Series(y).astype(float)
    if len(x) != len(y):
        raise ValueError("x and y must have the same number of observations")
    if len(x) < 3:
        raise ValueError("need at least 3 observations to fit a regression")

    r = float(x.corr(y))

    X = sm.add_constant(x)
    model = sm.OLS(y.values, X.values).fit()
    intercept, slope = model.params

    return ShelfSensitivityResult(
        x_label=x_label,
        y_label=y_label,
        n=len(x),
        r=r,
        r_squared=float(model.rsquared),
        slope=float(slope),
        intercept=float(intercept),
        p_value=float(model.pvalues[1]),
        std_err=float(model.bse[1]),
    )


def predict(result: ShelfSensitivityResult, x_value: float) -> float:
    """Predict Y (e.g. sales) for a given X (e.g. shelf space) using the fitted line."""
    return result.intercept + result.slope * x_value


def format_report(result: ShelfSensitivityResult, alpha: float = 0.05) -> str:
    lines = []
    lines.append("=" * 56)
    lines.append(f"   {result.y_label.upper()} vs. {result.x_label.upper()} — REGRESSION")
    lines.append("=" * 56)
    lines.append(f"n = {result.n}")
    lines.append(f"Pearson r:  {result.r:.4f}   R-squared: {result.r_squared:.4f}")
    lines.append(
        f"Equation:   {result.y_label} = {result.intercept:.3f} + {result.slope:.3f} * {result.x_label}"
    )
    lines.append(f"Slope p-value: {result.p_value:.4f}   Std. error: {result.std_err:.4f}")
    verdict = "SIGNIFICANT" if result.p_value < alpha else "NOT significant"
    lines.append(f"Slope significant at alpha={alpha}: {verdict}")
    return "\n".join(lines)
