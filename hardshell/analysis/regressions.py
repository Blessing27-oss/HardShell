"""statsmodels OLS regressions for HardShell metrics."""
# hardshell/analysis/regressions.py
from __future__ import annotations

import pandas as pd
import statsmodels.formula.api as smf


def calculate_utility_tax(df: pd.DataFrame):
    """OLS: utility_score ~ condition (legacy 3-condition design).

    The coefficient for each non-baseline condition is the Utility Tax.
    """
    df = df.copy()
    df["condition"] = pd.Categorical(
        df["condition"],
        categories=sorted(df["condition"].unique()),
    )
    return smf.ols(formula="utility_score ~ condition", data=df).fit()


def factorial_regression(df: pd.DataFrame) -> tuple:
    """2×2 factorial OLS for the tool_defense experiment.

    Returns (utility_model, asr_model) — both fit on:
        DV ~ attack * tool_def

    where:
        attack   = inject_payload (0/1)
        tool_def = tool_defense   (0/1)

    The interaction term captures the incremental effect of the defense
    specifically under attack — i.e., the defense effectiveness.

    Interpretation guide
    --------------------
    utility_model:
        Intercept           → baseline utility (no attack, no defense)
        attack              → utility change due to attack alone
        tool_def            → utility tax from defense alone (false-positive cost)
        attack:tool_def     → interaction: does defense restore utility under attack?

    asr_model:
        Intercept           → baseline ASR (should be ~0 with no attack)
        attack              → raw ASR increase from attack
        tool_def            → ASR change from defense alone (should be ~0 or negative)
        attack:tool_def     → how much the defense reduces ASR (negative = effective)
    """
    df = df.copy()
    df["attack"]   = df["inject_payload"].astype(int)
    df["tool_def"] = df["tool_defense"].astype(int)

    # Drop rows where judge evaluation failed (value = -1)
    valid_utility = df[df["utility_score"] >= 0].copy()
    valid_asr     = df[df["target_execution_asr"] >= 0].copy()

    utility_model = smf.ols(
        "utility_score ~ attack * tool_def", data=valid_utility
    ).fit()

    asr_model = smf.ols(
        "target_execution_asr ~ attack * tool_def", data=valid_asr
    ).fit()

    return utility_model, asr_model


def format_factorial_results(utility_model, asr_model) -> str:
    """Format both factorial regression results as a readable text summary."""
    sep = "=" * 60
    lines = [
        sep,
        "2×2 Factorial Regression: tool_defense × attack",
        sep,
        "",
        "── Utility Score (task quality) ──",
        str(utility_model.summary()),
        "",
        "── Attack Success Rate (ASR) ──",
        str(asr_model.summary()),
        "",
        "── Effect Sizes ──",
    ]

    def _fmt_coef(model, name: str, label: str) -> str:
        try:
            coef = model.params[name]
            pval = model.pvalues[name]
            ci   = model.conf_int().loc[name]
            sig  = "***" if pval < 0.001 else ("**" if pval < 0.01 else ("*" if pval < 0.05 else "ns"))
            return f"  {label:40s}  β={coef:+.3f}  95%CI=[{ci[0]:+.3f},{ci[1]:+.3f}]  p={pval:.4f}  {sig}"
        except KeyError:
            return f"  {label:40s}  (not estimated)"

    for model, name in [(utility_model, "Utility"), (asr_model, "ASR")]:
        lines.append(f"\n  {name}:")
        lines.append(_fmt_coef(model, "attack",          "  Main effect: attack"))
        lines.append(_fmt_coef(model, "tool_def",        "  Main effect: tool_defense"))
        lines.append(_fmt_coef(model, "attack:tool_def", "  Interaction: attack × tool_defense"))

    lines += [
        "",
        "Note: interaction attack:tool_def < 0 in ASR model = defense is effective.",
        "      interaction attack:tool_def > 0 in Utility model = defense restores utility under attack.",
        sep,
    ]
    return "\n".join(lines)