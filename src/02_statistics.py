"""
Step 2 - Statistical analysis: which factors are associated with default?

1. One variable at a time:
   - categorical variables: chi-square test + Cramer's V (strength, 0 to 1)
   - numeric variables: Mann-Whitney U test + rank-biserial correlation (-1 to 1)
   Because we run many tests, we also give Bonferroni-adjusted p-values.
2. All variables together: logistic regression (statsmodels) with odds ratios
   and 95% confidence intervals ("all other things being equal").

These are associations, not causes.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns
import statsmodels.api as sm

import common as C


def cramers_v(table):
    """Cramer's V = sqrt(chi2 / (n * (min(rows, cols) - 1)))."""
    chi2 = st.chi2_contingency(table)[0]
    n = table.to_numpy().sum()
    return np.sqrt(chi2 / (n * (min(table.shape) - 1)))


def univariate_tests(df):
    """Test each variable against default and rank them by effect size."""
    rows = []
    for col in C.CATEGORICAL_COLS:
        table = pd.crosstab(df[col], df["default"])
        chi2, p, _, _ = st.chi2_contingency(table)
        rows.append({"variable": col, "test": "chi-square", "statistic": chi2,
                     "effect_size": cramers_v(table), "effect_measure": "Cramer's V", "p_value": p})

    for col in C.NUMERIC_COLS:
        good = df.loc[df["default"] == 0, col]
        bad = df.loc[df["default"] == 1, col]
        u, p = st.mannwhitneyu(bad, good, alternative="two-sided")
        # rank-biserial r > 0 means defaulters tend to have HIGHER values
        r = 2 * u / (len(bad) * len(good)) - 1
        rows.append({"variable": col, "test": "Mann-Whitney U", "statistic": u,
                     "effect_size": r, "effect_measure": "rank-biserial r", "p_value": p})

    out = pd.DataFrame(rows)
    out["p_bonferroni"] = np.minimum(out["p_value"] * len(out), 1.0)
    out["significant_5pct"] = out["p_bonferroni"] < 0.05
    out = out.sort_values("effect_size", key=abs, ascending=False)
    return out.round({"statistic": 1, "effect_size": 3})


def design_matrix(df):
    """Dummy variables for categories (reference = most frequent category)
    and standardised numeric variables (1 unit = 1 standard deviation)."""
    parts = []
    for col in C.MODEL_CATEGORICAL:
        reference = df[col].value_counts().index[0]
        dummies = pd.get_dummies(df[col], prefix=col, prefix_sep=" = ", dtype=float)
        parts.append(dummies.drop(columns=f"{col} = {reference}"))
    for col in C.MODEL_NUMERIC:
        parts.append(((df[col] - df[col].mean()) / df[col].std()).rename(f"{col} (+1 SD)"))
    return pd.concat(parts, axis=1)


def logistic_regression(df):
    """Fit the logit model and return odds ratios with 95% confidence intervals."""
    X = sm.add_constant(design_matrix(df))
    result = sm.Logit(df["default"], X).fit(disp=0)
    ci = result.conf_int()
    out = pd.DataFrame({
        "term": result.params.index,
        "odds_ratio": np.exp(result.params),
        "ci_low": np.exp(ci[0]),
        "ci_high": np.exp(ci[1]),
        "p_value": result.pvalues,
    }).query("term != 'const'")
    out["effect"] = np.where(out["odds_ratio"] > 1, "higher risk", "lower risk")
    print(f"McFadden pseudo R2 = {result.prsquared:.3f}, n = {int(result.nobs)}")
    return out.sort_values("p_value").round(3)


def plot_default_rate(df, col, fname):
    """Bar chart of the default rate for each category of one variable."""
    rates = df.groupby(col)["default"].agg(["mean", "size"]).sort_values("mean")
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(rates) + 1.5))
    ax.barh(rates.index, 100 * rates["mean"], color=C.BLUE)
    ax.axvline(100 * df["default"].mean(), color=C.RED, ls="--", label="average (30%)")
    for i, (rate, n) in enumerate(zip(rates["mean"], rates["size"])):
        ax.text(100 * rate + 0.5, i, f"{100 * rate:.0f}% (n={n})", va="center", fontsize=8)
    ax.set_xlabel("Default rate (%)")
    ax.set_title(f"Default rate by {col}")
    ax.legend(loc="lower right")
    C.save_fig(fig, fname)


def plot_numeric(df):
    """Box plots of the three real numeric variables for good vs bad credits."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, col in zip(axes, ["duration_months", "credit_amount", "age"]):
        sns.boxplot(data=df, x="default", y=col, hue="default", palette=[C.BLUE, C.RED],
                    legend=False, ax=ax)
        ax.set_xticks([0, 1], ["repaid (0)", "default (1)"])
        ax.set_xlabel("")
        ax.set_title(col)
    C.save_fig(fig, "stats_numeric_boxplots.png")


def plot_odds_ratios(odds):
    """Forest plot of the 15 most significant odds ratios."""
    top = odds.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = np.where(top["odds_ratio"] > 1, C.RED, C.GREEN)
    ax.errorbar(top["odds_ratio"], range(len(top)),
                xerr=[top["odds_ratio"] - top["ci_low"], top["ci_high"] - top["odds_ratio"]],
                fmt="none", ecolor=C.GREY, capsize=3)
    ax.scatter(top["odds_ratio"], range(len(top)), color=colors, zorder=3)
    ax.axvline(1, color="black", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_yticks(range(len(top)), top["term"], fontsize=8)
    ax.set_xlabel("Odds ratio (log scale, 95% CI). Right of 1 = higher risk")
    ax.set_title("Adjusted odds ratios - 15 most significant terms")
    C.save_fig(fig, "stats_odds_ratios.png")


def main():
    print("STEP 2 - Statistical analysis")
    df = C.load_data()

    tests = univariate_tests(df)
    C.save_table(tests, "stats_univariate_tests.csv")
    print("\nUnivariate tests (sorted by effect size):")
    print(tests[["variable", "effect_measure", "effect_size", "p_bonferroni"]].to_string(index=False))

    odds = logistic_regression(df)
    C.save_table(odds, "stats_odds_ratios.csv")
    print("\nLogistic regression - 10 most significant terms:")
    print(odds.head(10).to_string(index=False))

    for col in ["checking_status", "credit_history", "savings", "purpose"]:
        plot_default_rate(df, col, f"stats_default_by_{col}.png")
    plot_numeric(df)
    plot_odds_ratios(odds)


if __name__ == "__main__":
    main()
