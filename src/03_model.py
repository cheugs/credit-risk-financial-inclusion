"""
Step 3 - Interpretable baseline model, decision threshold and risk segmentation.

1. Logistic regression without sex, age or foreign-worker status.
2. Class imbalance: we compare a normal model with a "balanced" one.
   Both rank clients the same way, but only the normal model gives
   realistic probabilities (calibration).
3. Decision rule: the cost matrix (5 vs 1) gives a threshold of 1/6.
   We compare it with simple rules (approve everyone, reject everyone).
4. Approval vs cost: what does each threshold mean for access to credit?
5. Risk segmentation: 5 risk bands with 95% confidence intervals.
"""
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from statsmodels.stats.proportion import proportion_confint

import common as C


def compare_imbalance_methods(X_train, y_train):
    """Normal vs class_weight='balanced' model, using 5-fold CV on the training set."""
    rows, scores = [], {}
    for name, weight in [("normal", None), ("balanced", "balanced")]:
        p = C.cv_scores(C.make_model(class_weight=weight), X_train, y_train)
        scores[name] = p
        rows.append({
            "model": name,
            "roc_auc": roc_auc_score(y_train, p),
            "mean_predicted_pd": p.mean(),
            "observed_default_rate": y_train.mean(),
            "brier_score": brier_score_loss(y_train, p),
        })
    return pd.DataFrame(rows).round(3), scores


def plot_calibration(y, scores):
    """Reliability diagram: predicted probability vs observed default rate (10 bins)."""
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for (name, p), color in zip(scores.items(), [C.BLUE, C.RED]):
        bins = pd.qcut(p, 10, duplicates="drop")
        cal = pd.DataFrame({"p": p, "y": y}).groupby(bins, observed=True).mean()
        ax.plot(cal["p"], cal["y"], "o-", color=color, label=f"{name} model")
    ax.set_xlabel("Predicted probability of default")
    ax.set_ylabel("Observed default rate")
    ax.set_title("Calibration (5-fold CV, training set)")
    ax.legend()
    C.save_fig(fig, "model_calibration.png")


def threshold_table(y, p):
    """Approval rate and cost for a range of thresholds."""
    rows = [C.decision_summary(y, p, t) for t in np.arange(0.05, 0.96, 0.01)]
    return pd.DataFrame(rows)


def plot_threshold(curve):
    """Expected cost and approval rate as a function of the threshold."""
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(curve["threshold"], curve["expected_cost"], color=C.RED, lw=2, label="expected cost")
    ax1.set_xlabel("Threshold: reject if P(default) >= threshold")
    ax1.set_ylabel("Expected cost per client", color=C.RED)
    ax1.axvline(C.THRESHOLD, color=C.RED, ls="--", label=f"cost threshold 1/6 = {C.THRESHOLD:.2f}")
    ax1.axvline(0.5, color=C.GREY, ls=":", label="usual threshold 0.50")
    ax2 = ax1.twinx()
    ax2.plot(curve["threshold"], curve["approval_rate"], color=C.BLUE, lw=2, label="approval rate")
    ax2.plot(curve["threshold"], curve["good_clients_approved"], color=C.BLUE, ls="--",
             label="good clients approved")
    ax2.set_ylabel("Share of clients", color=C.BLUE)
    ax2.set_ylim(0, 1)
    ax2.grid(False)
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [line.get_label() for line in lines], loc="center right", fontsize=8)
    ax1.set_title("Trade-off between cost and access to credit (5-fold CV, training set)")
    C.save_fig(fig, "model_threshold_tradeoff.png")


def compare_strategies(y_test, p_test):
    """Expected cost of simple rules and of the model, on the test set."""
    y_test = np.asarray(y_test)
    rules = {
        "approve everyone": np.zeros_like(p_test),   # nobody reaches the threshold
        "reject everyone": np.ones_like(p_test),     # everybody is above it
        "model, threshold 0.50": p_test,
        f"model, threshold {C.THRESHOLD:.3f}": p_test,
    }
    rows = []
    for name, p in rules.items():
        t = 0.5 if "0.50" in name else C.THRESHOLD
        rows.append({"strategy": name, **C.decision_summary(y_test, p, t)})
    out = pd.DataFrame(rows).drop(columns="threshold")
    best_simple = out.loc[out["strategy"].isin(["approve everyone", "reject everyone"]), "expected_cost"].min()
    out["saving_vs_best_simple_rule"] = 1 - out["expected_cost"] / best_simple
    return out.round(3)


def adjust_to_real_default_rate(p, sample_rate=0.30, real_rate=0.05):
    """The bank put many more bad credits in the sample (30%) than in reality
    (about 5%, Groemping 2019). This 'prior correction' rescales the odds of
    each probability to the real default rate (Saerens et al., 2002)."""
    odds = p / (1 - p)
    odds *= (real_rate / (1 - real_rate)) / (sample_rate / (1 - sample_rate))
    return odds / (1 + odds)


def risk_bands(y, p):
    """Split clients into 5 equal-size bands by score and compute the default
    rate in each band with a 95% Wilson confidence interval."""
    df = pd.DataFrame({"p": p, "default": np.asarray(y)})
    df["band"] = pd.qcut(df["p"], 5, labels=["A (lowest risk)", "B", "C", "D", "E (highest risk)"])
    out = df.groupby("band", observed=True).agg(
        n=("default", "size"), defaults=("default", "sum"),
        mean_score=("p", "mean"), min_score=("p", "min"), max_score=("p", "max"))
    low, high = proportion_confint(out["defaults"], out["n"], method="wilson")
    out["default_rate"] = out["defaults"] / out["n"]
    out["ci_low"], out["ci_high"] = low, high
    out["suggested_action"] = [
        "approve, standard terms",
        "approve, standard terms",
        "approve with smaller amount or shorter duration",
        "manual review (guarantor, collateral)",
        "decline or secured loan only",
    ]
    return out.reset_index().round(3)


def plot_risk_bands(bands):
    fig, ax = plt.subplots(figsize=(7, 4))
    rate = 100 * bands["default_rate"]
    err = [rate - 100 * bands["ci_low"], 100 * bands["ci_high"] - rate]
    ax.bar(bands["band"].astype(str), rate, yerr=err, capsize=4, color=C.BLUE)
    ax.axhline(30, color=C.RED, ls="--", label="average (30%)")
    ax.set_ylabel("Observed default rate (%)")
    ax.set_title("Default rate by risk band (5-fold CV, all 1,000 clients, 95% CI)")
    ax.tick_params(axis="x", labelsize=8)
    ax.legend()
    C.save_fig(fig, "model_risk_bands.png")


def main():
    print("STEP 3 - Model, threshold and segmentation")
    df = C.load_data()
    train, test = C.split_data(df)
    X_train, y_train = train[C.MODEL_FEATURES], train["default"]
    X_test, y_test = test[C.MODEL_FEATURES], test["default"]

    # 1-2. Class imbalance: normal vs balanced weights
    imbalance, scores = compare_imbalance_methods(X_train, y_train)
    C.save_table(imbalance, "model_imbalance_comparison.csv")
    print("\nNormal vs balanced model (5-fold CV on training set):")
    print(imbalance.to_string(index=False))
    plot_calibration(y_train, scores)

    # 3. Threshold: theory (1/6) vs the best threshold found on the training CV scores
    curve = threshold_table(y_train, scores["normal"])
    C.save_table(curve.round(3), "model_threshold_curve.csv")
    best_cv = curve.loc[curve["expected_cost"].idxmin(), "threshold"]
    print(f"\nThreshold from the cost matrix: {C.THRESHOLD:.3f} | best on training CV: {best_cv:.2f}")
    plot_threshold(curve)

    # Final model: trained once on the full training set, evaluated once on the test set
    model = C.make_model().fit(X_train, y_train)
    p_test = model.predict_proba(X_test)[:, 1]
    auc_test = roc_auc_score(y_test, p_test)
    print(f"Test ROC-AUC = {auc_test:.3f}")

    strategies = compare_strategies(y_test, p_test)
    C.save_table(strategies, "model_strategy_comparison.csv")
    print("\nStrategies on the test set:")
    print(strategies.to_string(index=False))

    # What if the real default rate (about 5%) is used instead of the sample rate (30%)?
    p_real = adjust_to_real_default_rate(p_test)
    what_if = pd.DataFrame([
        {"default_rate_assumed": 0.30, "approval_rate": (p_test < C.THRESHOLD).mean()},
        {"default_rate_assumed": 0.05, "approval_rate": (p_real < C.THRESHOLD).mean()},
    ]).round(3)
    C.save_table(what_if, "model_real_default_rate_what_if.csv")
    print("\nApproval rate at threshold 1/6 under the sample vs the real default rate:")
    print(what_if.to_string(index=False))

    # 5. Risk bands on cross-validated scores for all 1,000 clients
    p_all = C.cv_scores(C.make_model(), df[C.MODEL_FEATURES], df["default"])
    bands = risk_bands(df["default"], p_all)
    C.save_table(bands, "model_risk_bands.csv")
    print("\nRisk bands:")
    print(bands[["band", "n", "default_rate", "ci_low", "ci_high", "suggested_action"]].to_string(index=False))
    plot_risk_bands(bands)

    joblib.dump(model, C.MODEL_DIR / "logistic_model.joblib")
    summary = {
        "n_train": len(train), "n_test": len(test),
        "test_roc_auc": round(auc_test, 3),
        "threshold": round(C.THRESHOLD, 3),
        "best_threshold_on_training_cv": round(float(best_cv), 2),
        "features": C.MODEL_FEATURES,
        "excluded_sensitive": C.SENSITIVE_COLS,
    }
    with open(C.MODEL_DIR / "model_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
