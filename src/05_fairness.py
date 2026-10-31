"""
Step 5 - Fairness audit.

Question: does the model treat some groups worse, even though it does not
use sex, age or foreign-worker status?

We use cross-validated scores for all 1,000 clients (more people per group
than the test set alone) and the cost threshold 1/6. For each group:
  - approval rate
  - disparate impact (DI) = approval rate of the group / approval rate of the
    reference group. The "80% rule" flags DI < 0.8.
  - approval rate among good payers (equal opportunity: people who would repay
    should have the same chance to get a loan).
Small groups give noisy numbers, so we add 95% bootstrap confidence intervals.

Groups audited:
  - age: <= 25 vs > 25
  - foreign worker: yes (only 37 clients!) vs no
Sex is NOT audited: in this dataset single men and non-single women share the
same code, so sex cannot be recovered (Groemping, 2019).

Finally, a "blind" model (no sensitive data) is compared with a "full" model
(with age and foreign-worker status) to see if removing the data removes the gap.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import common as C

GROUPS = {  # attribute: (protected group, reference group)
    "age_group": ("<= 25", "> 25"),
    "foreign_worker": ("yes", "no"),
}
N_BOOTSTRAP = 2000


def group_metrics(y, approve, group, protected, reference):
    """Approval rates and gaps between the protected and the reference group."""
    def rates(mask):
        return approve[mask].mean(), approve[mask & (y == 0)].mean()

    appr_p, good_p = rates(group == protected)
    appr_r, good_r = rates(group == reference)
    return {
        "approval_protected": appr_p, "approval_reference": appr_r,
        "disparate_impact": appr_p / appr_r,
        "good_payers_approved_protected": good_p, "good_payers_approved_reference": good_r,
        "equal_opportunity_gap": good_p - good_r,
    }


def audit(y, p, df, rng):
    """Point estimates and bootstrap 95% CIs for every audited attribute."""
    y = np.asarray(y)
    approve = p < C.THRESHOLD
    rows = []
    for attr, (protected, reference) in GROUPS.items():
        group = df[attr].to_numpy()
        point = group_metrics(y, approve, group, protected, reference)

        # Bootstrap: resample clients with replacement and recompute the metrics.
        boot = []
        for _ in range(N_BOOTSTRAP):
            i = rng.integers(0, len(y), len(y))
            boot.append(group_metrics(y[i], approve[i], group[i], protected, reference))
        boot = pd.DataFrame(boot)

        rows.append({
            "attribute": attr, "protected_group": protected, "reference_group": reference,
            "n_protected": int((group == protected).sum()),
            "default_rate_protected": y[group == protected].mean(),
            "default_rate_reference": y[group == reference].mean(),
            **point,
            "di_ci_low": boot["disparate_impact"].quantile(0.025),
            "di_ci_high": boot["disparate_impact"].quantile(0.975),
            "eo_gap_ci_low": boot["equal_opportunity_gap"].quantile(0.025),
            "eo_gap_ci_high": boot["equal_opportunity_gap"].quantile(0.975),
        })
    return pd.DataFrame(rows).round(3)


def plot_audit(table):
    """Disparate impact with 95% CI for each attribute and model."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    labels = table["model"] + " | " + table["attribute"]
    err = [table["disparate_impact"] - table["di_ci_low"], table["di_ci_high"] - table["disparate_impact"]]
    ax.errorbar(table["disparate_impact"], range(len(table)), xerr=err, fmt="o", color=C.BLUE, capsize=4)
    ax.axvline(0.8, color=C.RED, ls="--", label="80% rule")
    ax.axvline(1.0, color="black", lw=1, label="equal approval")
    ax.set_yticks(range(len(table)), labels)
    ax.set_xlabel("Disparate impact (approval ratio, 95% bootstrap CI)")
    ax.set_title(f"Fairness audit at threshold {C.THRESHOLD:.3f}")
    ax.legend(fontsize=8)
    C.save_fig(fig, "fairness_disparate_impact.png")


def main():
    print("STEP 5 - Fairness audit")
    df = C.load_data()
    y = df["default"]
    rng = np.random.default_rng(C.RANDOM_STATE)

    blind = C.make_model()
    full = C.make_model(C.MODEL_CATEGORICAL + ["foreign_worker"], C.MODEL_NUMERIC + ["age"])
    results = []
    for name, model, features in [
        ("blind", blind, C.MODEL_FEATURES),
        ("full", full, C.MODEL_FEATURES + ["foreign_worker", "age"]),
    ]:
        p = C.cv_scores(model, df[features], y)
        table = audit(y, p, df, rng)
        table.insert(0, "model", name)
        table.insert(1, "cv_roc_auc", round(roc_auc_score(y, p), 3))
        results.append(table)

    results = pd.concat(results, ignore_index=True)
    C.save_table(results, "fairness_audit.csv")
    cols = ["model", "cv_roc_auc", "attribute", "n_protected", "approval_protected",
            "approval_reference", "disparate_impact", "di_ci_low", "di_ci_high",
            "equal_opportunity_gap", "eo_gap_ci_low", "eo_gap_ci_high"]
    print(results[cols].to_string(index=False))
    plot_audit(results)


if __name__ == "__main__":
    main()
