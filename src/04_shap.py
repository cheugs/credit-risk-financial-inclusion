"""
Step 4 - Explaining the model with SHAP.

For a logistic regression, the SHAP value of a feature is simply
    coefficient x (value of the client - average value in the training set)
on the log-odds scale. So SHAP gives the same information as the
coefficients, but per client. We check this formula below.

Outputs: global importance, beeswarm plot, and two individual explanations
(the kind of explanation a bank must be able to give to a rejected client).
"""
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

import common as C


def clean_name(name):
    """'cat__checking_status_no checking account' -> 'checking_status = no checking account'."""
    name = name.split("__", 1)[1]
    for col in C.MODEL_CATEGORICAL:
        if name.startswith(col + "_"):
            return f"{col} = {name[len(col) + 1:]}"
    return name


def main():
    print("STEP 4 - SHAP explanations")
    df = C.load_data()
    train, test = C.split_data(df)
    model = joblib.load(C.MODEL_DIR / "logistic_model.joblib")  # saved by step 3

    preprocess, logit = model.named_steps["preprocess"], model.named_steps["model"]
    X_train = preprocess.transform(train[C.MODEL_FEATURES])
    X_test = preprocess.transform(test[C.MODEL_FEATURES])
    names = [clean_name(n) for n in preprocess.get_feature_names_out()]

    explainer = shap.LinearExplainer(logit, shap.maskers.Independent(X_train, max_samples=len(X_train)))
    shap_values = explainer.shap_values(X_test)

    # Check: SHAP value = coefficient x (x - mean of training data)
    by_hand = logit.coef_[0] * (X_test - X_train.mean(axis=0))
    print(f"Max difference SHAP vs formula: {np.abs(shap_values - by_hand).max():.2e}")

    importance = pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
    importance = importance.sort_values("mean_abs_shap", ascending=False).round(3)
    C.save_table(importance, "shap_importance.csv")
    print(importance.head(10).to_string(index=False))

    # Global importance (top 15)
    top = importance.head(15).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"], top["mean_abs_shap"], color=C.BLUE)
    ax.set_xlabel("Mean |SHAP value| (log-odds of default)")
    ax.set_title("Most important features (test set)")
    ax.tick_params(axis="y", labelsize=8)
    C.save_fig(fig, "shap_importance.png")

    # Beeswarm: direction of each effect (red = feature present / high value)
    shap.summary_plot(shap_values, X_test, feature_names=names, max_display=12, show=False)
    plt.title("SHAP values: right = pushes towards default")
    C.save_fig(plt.gcf(), "shap_beeswarm.png")

    # Individual explanations for the safest and the riskiest client of the test set
    p_test = model.predict_proba(test[C.MODEL_FEATURES])[:, 1]
    for label, idx in [("safest", int(np.argmin(p_test))), ("riskiest", int(np.argmax(p_test)))]:
        explanation = shap.Explanation(values=shap_values[idx], base_values=explainer.expected_value,
                                       data=X_test[idx], feature_names=names)
        shap.waterfall_plot(explanation, max_display=8, show=False)
        plt.gcf().set_size_inches(10, 5)
        plt.title(f"{label.capitalize()} test client: P(default) = {p_test[idx]:.2f}")
        C.save_fig(plt.gcf(), f"shap_waterfall_{label}.png")


if __name__ == "__main__":
    main()
