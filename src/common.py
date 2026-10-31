"""
Shared settings and helper functions used by every step of the project.

Dataset: Statlog (German Credit Data), UCI Machine Learning Repository
(Hofmann, 1994). https://doi.org/10.24432/C5NC77

Important: the code table shipped with the data (german.doc) is wrong for
8 of the 20 variables. Groemping (2019) rebuilt the correct code table from
the original German sources. All labels below follow her corrected table.
"""
from pathlib import Path

import matplotlib

# Use a non-interactive backend when the scripts run from the terminal.
# Inside Jupyter we keep the inline backend so plots show in the notebook.
try:
    get_ipython()  # type: ignore[name-defined]  # only exists inside Jupyter
except NameError:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --------------------------------------------------------------------------- #
# Folders
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "data" / "raw" / "german.data"
PROCESSED_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "results" / "figures"
TABLE_DIR = ROOT / "results" / "tables"
MODEL_DIR = ROOT / "results" / "models"
DB_PATH = PROCESSED_DIR / "german_credit.db"

for folder in (PROCESSED_DIR, FIG_DIR, TABLE_DIR, MODEL_DIR):
    folder.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42  # fixed seed so every run gives the same results

# --------------------------------------------------------------------------- #
# Plot style
# --------------------------------------------------------------------------- #
sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "axes.titleweight": "bold"})
BLUE, RED, GREEN, GREY = "#1F6F8B", "#C0392B", "#1E8449", "#7F8C8D"


def save_fig(fig, name):
    """Save a figure in results/figures and close it."""
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, bbox_inches="tight")
    plt.close(fig)


def save_table(df, name):
    """Save a table in results/tables as CSV."""
    df.to_csv(TABLE_DIR / name, index=False)


# --------------------------------------------------------------------------- #
# Columns of german.data (in file order)
# --------------------------------------------------------------------------- #
RAW_COLUMNS = [
    "checking_status", "duration_months", "credit_history", "purpose",
    "credit_amount", "savings", "employment", "installment_rate",
    "personal_status_sex", "other_debtors", "residence_since", "property",
    "age", "other_installment_plans", "housing", "existing_credits", "job",
    "people_liable", "telephone", "foreign_worker", "credit_label",
]

# Corrected code table (Groemping, 2019, Table 1 and Appendix C).
# Variables marked (*) had wrong labels in the original german.doc.
VALUE_LABELS = {
    "checking_status": {  # (*)
        "A11": "no checking account",
        "A12": "< 0 DM",
        "A13": "0 to 200 DM",
        "A14": ">= 200 DM or salary account",
    },
    "credit_history": {  # (*)
        "A30": "delay in paying off in the past",
        "A31": "critical account / credits elsewhere",
        "A32": "no credits taken / all paid back duly",
        "A33": "existing credits paid back duly",
        "A34": "all credits at this bank paid back duly",
    },
    "purpose": {  # (*)
        "A40": "others",
        "A41": "car (new)",
        "A42": "car (used)",
        "A43": "furniture / equipment",
        "A44": "radio / television",
        "A45": "domestic appliances",
        "A46": "repairs",
        "A47": "education",
        "A48": "vacation",
        "A49": "retraining",
        "A410": "business",
    },
    "savings": {  # (*)
        "A61": "unknown / no savings account",
        "A62": "< 100 DM",
        "A63": "100 to 500 DM",
        "A64": "500 to 1000 DM",
        "A65": ">= 1000 DM",
    },
    "employment": {
        "A71": "unemployed",
        "A72": "< 1 year",
        "A73": "1 to 4 years",
        "A74": "4 to 7 years",
        "A75": ">= 7 years",
    },
    # installment_rate is stored as a number 1-4 in the raw file.
    # A HIGHER code means a LOWER share of income (Groemping, 2019).
    "installment_rate": {
        "1": ">= 35% of income",
        "2": "25 to 35% of income",
        "3": "20 to 25% of income",
        "4": "< 20% of income",
    },
    "personal_status_sex": {  # (*) sex CANNOT be recovered from this column
        "A91": "male: divorced / separated",
        "A92": "female non-single OR male single",
        "A93": "male: married / widowed",
        "A94": "female: single",
    },
    "other_debtors": {
        "A101": "none",
        "A102": "co-applicant",
        "A103": "guarantor",
    },
    "property": {  # (*)
        "A121": "unknown / no property",
        "A122": "car or other",
        "A123": "savings agreement / life insurance",
        "A124": "real estate",
    },
    "other_installment_plans": {
        "A141": "bank",
        "A142": "stores",
        "A143": "none",
    },
    "housing": {  # (*)
        "A151": "for free",
        "A152": "rent",
        "A153": "own",
    },
    "job": {
        "A171": "unemployed / unskilled non-resident",
        "A172": "unskilled resident",
        "A173": "skilled employee",
        "A174": "manager / self-employed / highly qualified",
    },
    "telephone": {
        "A191": "no",
        "A192": "yes",
    },
    "foreign_worker": {  # (*) the original table had yes/no swapped
        "A201": "no",
        "A202": "yes",
    },
}

# Columns used as numbers. The first three are real quantities; the last
# three are small ordered scores (e.g. residence_since: 1 = < 1 year ... 4 = >= 7 years).
NUMERIC_COLS = [
    "duration_months", "credit_amount", "age",
    "residence_since", "existing_credits", "people_liable",
]
CATEGORICAL_COLS = list(VALUE_LABELS.keys())

# Sensitive attributes: kept ONLY to check fairness, never used by the model.
SENSITIVE_COLS = ["personal_status_sex", "foreign_worker", "age"]
MODEL_CATEGORICAL = [c for c in CATEGORICAL_COLS if c not in SENSITIVE_COLS]
MODEL_NUMERIC = [c for c in NUMERIC_COLS if c not in SENSITIVE_COLS]
MODEL_FEATURES = MODEL_CATEGORICAL + MODEL_NUMERIC

# Cost matrix given with the dataset (german.doc):
# approving a client who then defaults costs 5, rejecting a good client costs 1.
COST_FN = 5.0  # false negative: a defaulter is approved
COST_FP = 1.0  # false positive: a good client is rejected
# Decision theory: reject when P(default) > C_FP / (C_FP + C_FN)  (Elkan, 2001).
THRESHOLD = COST_FP / (COST_FP + COST_FN)  # = 1/6 = 0.167


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_data():
    """Read german.data and return a clean table with readable labels.

    Added columns:
      default      1 = bad credit (did not repay as agreed), 0 = good credit
      age_group    "<= 25" or "> 25" (used in the fairness audit)
    """
    df = pd.read_csv(RAW_FILE, sep=r"\s+", header=None, names=RAW_COLUMNS, dtype=str)

    for col in NUMERIC_COLS:
        df[col] = df[col].astype(int)

    # Replace the codes (A11, A12, ...) with readable labels.
    for col, mapping in VALUE_LABELS.items():
        df[col] = df[col].map(mapping)

    df["default"] = (df["credit_label"] == "2").astype(int)
    df = df.drop(columns="credit_label")
    df["age_group"] = np.where(df["age"] <= 25, "<= 25", "> 25")
    return df


def split_data(df):
    """Split into 75% training and 25% test, keeping the 30/70 default ratio."""
    return train_test_split(df, test_size=0.25, stratify=df["default"], random_state=RANDOM_STATE)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
def make_model(categorical=MODEL_CATEGORICAL, numeric=MODEL_NUMERIC, class_weight=None):
    """Logistic regression with one-hot encoding (categories) and scaling (numbers)."""
    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ("num", StandardScaler(), numeric),
    ])
    model = LogisticRegression(max_iter=2000, class_weight=class_weight)
    return Pipeline([("preprocess", preprocess), ("model", model)])


def cv_scores(model, X, y):
    """5-fold cross-validated scores: each client is scored by a model that never saw them."""
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_predict(model, X, y, cv=folds, method="predict_proba")[:, 1]


def expected_cost(y_true, p_default, threshold):
    """Average cost per client when we reject everyone with P(default) >= threshold."""
    y_true = np.asarray(y_true)
    reject = np.asarray(p_default) >= threshold
    false_neg = np.sum(~reject & (y_true == 1))  # approved, then defaulted
    false_pos = np.sum(reject & (y_true == 0))   # rejected, but would have repaid
    return (COST_FN * false_neg + COST_FP * false_pos) / len(y_true)


def decision_summary(y_true, p_default, threshold):
    """Key numbers of a lending rule at a given threshold."""
    y_true = np.asarray(y_true)
    approve = np.asarray(p_default) < threshold
    good, bad = y_true == 0, y_true == 1
    return {
        "threshold": round(threshold, 3),
        "approval_rate": approve.mean(),
        "good_clients_approved": approve[good].mean(),  # share of good payers who get a loan
        "defaulters_approved": approve[bad].mean(),     # share of defaulters who get a loan
        "expected_cost": expected_cost(y_true, p_default, threshold),
    }
