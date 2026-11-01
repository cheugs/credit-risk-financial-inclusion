# Credit Risk & Financial Inclusion Study

[Français](README.md) | **English**

**Question:** Which factors are associated with loan default, and how can lenders evaluate
underserved customers fairly?

**Tools:** Python, pandas, SQL (SQLite), SciPy, statsmodels, scikit-learn, SHAP.

**Data:** [Statlog German Credit](https://doi.org/10.24432/C5NC77) (UCI), 1,000 consumer loans,
30% bad credits. The code table that comes with this dataset is wrong for 8 variables, so we use
the corrected table of Grömping (2019). See the report, section 3.2.

## Main results

| Topic | Result |
| --- | --- |
| Strongest factors | Checking account status, credit history, loan duration, savings |
| Model | Logistic regression without sex, age or foreign-worker status. Test ROC-AUC = 0.80 |
| Decision rule | Threshold P(default) = 1/6 from the 5:1 cost matrix. 22% cheaper than "reject everyone" |
| Access to credit | At this threshold only 40% of applicants and 51% of good clients are approved |
| Risk bands | Default rate from 6.5% (safest band) to 62.5% (riskiest band) |
| Fairness | Clients aged 25 or less are approved much less often, even though the model does not use age (disparate impact = 0.65) |

> These results are correlations in historical data from 1973-1975. They do not prove that
> any person is or is not creditworthy.

## Reports

- English report: [report/report_en.md](report/report_en.md) ([PDF](report/report_en.pdf))
- French report: [report/rapport_fr.md](report/rapport_fr.md) ([PDF](report/rapport_fr.pdf))
- Notebook walkthrough in English: [notebooks/credit_risk_study.ipynb](notebooks/credit_risk_study.ipynb)
- Notebook walkthrough in French: [notebooks/etude_risque_credit.ipynb](notebooks/etude_risque_credit.ipynb)

## Project structure

```
data/raw/            german.data (original data) and german.doc (original documentation)
data/processed/      clean CSV and SQLite database (created by step 1)
src/
  common.py          settings, corrected code table, data loading, model and cost functions
  01_data_prep.py    step 1: data preparation and SQL queries
  02_statistics.py   step 2: chi-square, Mann-Whitney, logistic regression (odds ratios)
  03_model.py        step 3: model, class imbalance, decision threshold, risk bands
  04_shap.py         step 4: SHAP explanations
  05_fairness.py     step 5: fairness audit with bootstrap confidence intervals
  run_all.py         runs steps 1 to 5
notebooks/           step-by-step notebooks with results (English and French)
results/             figures, tables and the saved model (created by the scripts)
report/              full report in English and French (Markdown and PDF)
```

## How to run

```bash
pip install -r requirements.txt
python src/run_all.py
```

The random seed is fixed, so every run gives the same results.

## Method in five steps

1. **Data and SQL:** apply the corrected code table, store the data in SQLite, compute default
   rates by segment with SQL.
2. **Statistics:** chi-square tests with Cramér's V, Mann-Whitney U tests with rank-biserial
   correlation, Bonferroni correction, and a logistic regression with odds ratios.
3. **Model:** logistic regression without sensitive data. Comparison of a normal and a
   "balanced" model (same ranking, but only the normal model gives realistic probabilities).
   Threshold from the cost matrix, comparison with simple rules, 5 risk bands.
4. **SHAP:** global importance and explanations for single clients.
5. **Fairness:** disparate impact and equal opportunity by age and foreign-worker status, with
   bootstrap confidence intervals, and a "blind" vs "full" model comparison.

## Limitations

- Only approved clients are in the data (selection bias), and bad credits were oversampled
  (30% in the sample, about 5% in reality).
- Small and old data (1,000 loans, southern Germany, 1973-1975).
- Sex cannot be recovered from the data, so it is not audited. Only 37 clients are foreign
  workers, so results for this group are uncertain.

## Main references

- Hofmann, H. (1994). *Statlog (German Credit Data)*. UCI Machine Learning Repository. https://doi.org/10.24432/C5NC77
- Grömping, U. (2019). *South German Credit Data: Correcting a Widely Used Data Set*. Report 4/2019, Beuth University of Applied Sciences Berlin.
- Elkan, C. (2001). The foundations of cost-sensitive learning. *IJCAI*.
- Kozodoi, N., Jacob, J., & Lessmann, S. (2022). Fairness in credit scoring. *European Journal of Operational Research, 297*(3).

The full list of references is at the end of each report.
