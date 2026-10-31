"""
Step 1 - Data preparation and SQL analysis.

1. Load the raw file and apply the corrected code table.
2. Store the clean table in a SQLite database.
3. Answer simple portfolio questions with SQL (default rate by segment).
4. Save a clean CSV for the next steps.
"""
import sqlite3

import pandas as pd

import common as C

# Each query answers one simple business question.
QUERIES = {
    "sql_01_overall": """
        SELECT COUNT(*)                          AS n_clients,
               SUM(is_default)                   AS n_defaults,
               ROUND(100.0 * AVG(is_default), 1) AS default_rate_pct
        FROM applicants;
    """,
    "sql_02_by_checking_status": """
        SELECT checking_status, COUNT(*) AS n,
               ROUND(100.0 * AVG(is_default), 1) AS default_rate_pct
        FROM applicants
        GROUP BY checking_status
        ORDER BY default_rate_pct DESC;
    """,
    "sql_03_by_credit_history": """
        SELECT credit_history, COUNT(*) AS n,
               ROUND(100.0 * AVG(is_default), 1) AS default_rate_pct
        FROM applicants
        GROUP BY credit_history
        ORDER BY default_rate_pct DESC;
    """,
    # HAVING removes very small groups, whose rates are not reliable.
    "sql_04_by_purpose": """
        SELECT purpose, COUNT(*) AS n,
               ROUND(100.0 * AVG(is_default), 1) AS default_rate_pct
        FROM applicants
        GROUP BY purpose
        HAVING COUNT(*) >= 20
        ORDER BY default_rate_pct DESC;
    """,
    "sql_05_loan_size_by_outcome": """
        SELECT CASE WHEN is_default = 1 THEN 'default' ELSE 'repaid' END AS outcome,
               COUNT(*)                       AS n,
               ROUND(AVG(credit_amount), 0)   AS avg_amount_dm,
               ROUND(AVG(duration_months), 1) AS avg_duration_months
        FROM applicants
        GROUP BY outcome;
    """,
    "sql_06_by_age_band": """
        SELECT CASE WHEN age <= 25 THEN '1) 19-25'
                    WHEN age <= 35 THEN '2) 26-35'
                    WHEN age <= 50 THEN '3) 36-50'
                    ELSE '4) 51+' END AS age_band,
               COUNT(*) AS n,
               ROUND(100.0 * AVG(is_default), 1) AS default_rate_pct
        FROM applicants
        GROUP BY age_band
        ORDER BY age_band;
    """,
}


def main():
    print("STEP 1 - Data preparation and SQL")

    df = C.load_data()
    print(f"Loaded {len(df)} clients and {df.shape[1]} columns.")

    # "default" is a reserved word in SQL, so the column is renamed in the database.
    with sqlite3.connect(C.DB_PATH) as conn:
        df.rename(columns={"default": "is_default"}).to_sql(
            "applicants", conn, if_exists="replace", index=False
        )
        for name, sql in QUERIES.items():
            result = pd.read_sql_query(sql, conn)
            C.save_table(result, f"{name}.csv")
            print(f"\n-- {name}\n{result.to_string(index=False)}")

    df.to_csv(C.PROCESSED_DIR / "german_credit_clean.csv", index=False)
    print(f"\nClean data saved to {C.PROCESSED_DIR / 'german_credit_clean.csv'}")


if __name__ == "__main__":
    main()
