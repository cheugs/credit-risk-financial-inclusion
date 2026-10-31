"""
Run the whole project from start to end:  python src/run_all.py
Results are written to results/figures, results/tables and results/models.
"""
import importlib

STEPS = ["01_data_prep", "02_statistics", "03_model", "04_shap", "05_fairness"]

if __name__ == "__main__":
    for step in STEPS:
        print("\n" + "=" * 70)
        importlib.import_module(step).main()
    print("\nDone. See the results/ folder.")
