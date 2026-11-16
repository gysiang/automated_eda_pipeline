# eda_automation_pipeline/notebook_generator.py
import argparse
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell


def generate_notebook(
    data_path: str,
    target_column: str,
    output_path: str = "automated_eda_baseline.ipynb",
):
    nb = new_notebook()
    cells = [
        new_markdown_cell(
            f"# Automated EDA & Baseline ML Notebook\n**Data:** `{data_path}` | **Target:** `{target_column}`"
        ),
        new_code_cell(f"""
import pandas as pd
from eda_automation_pipeline.config import configure_environment
from eda_automation_pipeline.ingestion import load_dataset
from eda_automation_pipeline.cleaning import clean_dataset
from eda_automation_pipeline.eda_plots import detect_problem_type, plot_univariate_and_bivariate, plot_correlation_heatmap
from eda_automation_pipeline.preprocessing import build_preprocessor
from eda_automation_pipeline.modeling import train_and_evaluate
from eda_automation_pipeline.feature_importance import plot_feature_importances
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

configure_environment()
df_raw = load_dataset(r"{data_path}")
TARGET_COL = "{target_column}"

df = clean_dataset(df_raw, TARGET_COL)
PROBLEM_TYPE = detect_problem_type(df[TARGET_COL])
print(f"Detected Problem Type: {{PROBLEM_TYPE}}")

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].copy()

if PROBLEM_TYPE != "Regression":
    le = LabelEncoder()
    y = pd.Series(le.fit_transform(y.astype(str)), index=y.index)

preprocessor = build_preprocessor(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

fitted_pipes, results_df = train_and_evaluate(X_train, y_train, X_test, y_test, preprocessor, PROBLEM_TYPE)
display(results_df)

plot_feature_importances(fitted_pipes, X_test, y_test, PROBLEM_TYPE)
        """),
    ]
    nb["cells"] = cells
    with open(output_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Generated notebook: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--target_column", type=str, required=True)
    args = parser.parse_args()
    generate_notebook(args.data_path, args.target_column)
