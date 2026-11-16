# eda_automation_pipeline/feature_importance.py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


def plot_feature_importances(
    fitted_pipelines: dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    problem_type: str,
    top_n: int = 15,
):
    # 1. Native Importances
    for name, pipe in fitted_pipelines.items():
        preproc = pipe.named_steps["preprocessor"]
        estimator = pipe.named_steps["model"]

        raw_names = preproc.get_feature_names_out()
        feature_names = [f.split("__", 1)[-1] for f in raw_names]

        if hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            coef = estimator.coef_
            importances = (
                np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
            )
        else:
            continue

        fi_df = (
            pd.DataFrame({"Feature": feature_names, "Importance": importances})
            .sort_values(by="Importance", ascending=False)
            .head(top_n)
        )

        plt.figure(figsize=(10, 5))
        sns.barplot(
            data=fi_df,
            x="Importance",
            y="Feature",
            palette="viridis",
            hue="Feature",
            legend=False,
        )
        plt.title(f"{name} - Top {top_n} Native Feature Importances")
        plt.tight_layout()
        plt.show()

    # 2. Permutation Importances
    scoring_metric = (
        "f1_weighted" if problem_type != "Regression" else "neg_root_mean_squared_error"
    )
    for name, pipe in fitted_pipelines.items():
        res = permutation_importance(
            pipe,
            X_test,
            y_test,
            n_repeats=5,
            random_state=42,
            scoring=scoring_metric,
            n_jobs=-1,
        )
        perm_df = (
            pd.DataFrame(
                {"Feature": X_test.columns, "Importance": res.importances_mean}
            )
            .sort_values(by="Importance", ascending=False)
            .head(top_n)
        )

        plt.figure(figsize=(10, 5))
        sns.barplot(
            data=perm_df,
            x="Importance",
            y="Feature",
            palette="magma",
            hue="Feature",
            legend=False,
        )
        plt.title(f"{name} - Top {top_n} Test Permutation Importances")
        plt.tight_layout()
        plt.show()
