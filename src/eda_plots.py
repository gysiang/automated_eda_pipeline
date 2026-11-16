# eda_automation_pipeline/eda_plots.py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def detect_problem_type(target_series: pd.Series) -> str:
    unique_targets = target_series.nunique()
    if pd.api.types.is_numeric_dtype(target_series) and unique_targets > 15:
        return "Regression"
    elif unique_targets == 2:
        return "Binary Classification"
    else:
        return "Multiclass Classification"


def plot_univariate_and_bivariate(df: pd.DataFrame, target_col: str, problem_type: str):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_features = [c for c in num_cols if c != target_col]
    cat_features = [c for c in cat_cols if c != target_col]

    # Univariate numerical plots
    for col in [c for c in num_features if df[c].nunique() > 2][:6]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        sns.histplot(df[col].dropna(), kde=True, ax=axes[0], color="steelblue")
        axes[0].set_title(f"Distribution of {col} (Skew: {df[col].skew():.2f})")
        sns.boxplot(x=df[col].dropna(), ax=axes[1], color="lightcoral")
        axes[1].set_title(f"Boxplot of {col}")
        plt.tight_layout()
        plt.show()

    # Bivariate plots
    if problem_type in ["Binary Classification", "Multiclass Classification"]:
        for col in num_features[:4]:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            sns.boxplot(
                data=df,
                x=target_col,
                y=col,
                ax=axes[0],
                palette="Set2",
                hue=target_col,
                legend=False,
            )
            sns.violinplot(
                data=df,
                x=target_col,
                y=col,
                ax=axes[1],
                palette="Set2",
                inner="quartile",
                hue=target_col,
                legend=False,
            )
            plt.tight_layout()
            plt.show()
    else:
        for col in num_features[:4]:
            plt.figure(figsize=(7, 4))
            sns.regplot(
                data=df,
                x=col,
                y=target_col,
                scatter_kws={"alpha": 0.4},
                line_kws={"color": "red"},
            )
            plt.tight_layout()
            plt.show()


def plot_correlation_heatmap(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) > 1:
        corr = df[num_cols].corr(method="pearson")
        mask = np.triu(np.ones_like(corr, dtype=bool))
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            corr,
            mask=mask,
            annot=len(num_cols) <= 10,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
        )
        plt.title("Pearson Correlation Matrix")
        plt.tight_layout()
        plt.show()
