# eda_automation_pipeline/modeling.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


def train_and_evaluate(X_train, y_train, X_test, y_test, preprocessor, problem_type):
    fitted_pipelines = {}
    results = []

    if problem_type in ["Binary Classification", "Multiclass Classification"]:
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest Classifier": RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
        }
        avg_method = "binary" if problem_type == "Binary Classification" else "weighted"

        for name, model in models.items():
            clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
            clf.fit(X_train, y_train)
            fitted_pipelines[name] = clf
            y_pred = clf.predict(X_test)

            results.append(
                {
                    "Model": name,
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision_score(
                        y_test, y_pred, average=avg_method, zero_division=0
                    ),
                    "Recall": recall_score(
                        y_test, y_pred, average=avg_method, zero_division=0
                    ),
                    "F1-Score": f1_score(
                        y_test, y_pred, average=avg_method, zero_division=0
                    ),
                }
            )
    else:
        models = {
            "Ridge Regression": Ridge(alpha=1.0, random_state=42),
            "Random Forest Regressor": RandomForestRegressor(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
        }
        for name, model in models.items():
            reg = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
            reg.fit(X_train, y_train)
            fitted_pipelines[name] = reg
            y_pred = reg.predict(X_test)

            results.append(
                {
                    "Model": name,
                    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
                    "MAE": mean_absolute_error(y_test, y_pred),
                    "R²": r2_score(y_test, y_pred),
                }
            )

    return fitted_pipelines, pd.DataFrame(results)
