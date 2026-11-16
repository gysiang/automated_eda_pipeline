# Automated EDA & Baseline ML Pipeline (`eda_automation_pipeline`)

A modular, production-ready Python framework designed to rapidly ingest raw datasets (CSV or SQLite `.db`), clean data anomalies, conduct rigorous Exploratory Data Analysis (EDA), build scikit-learn preprocessing pipelines, train baseline machine learning models, and automatically compile everything into a fully documented Jupyter Notebook (`.ipynb`).

---

## 🎯 Aim & Purpose

Before committing time and computational resources to building a heavy, complex machine learning pipeline, data scientists need a fast, reliable diagnostic overview of their data. This tool bridges the gap by automating:
1. **Data Ingestion & Inspection:** Automatically handles CSV and SQLite databases, inspecting data types, missingness, and structural shapes.
2. **Automated Data Cleaning:** Cleans duplicate records, strips whitespace, coerces dirty numerical strings, and filters out high-missingness or redundant columns.
3. **Exploratory Data Analysis (EDA):** Generates univariate distributions, bivariate target-relationship plots, and correlation matrices to spot multicollinearity.
4. **Leakage-Safe Preprocessing:** Encapsulates imputation and scaling/encoding inside modular scikit-learn `ColumnTransformer` and `Pipeline` objects.
5. **Baseline Modeling & Evaluation:** Trains lightweight baseline models (Logistic Regression, Random Forest, Ridge Regression) and evaluates them with comprehensive classification/regression metrics.
6. **Feature Importance Visualizations:** Extracts native coefficients/impurity importances and runs model-agnostic test-set permutation importances.

---

## 📁 Project Structure

```text
eda_automation_pipeline/
├── __init__.py
├── config.py              # Environment configuration & plot styling
├── ingestion.py           # Multi-source data loader (CSV & SQLite .db)
├── cleaning.py            # Automated data cleaning & type coercion
├── eda_plots.py           # Univariate, bivariate, & correlation visualization
├── preprocessing.py       # Modular scikit-learn ColumnTransformer builders
├── modeling.py            # Baseline model fitting & performance evaluation
├── feature_importance.py  # Native coefficient & test-set permutation importance plots
└── notebook_generator.py  # Orchestrator to programmatically build the .ipynb file
