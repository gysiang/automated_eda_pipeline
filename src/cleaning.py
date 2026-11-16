# eda_automation_pipeline/cleaning.py
import numpy as np
import pandas as pd


def clean_dataset(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Cleans raw dataframe by dropping duplicates, coercing dirty strings, and filtering high-missingness columns."""
    df_clean = df.copy()

    # 1. Drop exact duplicates
    initial_rows = len(df_clean)
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)
    print(f"Dropped {initial_rows - len(df_clean)} duplicate rows.")

    # 2. String strip and dirty numeric coercion
    for col in df_clean.select_dtypes(include=["object"]).columns:
        df_clean[col] = df_clean[col].astype(str).str.strip()
        df_clean[col] = df_clean[col].replace(
            {"nan": np.nan, "None": np.nan, "": np.nan, "NULL": np.nan, "?": np.nan}
        )

        if col != target_col:
            cleaned_series = df_clean[col].str.replace(r"[$,%]", "", regex=True)
            coerced = pd.to_numeric(cleaned_series, errors="coerce")
            valid_ratio = coerced.notnull().sum() / max(
                df_clean[col].notnull().sum(), 1
            )
            if valid_ratio > 0.80:
                df_clean[col] = coerced

    # 3. Drop columns with >50% missing data (excluding target)
    missing_pct = (df_clean.isnull().sum() / len(df_clean)) * 100
    high_missing = [c for c in missing_pct[missing_pct > 50.0].index if c != target_col]
    if high_missing:
        df_clean = df_clean.drop(columns=high_missing)
        print(f"Dropped columns with >50% missing values: {high_missing}")

    # 4. Drop constant or unique ID columns
    drop_cols = []
    for col in df_clean.columns:
        if col == target_col:
            continue
        if df_clean[col].nunique(dropna=False) <= 1:
            drop_cols.append(col)
        elif (df_clean[col].nunique() == len(df_clean)) and (
            pd.api.types.is_string_dtype(df_clean[col])
            or pd.api.types.is_integer_dtype(df_clean[col])
        ):
            drop_cols.append(col)

    if drop_cols:
        df_clean = df_clean.drop(columns=drop_cols)
        print(f"Dropped constant or unique ID columns: {drop_cols}")

    # 5. Drop rows with missing target
    if target_col in df_clean.columns:
        target_nulls = df_clean[target_col].isnull().sum()
        if target_nulls > 0:
            df_clean = df_clean.dropna(subset=[target_col]).reset_index(drop=True)
            print(f"Dropped {target_nulls} rows with missing target '{target_col}'.")

    return df_clean
