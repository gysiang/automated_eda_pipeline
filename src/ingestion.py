# eda_automation_pipeline/ingestion.py
import os
import sqlite3
import pandas as pd


def load_dataset(path: str) -> pd.DataFrame:
    """Connects to a CSV file or SQLite database and returns a Pandas DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data source not found at: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in [".db", ".sqlite", ".sqlite3"]:
        conn = sqlite3.connect(path)
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
        if tables.empty:
            conn.close()
            raise ValueError("No tables found in the SQLite database.")
        primary_table = tables["name"].iloc[0]
        print(f"Loading primary SQLite table: '{primary_table}'...")
        df = pd.read_sql(f"SELECT * FROM {primary_table}", conn)
        conn.close()
    elif ext == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError(
            f"Unsupported file extension: {ext}. Provide a .csv or .db file."
        )
    return df
