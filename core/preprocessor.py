# core/preprocessor.py
import pandas as pd
import numpy as np

def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()

def drop_high_missing(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Elimina columnas con más de `threshold` proporción de nulos."""
    return df.loc[:, df.isna().mean() < threshold]

def fill_missing(df, numeric_strategy="median", categorical_strategy="mode"):
    df = df.copy()
    num_cols = df.select_dtypes("number").columns
    cat_cols = df.select_dtypes(["object", "category"]).columns

    for col in num_cols:
        if numeric_strategy == "median":
            df[col] = df[col].fillna(df[col].median())
        elif numeric_strategy == "mean":
            df[col] = df[col].fillna(df[col].mean())
        elif numeric_strategy == "drop":
            df = df.dropna(subset=[col])

    for col in cat_cols:
        if categorical_strategy == "mode":
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "desconocido")
        elif categorical_strategy == "constant":
            df[col] = df[col].fillna("desconocido")
        elif categorical_strategy == "drop":
            df = df.dropna(subset=[col])

    return df.reset_index(drop=True)

def drop_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.drop(columns=[c for c in cols if c in df.columns])

def encode_categoricals(df: pd.DataFrame, max_cardinality: int = 20):
    """One-hot encoding para categóricas de baja cardinalidad."""
    cat_cols = df.select_dtypes(["object", "category", "bool"]).columns
    low_card = [c for c in cat_cols if df[c].nunique() <= max_cardinality]
    return pd.get_dummies(df, columns=low_card, drop_first=True)