# core/preprocessor.py
"""Transformaciones puras `DataFrame -> DataFrame` y transformadores sklearn.

Las transformaciones se pueden aplicar de dos formas equivalentes:

- como funciones (`cast_columns`, `normalize_columns`), para transformar el
  dataset que se muestra y se exporta;
- como pasos del pipeline (`ColumnTyper`, `ColumnNormalizer`), para que un
  modelo guardado aplique exactamente la misma transformación a datos nuevos.
"""
from typing import Optional

import numpy as np
import pandas as pd
from pandas.api import types as pdt
from sklearn.base import BaseEstimator, TransformerMixin

#: Métodos de normalización ofrecidos al usuario.
NORMALIZATION_METHODS = ("none", "standard", "minmax", "robust", "log")


def numeric_columns(df: pd.DataFrame) -> list:
    """Columnas numéricas del dataset (excluye booleanos)."""
    return [
        column
        for column in df.columns
        if pdt.is_numeric_dtype(df[column]) and not pdt.is_bool_dtype(df[column])
    ]


def categorical_columns(df: pd.DataFrame) -> list:
    """Columnas no numéricas: texto, categorías, booleanos y fechas.

    Se define por exclusión porque en pandas 3 las columnas de texto tienen
    dtype `str` y `select_dtypes(["object"])` ya no las incluye.
    """
    return [
        column
        for column in df.columns
        if not pdt.is_numeric_dtype(df[column]) or pdt.is_bool_dtype(df[column])
    ]


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()

def drop_high_missing(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Elimina columnas con más de `threshold` proporción de nulos."""
    return df.loc[:, df.isna().mean() < threshold]

def fill_missing(df, numeric_strategy="median", categorical_strategy="mode"):
    df = df.copy()
    num_cols = numeric_columns(df)
    cat_cols = categorical_columns(df)

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
    low_card = [c for c in categorical_columns(df) if df[c].nunique() <= max_cardinality]
    return pd.get_dummies(df, columns=low_card, drop_first=True)


# ---------------------------------------------------------------------------
# Tipos de datos
# ---------------------------------------------------------------------------

CAST_TARGETS = ("numerico", "categorico", "booleano", "fecha", "texto")


def cast_column(series: pd.Series, target_type: str) -> pd.Series:
    """Convierte una columna al tipo semántico indicado.

    Los valores que no se pueden convertir pasan a `NaN`, para que después
    los imputed o eliminen.
    """
    if target_type == "numerico":
        return pd.to_numeric(series, errors="coerce")
    if target_type == "categorico":
        return _as_object(series)
    if target_type == "texto":
        return _as_object(series)
    if target_type == "booleano":
        return _to_bool(series)
    if target_type == "fecha":
        return pd.to_datetime(series, errors="coerce")
    raise ValueError(f"Tipo de dato desconocido: {target_type}")


def _as_object(series: pd.Series) -> pd.Series:
    """Devuelve la columna como `object`, conservando `NaN` como valor vacío.

    No se usa el dtype `string` de pandas porque sus nulos (`pd.NA`) no los
    entienden los transformadores de scikit-learn.
    """
    return series.astype("object").where(series.notna(), np.nan)


def _to_bool(series: pd.Series) -> pd.Series:
    """Interpreta valores como `sí`/`no`, `true`/`false`, `1`/`0`."""
    if pdt.is_numeric_dtype(series):
        return series.map(lambda v: bool(v) if pd.notna(v) else np.nan)
    truthy = {"1", "true", "t", "yes", "y", "si", "sí", "verdadero"}
    falsy = {"0", "false", "f", "no", "n", "falso"}
    def convert(value):
        if pd.isna(value):
            return np.nan
        text = str(value).strip().lower()
        if text in truthy:
            return True
        if text in falsy:
            return False
        return np.nan
    return series.map(convert)


def cast_columns(df: pd.DataFrame, casts: dict) -> pd.DataFrame:
    """Aplica `cast_column` a varias columnas del dataset."""
    df = df.copy()
    for name, target_type in casts.items():
        if name in df.columns and target_type:
            df[name] = cast_column(df[name], target_type)
    return df


def date_features(df: pd.DataFrame, columns, parts=("year", "month", "day")) -> pd.DataFrame:
    """Sustituye cada columna de fecha por sus partes numéricas."""
    df = df.copy()
    for name in columns:
        if name not in df.columns:
            continue
        fechas = pd.to_datetime(df[name], errors="coerce")
        for part in parts:
            df[f"{name}_{part}"] = getattr(fechas.dt, part)
        df = df.drop(columns=[name])
    return df


# ---------------------------------------------------------------------------
# Normalización
# ---------------------------------------------------------------------------


def normalize_column(series: pd.Series, method: str) -> pd.Series:
    """Normaliza una columna numérica con el método indicado."""
    if method in (None, "none"):
        return series
    numeric = pd.to_numeric(series, errors="coerce")
    if method == "log":
        return np.log1p(numeric.clip(lower=0))
    if method == "standard":
        return (numeric - numeric.mean()) / _safe_std(numeric)
    if method == "minmax":
        span = numeric.max() - numeric.min()
        return (numeric - numeric.min()) / (span if span else 1.0)
    if method == "robust":
        q75, q25 = numeric.quantile(0.75), numeric.quantile(0.25)
        iqr = q75 - q25
        return (numeric - numeric.median()) / (iqr if iqr else 1.0)
    raise ValueError(f"Método de normalización desconocido: {method}")


def _safe_std(series: pd.Series) -> float:
    std = float(series.std(ddof=0))
    return std if std else 1.0


def normalize_columns(df: pd.DataFrame, columns, method: str) -> pd.DataFrame:
    """Normaliza varias columnas numéricas del dataset."""
    df = df.copy()
    for name in columns or []:
        if name in df.columns and pdt.is_numeric_dtype(df[name]):
            df[name] = normalize_column(df[name], method)
    return df


def normalization_plan(normalizations: dict) -> dict:
    """Agrupa `{columna: método}` por método, descartando los `none`."""
    plan: dict = {}
    for name, method in (normalizations or {}).items():
        if method and method != "none":
            plan.setdefault(method, []).append(name)
    return plan


def apply_transformations(
    df: pd.DataFrame,
    casts: Optional[dict] = None,
    normalizations: Optional[dict] = None,
) -> pd.DataFrame:
    """Aplica cambios de tipo y normalización en un solo paso.

    Es la transformación que se aplica al dataset para mostrarlo y guardarlo,
    y la que replican los pasos `ColumnTyper` y `ColumnNormalizer` del
    pipeline.
    """
    out = df
    if casts:
        out = cast_columns(out, casts)
    for method, columns in normalization_plan(normalizations).items():
        out = normalize_columns(out, columns, method)
    return out


def suggested_normalizations(df, profiles, factor: float = 10.0) -> dict:
    """Propone normalizar las columnas numéricas con una escala muy distinta.

    Se marca una columna cuando su rango es mucho mayor que la mediana de los
    rangos del dataset, que es el síntoma habitual de una variable mal
    escalada que domina modelos lineales, SVM y KNN.
    """
    numeric = [
        p.name
        for p in profiles
        if p.semantic_type == "numerico" and p.name in df.columns
    ]
    if len(numeric) < 2:
        return {}
    spread = {}
    for name in numeric:
        values = pd.to_numeric(df[name], errors="coerce").dropna()
        if values.empty:
            continue
        width = float(values.max() - values.min())
        if np.isfinite(width):
            spread[name] = width
    if len(spread) < 2:
        return {}
    reference = float(np.median(list(spread.values())))
    if reference <= 0:
        return {}
    return {
        name: "standard" for name, width in spread.items() if width > factor * reference
    }


# ---------------------------------------------------------------------------
# Transformadores para el pipeline
# ---------------------------------------------------------------------------


class ColumnTyper(BaseEstimator, TransformerMixin):
    """Paso de pipeline que aplica los cambios de tipo de datos."""

    def __init__(self, casts=None):
        self.casts = casts or {}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        frame = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        return cast_columns(frame, self.casts)


class ColumnNormalizer(BaseEstimator, TransformerMixin):
    """Paso de pipeline que normaliza columnas numéricas.

    Las constantes (media, desviación, mínimo, máximo, cuartiles) se calculan
    en `fit` con los datos de entrenamiento y se reutilizan en `transform`, de
    forma que al predecir con datos nuevos se aplica la misma transformación
    que durante el entrenamiento. Si no se ha llamado a `fit`, se normaliza
    cada lote por separado.
    """

    def __init__(self, normalizations=None):
        self.normalizations = normalizations or {}

    def fit(self, X, y=None):
        frame = _as_frame(X)
        stats = {}
        for method, columns in normalization_plan(self.normalizations).items():
            for column in columns:
                values = _numeric_values(frame, column)
                if values is None or values.empty:
                    continue
                stats[column] = _normalization_stats(method, values)
        self.stats_ = stats
        return self

    def transform(self, X):
        frame = _as_frame(X)
        stats = getattr(self, "stats_", None)
        if not stats:
            return _normalize_with_batch_stats(frame, self.normalizations)
        out = frame.copy()
        for column, entry in stats.items():
            if column not in out.columns:
                continue
            out[column] = _apply_stats(
                pd.to_numeric(out[column], errors="coerce"), entry
            )
        return out


def _as_frame(X) -> pd.DataFrame:
    return X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)


def _numeric_values(frame: pd.DataFrame, column: str):
    if column not in frame.columns:
        return None
    return pd.to_numeric(frame[column], errors="coerce").dropna()


def _normalization_stats(method: str, values: pd.Series) -> dict:
    entry = {"method": method}
    if method == "standard":
        entry["center"] = float(values.mean())
        entry["scale"] = _safe_std(values)
    elif method == "minmax":
        entry["center"] = float(values.min())
        entry["scale"] = float(values.max() - values.min()) or 1.0
    elif method == "robust":
        q75, q25 = values.quantile(0.75), values.quantile(0.25)
        entry["center"] = float(values.median())
        entry["scale"] = float(q75 - q25) or 1.0
    return entry


def _apply_stats(values: pd.Series, entry: dict) -> pd.Series:
    method = entry["method"]
    if method == "log":
        return np.log1p(values.clip(lower=0))
    return (values - entry["center"]) / entry["scale"]


def _normalize_with_batch_stats(frame: pd.DataFrame, normalizations: dict) -> pd.DataFrame:
    out = frame
    for method, columns in normalization_plan(normalizations).items():
        out = normalize_columns(out, columns, method)
    return out