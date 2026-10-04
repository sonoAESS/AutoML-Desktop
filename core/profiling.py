# core/profiling.py
"""Perfilado de columnas: qué tipo de dato es cada una y para qué sirve.

Este módulo es el que permite que la interfaz sea "inteligente": a partir de
un `DataFrame` decide qué tipo de tarea tiene sentido, qué columnas pueden
usarse como variables predictoras y cuáles son candidatas a objetivo.

No contiene lógica de interfaz ni de Qt.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import pandas as pd
from pandas.api import types as pdt

CLASSIFICATION = "classification"
REGRESSION = "regression"

#: Número máximo de clases distintas que se consideran "clasificación".
MAX_CLASSES = 20
#: A partir de estos valores únicos, una columna de texto se considera texto.
TEXT_MIN_UNIQUE = 50
#: Longitud media (en caracteres) a partir de la cual se considera texto libre.
TEXT_MIN_LENGTH = 15
#: Proporción de valores únicos que delata texto libre o identificadores.
TEXT_MIN_RATIO = 0.5
#: Filas mínimas para considerar que una columna numérica es un identificador.
IDENTIFIER_MIN_ROWS = 20
#: Porcentaje de nulos a partir del cual se avisa al usuario.
MISSING_WARN_RATIO = 0.3
#: Fracción de valores convertibles necesaria para poder treating una columna
#: de texto como numérica.
CONVERTIBLE_RATIO = 0.9

SEMANTIC_TYPES = (
    "numerico",
    "booleano",
    "categorico",
    "fecha",
    "texto",
    "identificador",
)

SEMANTIC_LABELS = {
    "numerico": "Numérico",
    "booleano": "Booleano",
    "categorico": "Categórico",
    "fecha": "Fecha",
    "texto": "Texto",
    "identificador": "Identificador",
}

BLOCKING_TYPES = ("texto", "identificador")


@dataclass(frozen=True)
class ColumnProfile:
    """Ficha de una columna del dataset."""

    name: str
    dtype: str
    semantic_type: str
    n_unique: int
    n_rows: int
    n_missing: int
    convertible_to_numeric: bool = False
    discrete: bool = False
    target_task: Optional[str] = None
    warnings: tuple = field(default_factory=tuple)

    @property
    def label(self) -> str:
        return SEMANTIC_LABELS.get(self.semantic_type, self.semantic_type)

    @property
    def missing_pct(self) -> float:
        if not self.n_rows:
            return 0.0
        return self.n_missing / self.n_rows

    @property
    def usable_as_feature(self) -> bool:
        """False si la columna no puede entrar en el pipeline tal cual."""
        return self.semantic_type not in BLOCKING_TYPES

    @property
    def usable_as_target(self) -> bool:
        return self.target_task is not None

    @property
    def is_numeric_like(self) -> bool:
        return self.semantic_type in ("numerico", "booleano")


def _numeric_ratio(series: pd.Series) -> float:
    """Fracción de valores no nulos que `pandas` consegue convertir a número."""
    values = series.dropna()
    if values.empty:
        return 0.0
    converted = pd.to_numeric(values, errors="coerce")
    return float(converted.notna().sum() / len(values))


def _is_integer_like(series: pd.Series) -> bool:
    values = series.dropna()
    if values.empty:
        return False
    try:
        return bool((values.astype("int64") == values).all())
    except (TypeError, ValueError):
        return False


def infer_semantic_type(series: pd.Series) -> str:
    """Deduce el tipo semántico de una columna.

    Devuelve uno de `SEMANTIC_TYPES`.
    """
    n_rows = int(len(series))
    n_unique = int(series.nunique(dropna=True))

    if pdt.is_datetime64_any_dtype(series):
        return "fecha"
    if pdt.is_bool_dtype(series):
        return "booleano"

    if pdt.is_numeric_dtype(series):
        if (
            n_rows >= IDENTIFIER_MIN_ROWS
            and n_unique == n_rows
            and _is_integer_like(series)
        ):
            return "identificador"
        return "numerico"

    if _looks_like_text(series) and n_unique > MAX_CLASSES:
        return "texto"
    if n_rows and n_unique == n_rows and n_rows >= IDENTIFIER_MIN_ROWS:
        return "identificador"
    if n_unique >= TEXT_MIN_UNIQUE:
        return "texto"
    if (
        n_rows
        and n_unique / n_rows >= TEXT_MIN_RATIO
        and n_unique > MAX_CLASSES
        and _numeric_ratio(series) < CONVERTIBLE_RATIO
    ):
        return "texto"
    return "categorico"


def _looks_like_text(series: pd.Series) -> bool:
    """True si los valores son cadenas largas tipo frase o comentario."""
    values = series.dropna().astype("string").head(200)
    if values.empty:
        return False
    return float(values.str.len().mean()) >= TEXT_MIN_LENGTH


def suggest_task_type(series: pd.Series) -> Optional[str]:
    """Tarea de ML sugerida para una posible variable objetivo.

    Devuelve `"classification"`, `"regression"` o `None` si la columna no
    sirve como objetivo (texto libre, identificadores o fechas).
    """
    semantic = infer_semantic_type(series)

    if semantic in BLOCKING_TYPES or semantic == "fecha":
        return None
    if semantic in ("booleano", "categorico"):
        return CLASSIFICATION
    if pdt.is_numeric_dtype(series):
        n_unique = int(series.nunique(dropna=True))
        if _is_integer_like(series) and n_unique <= MAX_CLASSES:
            return CLASSIFICATION
        return REGRESSION
    return CLASSIFICATION


def profile_column(series: pd.Series) -> ColumnProfile:
    """Construye la `ColumnProfile` de una columna del dataset."""
    semantic = infer_semantic_type(series)
    n_rows = int(len(series))
    n_missing = int(series.isna().sum())
    n_unique = int(series.nunique(dropna=True))
    ratio = _numeric_ratio(series)

    convertible = (
        not pdt.is_numeric_dtype(series)
        and semantic != "fecha"
        and ratio >= CONVERTIBLE_RATIO
    )
    discrete = bool(
        pdt.is_numeric_dtype(series)
        and not pdt.is_bool_dtype(series)
        and _is_integer_like(series)
        and n_unique <= MAX_CLASSES
    )

    profile = ColumnProfile(
        name=str(series.name),
        dtype=str(series.dtype),
        semantic_type=semantic,
        n_unique=n_unique,
        n_rows=n_rows,
        n_missing=n_missing,
        convertible_to_numeric=convertible,
        discrete=discrete,
    )
    return replace(
        profile,
        target_task=suggest_task_type(series),
        warnings=_column_warnings(profile, ratio),
    )


def _column_warnings(profile: ColumnProfile, numeric_ratio: float) -> tuple:
    """Avisos legibles para el usuario sobre una columna."""
    warnings = []
    if profile.missing_pct >= MISSING_WARN_RATIO:
        warnings.append(f"muchos nulos ({profile.missing_pct:.0%})")
    if profile.semantic_type == "identificador":
        warnings.append("posible identificador: normalmente conviene eliminarla")
    if profile.semantic_type == "texto":
        warnings.append("texto libre: requiere vectorización")
    if profile.semantic_type == "categorico" and profile.n_unique > MAX_CLASSES:
        warnings.append(f"muchas categorías ({profile.n_unique})")
    if profile.convertible_to_numeric:
        warnings.append(
            f"contiene números como texto ({numeric_ratio:.0%}): se puede convertir"
        )
    return tuple(warnings)


def profile_dataframe(df: pd.DataFrame) -> list:
    """Perfila todas las columnas de un `DataFrame`."""
    return [profile_column(df[col]) for col in df.columns]


def profile_by_name(df: pd.DataFrame) -> dict:
    """Igual que `profile_dataframe` pero indexado por nombre de columna."""
    return {profile.name: profile for profile in profile_dataframe(df)}


def suggest_target(profiles: list) -> Optional[ColumnProfile]:
    """Propone la columna objetivo más plausible.

    Se descartan identificadores, texto libre y fechas; entre el resto se
    prefiere la de menor cardinalidad, que suele ser la clase a predecir.
    """
    candidates = [
        profile
        for profile in profiles
        if profile.usable_as_target
        and profile.missing_pct < 1
        and profile.n_unique > 1
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda profile: profile.n_unique)


def dataset_warnings(profiles: list) -> list:
    """Resumen de los problemas que bloquean o dificultan el modelado."""
    messages = []
    blocking = [p for p in profiles if not p.usable_as_feature]
    if blocking:
        messages.append(
            "Columnas no utilizables como variables predictoras: "
            + ", ".join(f"{p.name} ({p.label})" for p in blocking)
            + ". Conviértelas o elimínalas."
        )
    if not any(p.usable_as_target for p in profiles):
        messages.append(
            "No se detecta ninguna columna válida como variable objetivo."
        )
    n_const = [p for p in profiles if p.n_unique <= 1]
    if n_const:
        messages.append(
            "Columnas constantes (sin información): "
            + ", ".join(p.name for p in n_const)
            + "."
        )
    return messages