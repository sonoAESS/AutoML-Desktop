# core/persistence.py
"""Guardado y reutilización de un modelo ya construido.

Un *bundle* (extensión `.automl`) contiene en un único archivo:

- el pipeline entrenado;
- los metadatos necesarios para volver a usarlo (tarea, objetivo, columnas
  requeridas con sus tipos, clases, métricas, versiones);
- opcionalmente el dataset ya transformado con el que se entrenó.

Advertencia: el formato usa `joblib`, que al leer ejecuta código Python. Como
ocurre con cualquier fichero `pickle`, carga solo bundles de confianza.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import joblib
import pandas as pd
import sklearn

from core import model_specs, preprocessor, profiling

BUNDLE_VERSION = 1
BUNDLE_SUFFIX = ".automl"
PREDICTION_COLUMN = "prediccion"


@dataclass(frozen=True)
class BundleMetadata:
    """Ficha de un modelo guardado, suficiente para volver a aplicarlo."""

    created_at: str
    app_version: str
    sklearn_version: str
    task_type: str
    target_column: str
    model_name: str
    family: str
    feature_columns: tuple
    feature_dtypes: dict
    classes: tuple = ()
    metrics: dict = field(default_factory=dict)
    best_params: dict = field(default_factory=dict)
    casts: dict = field(default_factory=dict)
    normalizations: dict = field(default_factory=dict)
    n_train_rows: int = 0
    dataset_included: bool = False
    dataset_path: Optional[str] = None
    notes: str = ""

    def as_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        """Resumen legible para la interfaz."""
        lineas = [
            f"Modelo: {self.model_name} ({model_specs.family_label(self.family)})",
            f"Tarea: {self.task_type}",
            f"Objetivo: {self.target_column}",
            f"Creado: {self.created_at}",
            f"Filas de entrenamiento: {self.n_train_rows}",
            f"Columnas requeridas: {len(self.feature_columns)}",
        ]
        if self.metrics:
            metricas = ", ".join(
                f"{k}={v:.4f}" if isinstance(v, (int, float)) else f"{k}={v}"
                for k, v in self.metrics.items()
                if k != "best_params"
            )
            lineas.append(f"Métricas: {metricas}")
        if self.dataset_included:
            lineas.append("Incluye el dataset transformado")
        return "\n".join(lineas)


@dataclass(frozen=True)
class Bundle:
    """Bundle cargado desde disco."""

    pipeline: Any
    metadata: BundleMetadata
    dataset: Optional[pd.DataFrame] = None


@dataclass(frozen=True)
class SchemaReport:
    """Resultado de comprobar un CSV nuevo contra el esquema del modelo."""

    missing: tuple = ()
    unexpected: tuple = ()
    n_rows: int = 0
    n_converted: int = 0

    @property
    def ok(self) -> bool:
        return not self.missing

    def describe(self) -> str:
        lineas = []
        if self.missing:
            lineas.append(
                "Faltan columnas obligatorias: " + ", ".join(self.missing)
            )
        if self.unexpected:
            lineas.append(
                "Columnas ignoradas (no las usa el modelo): "
                + ", ".join(self.unexpected)
            )
        if self.n_converted:
            lineas.append(
                f"{self.n_converted} columna(s) convertidas al tipo con el que "
                "se entrenó"
            )
        if self.ok and not lineas:
            lineas.append("Las columnas coinciden con el esquema del modelo.")
        return "\n".join(lineas)


def build_metadata(
    pipeline,
    target_column: str,
    task_type: str,
    model_name: str,
    metrics: Optional[dict] = None,
    df: Optional[pd.DataFrame] = None,
    casts: Optional[dict] = None,
    normalizations: Optional[dict] = None,
    app_version: str = "1.0",
    dataset_path: Optional[str] = None,
    notes: str = "",
) -> BundleMetadata:
    """Construye los metadatos de un modelo a partir del pipeline entrenado."""
    steps = getattr(pipeline, "named_steps", {})
    preprocessor_step = steps.get("preprocessor")
    feature_columns = tuple(preprocessor_step.feature_names_in_) if (
        hasattr(preprocessor_step, "feature_names_in_")
    ) else ()

    model = steps.get("model")
    clases = getattr(model, "classes_", None)
    classes = tuple(clases.tolist()) if clases is not None else ()

    metrics = dict(metrics or {})
    best_params = metrics.get("best_params") or {}
    if not isinstance(best_params, dict):
        best_params = {}

    feature_dtypes = {}
    if df is not None:
        feature_dtypes = {
            column: str(df[column].dtype)
            for column in feature_columns
            if column in df.columns
        }

    return BundleMetadata(
        created_at=datetime.now(timezone.utc).astimezone().strftime(
            "%Y-%m-%d %H:%M"
        ),
        app_version=app_version,
        sklearn_version=sklearn.__version__,
        task_type=task_type,
        target_column=target_column,
        model_name=model_name,
        family=_family_of(model_name, task_type),
        feature_columns=feature_columns,
        feature_dtypes=feature_dtypes,
        classes=classes,
        metrics={k: v for k, v in metrics.items() if k != "best_params"},
        best_params=best_params,
        casts=dict(casts or {}),
        normalizations=dict(normalizations or {}),
        n_train_rows=int(len(df)) if df is not None else 0,
        dataset_included=False,
        dataset_path=dataset_path,
        notes=notes,
    )


def _family_of(model_name: str, task_type: str) -> str:
    try:
        return model_specs.get_spec(model_name, task_type).family
    except KeyError:
        return "desconocida"


def save_bundle(
    path,
    pipeline,
    metadata: BundleMetadata,
    dataset: Optional[pd.DataFrame] = None,
) -> str:
    """Guarda pipeline, metadatos y dataset en un único archivo `.automl`."""
    path = Path(path)
    if path.suffix == "":
        path = path.with_suffix(BUNDLE_SUFFIX)
    if dataset is not None:
        metadata = BundleMetadata(
            **{**metadata.as_dict(), "dataset_included": True}
        )
    payload = {
        "bundle_version": BUNDLE_VERSION,
        "pipeline": pipeline,
        "metadata": metadata,
        "dataset": dataset,
    }
    joblib.dump(payload, path, compress=3)
    return str(path)


def load_bundle(path) -> Bundle:
    """Carga un bundle `.automl`.

    Lanza `ValueError` si el archivo no es un bundle válido o si su versión no
    es compatible con esta versión de la aplicación.
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(f"No existe el archivo: {path}")
    payload = joblib.load(path)
    if not isinstance(payload, dict) or "pipeline" not in payload:
        raise ValueError("El archivo no es un modelo de AutoML Desktop.")
    version = payload.get("bundle_version", 0)
    if version != BUNDLE_VERSION:
        raise ValueError(
            f"Versión de modelo {version} no compatible con esta aplicación "
            f"(se espera la {BUNDLE_VERSION})."
        )
    return Bundle(
        pipeline=payload["pipeline"],
        metadata=payload["metadata"],
        dataset=payload.get("dataset"),
    )


def check_schema(df: pd.DataFrame, metadata: BundleMetadata) -> SchemaReport:
    """Compara las columnas de un CSV con las que el modelo espera."""
    requeridas = list(metadata.feature_columns)
    if metadata.target_column in df.columns and metadata.target_column not in requeridas:
        requeridas = requeridas + [metadata.target_column]

    missing = [column for column in requeridas if column not in df.columns]
    unexpected = [column for column in df.columns if column not in requeridas]
    return SchemaReport(
        missing=tuple(missing),
        unexpected=tuple(unexpected),
        n_rows=int(len(df)),
    )


def align_features(
    df: pd.DataFrame,
    metadata: BundleMetadata,
    strict: bool = True,
) -> tuple:
    """Prepara un CSV nuevo para pasarlo por el pipeline.

    Selecciona y ordena las columnas requeridas y ajusta sus tipos a los
    usados en el entrenamiento. Los cambios de tipo y la normalización los
    aplica después el propio pipeline, con las constantes ajustadas en
    `fit`; no se repiten aquí para no aplicarlos dos veces.

    Parámetros
    ----------
    strict : bool
        Si es `True`, falla cuando falta alguna columna obligatoria.

    Devuelve
    --------
    (X, y, SchemaReport)
    """
    report = check_schema(df, metadata)
    if report.missing and strict:
        raise ValueError(
            "Faltan columnas obligatorias en el archivo: "
            + ", ".join(report.missing)
        )

    frame = df.copy()
    faltantes = set(report.missing)
    convertible = 0
    for column, dtype in metadata.feature_dtypes.items():
        # Las columnas ausentes se rellenan después con nulos: no se
        # convierten aquí, porque pasarían a ser cadenas "nan".
        if column not in frame.columns or column in faltantes:
            continue
        actual = str(frame[column].dtype)
        if actual != dtype:
            frame[column] = _coerce_dtype(frame[column], dtype)
            convertible += 1

    y = frame[metadata.target_column] if metadata.target_column in frame.columns else None
    X = frame.drop(columns=[metadata.target_column], errors="ignore")
    X = X[[column for column in metadata.feature_columns if column in X.columns]]
    X = X.reindex(columns=list(metadata.feature_columns))

    return X, y, SchemaReport(
        missing=report.missing,
        unexpected=report.unexpected,
        n_rows=report.n_rows,
        n_converted=convertible,
    )


def _coerce_dtype(series: pd.Series, dtype: str) -> pd.Series:
    """Convierte una columna al dtype con el que se entrenó."""
    try:
        if dtype.startswith("datetime"):
            return pd.to_datetime(series, errors="coerce")
        if dtype in ("float64", "float32"):
            return pd.to_numeric(series, errors="coerce").astype("float64")
        if dtype in ("int64", "int32", "Int64"):
            return pd.to_numeric(series, errors="coerce").astype("Int64")
        if dtype == "bool":
            return preprocessor.cast_column(series, "booleano")
        if dtype.startswith("string") or dtype in ("str", "object"):
            return preprocessor.cast_column(series, "texto")
        return series
    except (TypeError, ValueError):
        return series


def predict(bundle: Bundle, df: pd.DataFrame, strict: bool = True):
    """Aplica un modelo guardado a un dataset nuevo.

    Devuelve
    --------
    (predicciones, report, X, y)
        `predicciones` es un `DataFrame` con la columna `prediccion` y, si el
        modelo da probabilidades, una columna por clase.
    """
    X, y, report = align_features(df, bundle.metadata, strict=strict)
    modelo = bundle.pipeline
    predicciones = modelo.predict(X)

    salida = pd.DataFrame({PREDICTION_COLUMN: predicciones}, index=X.index)
    if hasattr(modelo, "predict_proba"):
        try:
            probas = modelo.predict_proba(X)
            clases = bundle.metadata.classes or getattr(modelo, "classes_", ())
            for indice, clase in enumerate(clases):
                salida[f"prob_{clase}"] = probas[:, indice]
        except (AttributeError, ValueError, IndexError):
            pass
    return salida, report, X, y


def profile_input(df: pd.DataFrame, metadata: BundleMetadata) -> list:
    """Perfila las columnas de entrada para detectar errores de tipo."""
    columnas = list(metadata.feature_columns) + [metadata.target_column]
    return [
        profile
        for profile in profiling.profile_dataframe(df)
        if profile.name in columnas
    ]