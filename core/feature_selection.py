"""Selección de atributos: catálogo, selectores y lectura de la selección.

Solo lógica pura, sin Qt: recibe un `DataFrame` o una matriz y devuelve un
selector de scikit-learn sin ajustar, o la tabla de lo que se seleccionó.

El selector se coloca **dentro** del pipeline, después del `ColumnTransformer`
y del balanceo, para que nunca vea el conjunto de prueba.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import (
    SelectFromModel,
    SelectKBest,
    SelectPercentile,
    chi2,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)

from core.model_specs import CLASSIFICATION, REGRESSION

#: Columnas de la tabla que devuelve `describe_selection`.
TABLA_SELECCION = (
    "atributo",
    "n_columnas_codificadas",
    "puntuacion",
    "seleccionado",
)


@dataclass(frozen=True)
class SelectionMethod:
    """Un método de selección de atributos del catálogo.

    Attributes:
        id: Identificador corto (`"chi2"`, `"anova"`, …).
        label: Nombre para la interfaz, en español.
        supports_regression: Si el método sirve para tareas de regresión.
        requires_non_negative: Si necesita que la matriz no tenga negativos.
        supports_percentile: Si admite cortar por porcentaje en vez de por `k`.
        default_k: Número de atributos por defecto.
        docstring: Explicación breve de cuándo conviene usarlo.
    """

    id: str
    label: str
    supports_regression: bool
    requires_non_negative: bool
    supports_percentile: bool
    default_k: Optional[int]
    docstring: str


SELECTION_METHODS: Dict[str, SelectionMethod] = {
    "chi2": SelectionMethod(
        id="chi2",
        label="Chi-cuadrado",
        supports_regression=False,
        requires_non_negative=True,
        supports_percentile=True,
        default_k=10,
        docstring=(
            "Compara cada atributo con la clase. Exige valores no negativos, "
            "por eso el pipeline añade un MinMaxScaler antes."
        ),
    ),
    "anova": SelectionMethod(
        id="anova",
        label="ANOVA F",
        supports_regression=True,
        requires_non_negative=False,
        supports_percentile=True,
        default_k=10,
        docstring=(
            "Test F: rápido y con supuestos asumidos. Es el punto de partida "
            "para datos limpios."
        ),
    ),
    "mutual_info": SelectionMethod(
        id="mutual_info",
        label="Información mutua",
        supports_regression=True,
        requires_non_negative=False,
        supports_percentile=True,
        default_k=10,
        docstring=(
            "Detecta relaciones no lineales. Es el más lento de los tres "
            "estadísticos."
        ),
    ),
    "embedded": SelectionMethod(
        id="embedded",
        label="Embebido (bosque aleatorio)",
        supports_regression=True,
        requires_non_negative=False,
        supports_percentile=False,
        default_k=10,
        docstring=(
            "Deja que un bosque aleatorio decida qué atributos usar. Slow y "
            "caro, pero suele ganar en exactitud."
        ),
    ),
}


def available_methods(task_type: str) -> List[str]:
    """Métodos que admiten la tarea, en el orden del catálogo."""
    return [
        metodo
        for metodo, spec in SELECTION_METHODS.items()
        if task_type != REGRESSION or spec.supports_regression
    ]


def method_label(method: str) -> str:
    """Nombre del método para la interfaz, o el `id` si no está en el catálogo."""
    spec = SELECTION_METHODS.get(method)
    return spec.label if spec else str(method)


# ----------------------------------------------------------------------
# Selectores seguros
# ----------------------------------------------------------------------
# `SelectKBest` y `SelectFromModel` avisan con `warnings` cuando se piden más
# atributos de los que hay y dejan el recorte a sklearn: el aviso va en inglés
# y el resultado cambia según la versión. Estas clases lo hacen siempre igual.
# Además tienen que ser importables por nombre: el pipeline se serializa con
# joblib dentro del bundle.


class SafeSelectKBest(SelectKBest):
    """`SelectKBest` que recorta `k` al número de columnas disponibles."""

    def fit(self, X, y):
        self.k_effective_ = int(min(self.k, np.asarray(X).shape[1]))
        original = self.k
        self.k = self.k_effective_
        try:
            super().fit(X, y)
        finally:
            self.k = original
        return self


class SafeSelectFromModel(SelectFromModel):
    """`SelectFromModel` que se queda con las `k` mejores y recorta `max_features`.

    No se apoya en `max_features` para elegir el corte: según la versión de
    scikit-learn ese parámetro pasó de "las k mejores" a un umbral sobre la
    media, así que el número de atributos acababa siendo otro. Aquí la máscara
    se calcula con las importancias del estimador, que es lo que el método
    embebido promete y lo que no depende de la versión instalada.
    """

    def fit(self, X, y, **params):
        self.max_features_effective_ = int(
            min(self.max_features, np.asarray(X).shape[1])
        )
        original = self.max_features
        self.max_features = self.max_features_effective_
        try:
            super().fit(X, y, **params)
        finally:
            self.max_features = original

        importancias = getattr(self.estimator_, "feature_importances_", None)
        if importancias is not None:
            importancias = np.asarray(importancias, dtype=float)
            orden = np.argsort(importancias, kind="stable")[::-1]
            mascara = np.zeros(len(importancias), dtype=bool)
            mascara[orden[: self.max_features_effective_]] = True
            self._safe_mask = mascara
        else:
            self._safe_mask = super()._get_support_mask()

        self.n_features_ = int(self._safe_mask.sum())
        return self

    def get_support(self, indices: bool = False):
        mascara = getattr(self, "_safe_mask", None)
        if mascara is None:
            return super().get_support(indices=indices)
        return np.where(mascara)[0] if indices else mascara


# ----------------------------------------------------------------------
def _score_func(method: str, task_type: str):
    """Función de puntuación según método y tarea.

    La información mutua no es determinista, así que se fija `random_state`
    con `partial` en vez de dejar que sklearn sortee.
    """
    regresion = task_type == REGRESSION
    if method == "chi2":
        return chi2
    if method == "anova":
        return f_regression if regresion else f_classif
    if method == "mutual_info":
        if regresion:
            return partial(mutual_info_regression, random_state=42)
        return partial(mutual_info_classif, random_state=42)
    raise ValueError(f"El método {method!r} no usa función de puntuación.")


def _random_forest(task_type: str, random_state: int, n_estimators: int = 100):
    """Bosque aleatorio para el método embebido."""
    if task_type == REGRESSION:
        return RandomForestRegressor(
            n_estimators=n_estimators, random_state=random_state
        )
    return RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)


def build_selector(
    method: str,
    task_type: str,
    k: Optional[int] = None,
    percentile: Optional[float] = None,
    random_state: int = 42,
    n_estimators: int = 100,
) -> BaseEstimator:
    """Selector sin ajustar para el método y la tarea indicados.

    Args:
        method: Identificador del catálogo.
        task_type: `"classification"` o `"regression"`.
        k: Número de atributos a conservar; si es `None`, el del catálogo.
        percentile: Porcentaje de atributos a conservar, alternativa a `k`.
        random_state: Semilla para todo lo que tenga aleatoriedad.
        n_estimators: Árboles del bosque en el método `embedded`.

    Returns:
        `SelectPercentile`, `SafeSelectFromModel` o `SafeSelectKBest`.

    Raises:
        ValueError: Método desconocido, incompatible con la tarea o porcentaje
            no soportado.
    """
    spec = SELECTION_METHODS.get(method)
    if spec is None:
        raise ValueError(f"Método de selección desconocido: {method!r}.")
    if task_type == REGRESSION and not spec.supports_regression:
        raise ValueError(f"El método {spec.label} no se puede usar en regresión.")

    if percentile is not None:
        if not spec.supports_percentile:
            raise ValueError(f"{spec.label} no admite el criterio por porcentaje.")
        return SelectPercentile(
            score_func=_score_func(method, task_type), percentile=percentile
        )

    k_efectivo = int(k) if k else spec.default_k
    if method == "embedded":
        return SafeSelectFromModel(
            estimator=_random_forest(task_type, random_state, n_estimators),
            max_features=k_efectivo,
            threshold=None,
        )
    return SafeSelectKBest(score_func=_score_func(method, task_type), k=k_efectivo)


# ----------------------------------------------------------------------
def _last_encoder(transform) -> Any:
    """El `OneHotEncoder` final de un transformer del `ColumnTransformer`.

    La rama del `ColumnTransformer` puede ser el codificador directamente o
    una `Pipeline` que acabe en él, según cómo la montara el preprocesador.
    """
    candidatos = [paso[1] for paso in getattr(transform, "steps", []) or []]
    candidatos.append(transform)
    for candidato in candidatos:
        if hasattr(candidato, "categories_"):
            return candidato
    return None


def feature_origin_map(preprocessor, names: List[str]) -> List[str]:
    """Nombre del atributo original de cada columna transformada.

    `get_feature_names_out()` produce `cat__ciudad_Madrid`, de donde solo se
    puede quitar el prefijo de rama (`cat__`), no el valor codificado. El
    reparto real se reconstruye desde el `ColumnTransformer` ya ajustado, que
    es donde está la información. El recorrido es posicional y coincide con el
    orden de `get_feature_names_out()`.

    Args:
        preprocessor: `ColumnTransformer` ya ajustado.
        names: Nombres de las columnas transformadas.

    Returns:
        Una etiqueta original por columna transformada.
    """
    origins: List[str] = []
    for _name, transform, cols in getattr(preprocessor, "transformers_", []):
        cols = [cols] if isinstance(cols, str) else list(cols)
        encoder = _last_encoder(transform)
        if encoder is None:
            # Una salida por columna de entrada.
            origins.extend(cols)
        else:
            # Una salida por categoría de cada columna.
            for col, categorias in zip(cols, encoder.categories_):
                origins.extend([col] * len(categorias))

    if len(origins) != len(names):
        # `drop` o `handle_unknown` raro descuadran el recuento: se degrada la
        # etiqueta quitando el prefijo de rama antes que fallar.
        origins = [name.split("__", 1)[-1] for name in names]
    return origins


def _scores_of(selector) -> np.ndarray:
    """Puntuación de cada columna: `scores_` o importancias del estimador."""
    scores = getattr(selector, "scores_", None)
    if scores is None:
        estimator = getattr(selector, "estimator_", None)
        scores = getattr(estimator, "feature_importances_", None)
    if scores is None:
        return np.full(len(getattr(selector, "support_", [])), np.nan)
    return np.asarray(scores, dtype=float)


def describe_selection(pipe, columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Qué atributos sobrevivieron a la selección y con qué puntuación.

    Las columnas codificadas de un mismo atributo (las de un one-hot) se
    agregan bajo el nombre original: `seleccionado` es cierto si se quedó
    alguna, y la puntuación es la mejor de las suyas.

    Args:
        pipe: Pipeline ajustado; solo se mira su paso `selector`.
        columns: Nombres transformados; si es `None` se piden al preprocessor.

    Returns:
        `DataFrame` con las columnas de `TABLA_SELECCION`, ordenado por
        puntuación descendente. Si no hay selector, un `DataFrame` vacío con
        esas mismas columnas, para que la UI pueda llamarlo siempre.
    """
    vacio = pd.DataFrame(columns=list(TABLA_SELECCION))
    steps = getattr(pipe, "named_steps", {})
    selector = steps.get("selector")
    if selector is None:
        return vacio

    preprocessor = steps.get("preprocessor")
    if columns is None:
        if preprocessor is None:
            return vacio
        columns = list(preprocessor.get_feature_names_out())

    soporte = np.asarray(selector.get_support(), dtype=bool)
    scores = _scores_of(selector)
    if len(scores) != len(columns):
        scores = np.full(len(columns), np.nan)

    if preprocessor is not None:
        origins = feature_origin_map(preprocessor, columns)
    else:
        origins = [name.split("__", 1)[-1] for name in columns]
    if len(origins) != len(columns):
        origins = [name.split("__", 1)[-1] for name in columns]

    filas: List[Dict[str, Any]] = []
    for origen, marca, puntuacion in zip(origins, soporte, scores):
        fila = next((f for f in filas if f["atributo"] == origen), None)
        if fila is None:
            fila = {
                "atributo": origen,
                "n_columnas_codificadas": 0,
                "puntuacion": -np.inf,
                "seleccionado": False,
            }
            filas.append(fila)
        fila["n_columnas_codificadas"] += 1
        fila["puntuacion"] = max(fila["puntuacion"], puntuacion)
        fila["seleccionado"] = fila["seleccionado"] or bool(marca)

    tabla = pd.DataFrame(filas, columns=list(TABLA_SELECCION))
    tabla = tabla.sort_values("puntuacion", ascending=False, kind="stable")
    return tabla.reset_index(drop=True)


def selected_feature_names(pipe) -> List[str]:
    """Nombres transformados que el selector dejó pasar."""
    steps = getattr(pipe, "named_steps", {})
    selector = steps.get("selector")
    preprocessor = steps.get("preprocessor")
    if selector is None or preprocessor is None:
        return []
    columns = list(preprocessor.get_feature_names_out())
    soporte = np.asarray(selector.get_support(), dtype=bool)
    return [name for name, marca in zip(columns, soporte) if marca]
