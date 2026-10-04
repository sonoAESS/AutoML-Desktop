# core/balancing.py
"""Análisis de la distribución de clases y estrategias de balanceo.

La lógica vive aquí y no en la interfaz: la UI solo elige una estrategia y
muestra sus números. Nada de esto depende de Qt.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

#: (umbral de índice de desbalance, etiqueta). El desbalance es
#: `n_mayor / n_minor`: 1.0 significa que las clases están equilibradas.
IR_NIVELES = ((1.5, "equilibrado"), (3.0, "moderado"))
IR_SEVERO = "severo"


@dataclass(frozen=True)
class ClassCount:
    """Número de instancias de una clase."""

    clase: Any
    n: int
    pct: float


@dataclass(frozen=True)
class ClassDistribution:
    """Resumen de la distribución de clases de una variable objetivo."""

    rows: tuple
    total: int
    n_classes: int
    majority: Any = None
    minority: Any = None
    majority_n: int = 0
    minority_n: int = 0
    imbalance_ratio: float = 1.0
    shannon_entropy: float = 0.0
    normalized_entropy: float = 0.0
    level: str = IR_NIVELES[0][1]

    def as_dict(self) -> dict:
        return asdict(self)

    def describe(self) -> str:
        """Resumen en una línea, para la interfaz."""
        if self.n_classes <= 1:
            return (
                f"{self.total} instancias · una sola clase "
                f"({self.majority}): no hay desbalance que corregir."
            )
        return (
            f"{self.n_classes} clases · {self.total} instancias · "
            f"IR = {self.imbalance_ratio:.2f} (n_mayor/n_minor) · "
            f"desbalance {self.level} · "
            f"entropía normalizada {self.normalized_entropy:.2f}"
        )

    @property
    def is_imbalanced(self) -> bool:
        return self.imbalance_ratio >= IR_NIVELES[0][0]


@dataclass(frozen=True)
class BalancingStrategy:
    """Estrategia de balanceo disponible para una tarea."""

    id: str
    label: str
    description: str
    needs_resampling: bool = False
    requires_smote: bool = False


BALANCING_STRATEGIES = {
    "none": BalancingStrategy(
        id="none",
        label="Sin balancear",
        description="Se conserva la distribución original de las clases.",
    ),
    "class_weight": BalancingStrategy(
        id="class_weight",
        label="Pesos de clase",
        description=(
            "El modelo recibe pesos inversos a la frecuencia de cada clase. "
            "No remuestrea: es la opción más rápida y la más predecible."
        ),
    ),
    "random_under": BalancingStrategy(
        id="random_under",
        label="Submuestreo aleatorio",
        description=("Baja las clases mayoritarias hasta el tamaño de la minoritaria."),
        needs_resampling=True,
    ),
    "random_over": BalancingStrategy(
        id="random_over",
        label="Sobremuestreo aleatorio",
        description=(
            "Repite ejemplos de las clases minoritarias hasta igualar a la "
            "mayoritaria."
        ),
        needs_resampling=True,
    ),
    "smote": BalancingStrategy(
        id="smote",
        label="SMOTE",
        description=(
            "Crea ejemplos sintéticos entre los vecinos de cada clase " "minoritaria."
        ),
        needs_resampling=True,
        requires_smote=True,
    ),
    "smoten": BalancingStrategy(
        id="smoten",
        label="SMOTE-NC",
        description=(
            "SMOTE para variables mixtas: respeta las columnas categóricas al "
            "interpolar."
        ),
        needs_resampling=True,
        requires_smote=True,
    ),
}

#: Orden en que se muestran en el desplegable de la interfaz.
STRATEGY_ORDER = (
    "none",
    "class_weight",
    "random_under",
    "random_over",
    "smote",
    "smoten",
)


def class_distribution(y) -> ClassDistribution:
    """Calcula la distribución de clases de una variable objetivo.

    Las filas van ordenadas por número de instancias descendente y, a igualdad,
    por el texto de la clase, para que la tabla sea estable entre refrescos.

    Args:
        y: serie, array o lista con los valores de la clase.

    Returns:
        Un `ClassDistribution` con el resumen completo.
    """
    valores = pd.Series(y).dropna()
    if valores.empty:
        return ClassDistribution(rows=(), total=0, n_classes=0, imbalance_ratio=1.0)

    conteos = valores.value_counts()
    orden = np.lexsort(
        (np.array([str(clase) for clase in conteos.index]), -conteos.values)
    )
    clases = conteos.index[orden]
    conteos = conteos.values[orden]
    total = int(conteos.sum())

    rows = tuple(
        ClassCount(clase=clase, n=int(n), pct=float(n) / total)
        for clase, n in zip(clases, conteos)
    )
    n_classes = int(len(clases))
    majority_n = int(conteos[0])
    minority_n = int(conteos[-1])
    ratio = float(majority_n / minority_n) if minority_n else 1.0

    entropia = 0.0
    if n_classes > 1:
        probabilidades = conteos / total
        entropia = float(-np.sum(probabilidades * np.log2(probabilidades)))
    normalizada = entropia / math.log2(n_classes) if n_classes > 1 else 0.0

    return ClassDistribution(
        rows=rows,
        total=total,
        n_classes=n_classes,
        majority=clases[0],
        minority=clases[-1],
        majority_n=majority_n,
        minority_n=minority_n,
        imbalance_ratio=ratio,
        shannon_entropy=entropia,
        normalized_entropy=normalizada,
        level=nivel_de_desbalance(ratio),
    )


def nivel_de_desbalance(ratio: float) -> str:
    """Etiqueta del nivel de desbalance según el índice de desbalance."""
    for umbral, etiqueta in IR_NIVELES:
        if ratio < umbral:
            return etiqueta
    return IR_SEVERO


def available_strategies(task_type: str, model_name: Optional[str] = None) -> list:
    """Estrategias disponibles para una tarea y, si se indica, un modelo.

    En regresión solo tiene sentido `none`. En clasificación se quita
    `class_weight` si el modelo elegido no admite ese parámetro.
    """
    from core import model_specs

    if task_type != model_specs.CLASSIFICATION:
        return ["none"]

    estrategias = list(STRATEGY_ORDER)
    if model_name:
        try:
            spec = model_specs.get_spec(model_name, task_type)
        except KeyError:
            return estrategias
        if spec.class_weight_path is None:
            estrategias.remove("class_weight")
    return estrategias


class BalancedSampler(BaseEstimator):
    """Remuestrea el conjunto de **entrenamiento** dentro del pipeline.

    Va después del preprocesador y antes del modelo. `fit` no remuestrea (solo
    memoriza tamaños y clases); el remuestreo ocurre en `fit_transform`, que es
    lo que ejecuta `Pipeline.fit`. En `predict` el paso se comporta como la
    identidad, de modo que el esquema del dataset no cambia.

    El remuestreo se hace por índices para que funcione igual con `ndarray`,
    `DataFrame` y matriz dispersa, que es lo que produce el `ColumnTransformer`
    cuando hay variables categóricas.
    """

    def __init__(
        self,
        method: str = "random_under",
        k_neighbors: int = 5,
        random_state: int = 42,
        task_type: str = "classification",
        categorical_features=None,
    ):
        self.method = method
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.task_type = task_type
        self.categorical_features = categorical_features

    # --- estadísticas que consume la interfaz y el bundle ---
    @property
    def n_before(self) -> Optional[int]:
        return getattr(self, "_n_before", None)

    @property
    def n_after(self) -> Optional[int]:
        return getattr(self, "_n_after", None)

    @property
    def classes_before(self) -> tuple:
        return getattr(self, "_classes_before", ())

    @property
    def classes_after(self) -> tuple:
        return getattr(self, "_classes_after", ())

    @property
    def stats(self) -> dict:
        """Lo que se guarda en los metadatos del bundle."""
        return {
            "method": self.method,
            "k_neighbors": self.k_neighbors,
            "categorical_features": self.categorical_features,
            "n_before": self.n_before,
            "n_after": self.n_after,
            "classes_before": list(self.classes_before),
            "classes_after": list(self.classes_after),
        }

    # --- sklearn ---
    def fit(self, X, y=None):
        self._comprobar_tarea()
        y = np.asarray(y)
        self._y = y
        self._clases, conteos = np.unique(y, return_counts=True)
        self._conteos = conteos
        self._n_before = int(y.size)
        self._n_after = None
        self._classes_before = tuple(self._clases.tolist())
        self._classes_after = ()
        return self

    def fit_resample(self, X, y=None):
        """Remuestrea y devuelve `(X_resampleado, y_resampleado)`.

        Es la interfaz de `imblearn`: por eso el balancer puede vivir dentro
        de un pipeline y seguir llegando el `y` ajustado al modelo. No define
        `transform` ni `fit_transform` a propósito: `imblearn.pipeline.Pipeline`
        rechaza los pasos que son a la vez transformadores y samplers, y en
        `predict` los samplers se omiten (el esquema no cambia).
        """
        self.fit(X, y)
        X2, y2 = self._remuestrear(X, y)
        self._guardar_resultado(y2)
        return X2, y2

    def _remuestrear(self, X, y):
        """Aplica la estrategia y devuelve `(X', y')`."""
        if self.method in (None, "none"):
            return X, y

        if self.method in ("random_under", "random_over"):
            if self.method == "random_under":
                indices = self._indices_submuestreo()
            else:
                indices = self._indices_sobremuestreo()
            return _indexar(X, indices), np.asarray(y)[indices]

        return self._indices_sinteticos(X, y)

    def _guardar_resultado(self, y_resampleado):
        """Anota tamaños y clases para la interfaz y el bundle."""
        self._n_after = int(len(y_resampleado))
        self._classes_after = tuple(np.unique(y_resampleado).tolist())

    # --- utilidades ---
    def _comprobar_tarea(self):
        from core import model_specs

        if self.task_type != model_specs.CLASSIFICATION:
            raise ValueError(
                "El balanceo de clases solo tiene sentido en clasificación; "
                f"la tarea indicada es «{self.task_type}»."
            )

    def _rng(self):
        return np.random.default_rng(self.random_state)

    def _indices_submuestreo(self) -> np.ndarray:
        rng = self._rng()
        objetivo = int(self._conteos.min())
        partes = []
        for clase, n in zip(self._clases, self._conteos):
            posiciones = np.flatnonzero(self._y == clase)
            if n > objetivo:
                partes.append(rng.choice(posiciones, size=objetivo, replace=False))
            else:
                partes.append(posiciones)
        return np.sort(np.concatenate(partes))

    def _indices_sobremuestreo(self) -> np.ndarray:
        rng = self._rng()
        objetivo = int(self._conteos.max())
        partes = []
        for clase, n in zip(self._clases, self._conteos):
            posiciones = np.flatnonzero(self._y == clase)
            if n < objetivo:
                extra = rng.choice(posiciones, size=objetivo - int(n), replace=True)
                partes.append(np.concatenate([posiciones, extra]))
            else:
                partes.append(posiciones)
        return np.sort(np.concatenate(partes))

    def _indices_sinteticos(self, X, y):
        metodo = self._minimo_vecinos()
        if self.method == "smote":
            from imblearn.over_sampling import SMOTE

            generador = SMOTE(random_state=self.random_state, k_neighbors=metodo)
        elif self.method == "smoten":
            from imblearn.over_sampling import SMOTENC

            generador = SMOTENC(
                random_state=self.random_state,
                k_neighbors=metodo,
                categorical_features=self.categorical_features or [],
            )
        else:
            raise ValueError(
                f"Estrategia de balanceo desconocida: «{self.method}». "
                f"Usa una de: {', '.join(STRATEGY_ORDER)}."
            )

        X_denso = X.toarray() if hasattr(X, "toarray") else X
        try:
            X_res, y_res = generador.fit_resample(X_denso, y)
        except ValueError as exc:
            raise ValueError(
                f"SMOTE no se pudo aplicar: {exc}. Si la clase minoritaria "
                "tiene muy pocas instancias, prueba con sobremuestreo "
                "aleatorio o reduce el número de vecinos."
            ) from exc
        return X_res, y_res

    def _minimo_vecinos(self) -> int:
        vecinos = int(self.k_neighbors)
        minimo = int(self._conteos.min()) if self._conteos.size else 0
        if minimo < 2:
            raise ValueError(
                "SMOTE necesita al menos 2 instancias de la clase "
                f"minoritaria y hay {minimo}. Usa sobremuestreo aleatorio."
            )
        if minimo <= vecinos:
            vecinos = max(minimo - 1, 1)
        return vecinos


def _indexar(X, indices):
    """Selecciona filas manteniendo el tipo de contenedor de entrada.

    Un `DataFrame` necesita `iloc`: con `X[indices]` pandas entiende que se
    piden columnas y devuelve un array de objetos.
    """
    if hasattr(X, "iloc"):
        return X.iloc[indices]
    try:
        return X[indices]
    except (IndexError, KeyError, TypeError):
        return np.asarray(X)[indices]
