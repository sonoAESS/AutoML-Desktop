# core/model_specs.py
"""Catálogo declarativo de modelos disponibles en la aplicación.

Cada modelo se describe con un `ModelSpec`: familia, tarea, fábrica y espacio
de búsqueda. `model_trainer` usa el catálogo para construir los pipelines y la
interfaz lo usa para ofrecer únicamente las opciones coherentes con los datos
detectados y para generar los controles de hiperparámetros.

No contiene lógica de interfaz ni de Qt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

CLASSIFICATION = "classification"
REGRESSION = "regression"

ALL_FAMILIES = "todas"

FAMILIES = {
    "lineal": "Modelos lineales",
    "arbol": "Árboles de decisión",
    "ensemble": "Conjuntos de modelos",
    "svm": "Máquinas de soporte vectorial",
    "vecinos": "Vecinos más cercanos",
}

#: Métricas de scoring ofrecidas para cada tarea.
METRICS = {
    CLASSIFICATION: ("accuracy", "f1_macro", "roc_auc"),
    REGRESSION: ("rmse", "mae", "r2"),
}

#: Nombre legible de cada métrica, para la interfaz y la documentación.
METRIC_LABELS = {
    "accuracy": "Exactitud",
    "f1_macro": "F1 (macro)",
    "roc_auc": "AUC (ROC One-vs-Rest)",
    "rmse": "RMSE",
    "mae": "MAE",
    "r2": "R²",
}

#: Métricas que se muestran siempre para una tarea, en orden de presentación.
METRIC_KEYS = {
    CLASSIFICATION: ("accuracy", "f1_macro", "roc_auc"),
    REGRESSION: ("rmse", "mae", "r2"),
}

#: Métrica con la que se ordenan y comparan los modelos.
RANKING_METRIC = {CLASSIFICATION: "f1_macro", REGRESSION: "r2"}

#: Filas mínimas recomendadas para que una familia de modelos tenga sentido.
MIN_ROWS = {"svm": 50, "vecinos": 30}

ALL_VALUES = "todos"

#: Marca "el usuario no ha tocado este parámetro": se usa el dominio
#: completo. Permite distinguir de un `None` elegido explícitamente.
SIN_SELECCION = object()


@dataclass(frozen=True)
class ParamSpec:
    """Un hiperparámetro ofrecible al usuario.

    `values` es el dominio que se puede explorar; la interfaz deja elegir un
    valor o, con el sentinel `ALL_VALUES`, el dominio completo.
    """

    name: str
    label: str
    values: tuple = ()

    def display(self, value: Any) -> str:
        """Texto para mostrar un valor en la interfaz."""
        if value is None:
            return "ninguno"
        if value is True:
            return "sí"
        if value is False:
            return "no"
        return str(value)

    def as_grid(self, selection: Any = SIN_SELECCION) -> dict:
        """Devuelve el trozo de grid correspondiente a la selección.

        `SIN_SELECCION` y `ALL_VALUES` usan el dominio completo. Cualquier
        otro valor se toma literal, incluido `None`: `max_depth=None` es una
        decisión del usuario ("sin límite"), no una ausencia de selección.
        """
        if selection is SIN_SELECCION or selection == ALL_VALUES:
            values = list(self.values)
        elif isinstance(selection, (list, tuple)):
            values = list(selection)
        else:
            values = [selection]
        return {f"model__{self.name}": values}


@dataclass(frozen=True)
class ModelSpec:
    """Ficha de un modelo: cómo se construye y con qué parámetros se ajusta."""

    name: str
    family: str
    task_type: str
    factory: Callable[[], Any]
    params: tuple = ()
    supports_probability: bool = False
    #: Ruta del parámetro que acepta `class_weight="balanced"`, si existe.
    class_weight_path: Optional[str] = None
    notes: str = ""

    def build(self) -> Any:
        """Instancia nueva del estimador (nunca se reutiliza la instancia)."""
        return self.factory()

    def param(self, name: str) -> Optional[ParamSpec]:
        for spec in self.params:
            if spec.name == name:
                return spec
        return None

    def grid(self, selections: Optional[dict] = None) -> dict:
        """Grid de búsqueda a partir de las selecciones de la interfaz."""
        selections = selections or {}
        grid = {}
        for spec in self.params:
            seleccion = (
                selections[spec.name] if spec.name in selections else SIN_SELECCION
            )
            grid.update(spec.as_grid(seleccion))
        return grid


@dataclass(frozen=True)
class ModelCompatibility:
    """Comprobación de si un modelo sirve para unos datos concretos."""

    name: str
    family: str
    compatible: bool
    reason: str = ""


# ---------------------------------------------------------------------------
# Catálogo
# ---------------------------------------------------------------------------

_TREE_PARAMS = (
    ParamSpec("max_depth", "Profundidad máxima", (None, 3, 5, 10, 20)),
    ParamSpec("min_samples_split", "Mínimo para dividir", (2, 5, 10)),
    ParamSpec("min_samples_leaf", "Mínimo en hoja", (1, 2, 5)),
)

_FOREST_PARAMS = (
    ParamSpec("n_estimators", "Número de árboles", (100, 200, 400)),
    ParamSpec("max_depth", "Profundidad máxima", (None, 10, 20)),
    ParamSpec("min_samples_split", "Mínimo para dividir", (2, 5, 10)),
    ParamSpec("max_features", "Variables por división", ("sqrt", "log2")),
)

_KNN_PARAMS = (
    ParamSpec("n_neighbors", "Número de vecinos", (3, 5, 7, 11, 15)),
    ParamSpec("weights", "Peso de vecinos", ("uniform", "distance")),
    ParamSpec("p", "Métrica", (1, 2)),
)

CLASSIFICATION_SPECS = {
    "Regresión Logística": ModelSpec(
        name="Regresión Logística",
        family="lineal",
        task_type=CLASSIFICATION,
        factory=lambda: LogisticRegression(max_iter=1000),
        params=(
            ParamSpec(
                "C", "C (inverso de regularización)", (0.01, 0.1, 1.0, 10.0, 100.0)
            ),
            ParamSpec("class_weight", "Peso de clases", (None, "balanced")),
        ),
        supports_probability=True,
        class_weight_path="model__class_weight",
        notes="Interpretable y rápido. Funciona bien con muchas clases.",
    ),
    "Árbol de Decisión": ModelSpec(
        name="Árbol de Decisión",
        family="arbol",
        task_type=CLASSIFICATION,
        factory=lambda: DecisionTreeClassifier(random_state=42),
        params=_TREE_PARAMS,
        supports_probability=True,
        class_weight_path="model__class_weight",
        notes="Muy interpretable, propenso a sobreajustar.",
    ),
    "Random Forest": ModelSpec(
        name="Random Forest",
        family="ensemble",
        task_type=CLASSIFICATION,
        factory=lambda: RandomForestClassifier(n_estimators=200, random_state=42),
        params=_FOREST_PARAMS,
        supports_probability=True,
        class_weight_path="model__class_weight",
        notes="Robusto y preciso; menos interpretable.",
    ),
    "SVM": ModelSpec(
        name="SVM",
        family="svm",
        task_type=CLASSIFICATION,
        factory=lambda: CalibratedClassifierCV(SVC(random_state=42), ensemble=False),
        params=(
            ParamSpec(
                "estimator__C", "C (inverso de regularización)", (0.1, 1.0, 10.0, 100.0)
            ),
            ParamSpec("estimator__kernel", "Núcleo", ("rbf", "linear")),
            ParamSpec("estimator__gamma", "Gamma", ("scale", "auto")),
        ),
        supports_probability=True,
        class_weight_path="model__estimator__class_weight",
        notes="Muy potente con muchas variables; lento con muchos datos.",
    ),
    "KNN": ModelSpec(
        name="KNN",
        family="vecinos",
        task_type=CLASSIFICATION,
        factory=lambda: KNeighborsClassifier(),
        params=_KNN_PARAMS,
        supports_probability=True,
        notes="Simple; sensible a la escala de las variables.",
    ),
}

REGRESSION_SPECS = {
    "Regresión Lineal": ModelSpec(
        name="Regresión Lineal",
        family="lineal",
        task_type=REGRESSION,
        factory=lambda: LinearRegression(),
        params=(
            ParamSpec("fit_intercept", "Con término independiente", (True, False)),
        ),
        notes="Línea base interpretable para relaciones lineales.",
    ),
    "Ridge": ModelSpec(
        name="Ridge",
        family="lineal",
        task_type=REGRESSION,
        factory=lambda: Ridge(),
        params=(
            ParamSpec("alpha", "Regularización", (0.01, 0.1, 1.0, 10.0, 100.0)),
            ParamSpec("fit_intercept", "Con término independiente", (True, False)),
        ),
        notes="Regresión lineal con regularización L2.",
    ),
    "Árbol de Decisión": ModelSpec(
        name="Árbol de Decisión",
        family="arbol",
        task_type=REGRESSION,
        factory=lambda: DecisionTreeRegressor(random_state=42),
        params=_TREE_PARAMS,
        notes="Muy interpretable, propenso a sobreajustar.",
    ),
    "Random Forest": ModelSpec(
        name="Random Forest",
        family="ensemble",
        task_type=REGRESSION,
        factory=lambda: RandomForestRegressor(n_estimators=200, random_state=42),
        params=_FOREST_PARAMS,
        notes="Robusto y preciso; menos interpretable.",
    ),
    "SVR": ModelSpec(
        name="SVR",
        family="svm",
        task_type=REGRESSION,
        factory=lambda: SVR(),
        params=(
            ParamSpec("C", "C (inverso de regularización)", (0.1, 1.0, 10.0, 100.0)),
            ParamSpec("kernel", "Núcleo", ("rbf", "linear")),
            ParamSpec("epsilon", "Epsilon", (0.01, 0.1, 0.5)),
        ),
        notes="Muy potente con muchas variables; lento con muchos datos.",
    ),
    "KNN": ModelSpec(
        name="KNN",
        family="vecinos",
        task_type=REGRESSION,
        factory=lambda: KNeighborsRegressor(),
        params=_KNN_PARAMS,
        notes="Simple; sensible a la escala de las variables.",
    ),
}

#: Catálogo indexado por tarea.
SPECS = {CLASSIFICATION: CLASSIFICATION_SPECS, REGRESSION: REGRESSION_SPECS}


def iter_specs(task_type: Optional[str] = None):
    """Itera sobre `(nombre, ficha)` de los modelos de una tarea."""
    tasks = [task_type] if task_type else [CLASSIFICATION, REGRESSION]
    for task in tasks:
        yield from SPECS[task].items()


def get_spec(model_name: str, task_type: Optional[str] = None) -> ModelSpec:
    """Ficha de un modelo. Lanza `KeyError` si el modelo no existe."""
    if task_type:
        return SPECS[task_type][model_name]
    for _, spec in iter_specs():
        if spec.name == model_name:
            return spec
    raise KeyError(model_name)


def list_models(
    task_type: Optional[str] = None,
    family: Optional[str] = None,
) -> list:
    """Nombres de modelos filtrados por tarea y familia."""
    return [
        name
        for name, spec in iter_specs(task_type)
        if not family or family == ALL_FAMILIES or spec.family == family
    ]


def list_families(task_type: Optional[str] = None) -> list:
    """Familias que tienen al menos un modelo para la tarea indicada."""
    families = []
    for _, spec in iter_specs(task_type):
        if spec.family not in families:
            families.append(spec.family)
    return [key for key in FAMILIES if key in families]


def available_metrics(task_type: str, model_name: Optional[str] = None) -> list:
    """Métricas de scoring válidas para una tarea.

    `model_name` se acepta por compatibilidad pero **no** filtra: el AUC
    aparece siempre y su indisponibilidad se explica después, con
    `metrics["auc_motivo"]` (ver `model_trainer._compute_metrics`). Para el
    catálogo actual el valor devuelto no cambia en ningún caso real.
    """
    return list(METRICS.get(task_type, ()))


def class_weight_path(model_name: str, task_type: str = CLASSIFICATION):
    """Ruta del parámetro `class_weight` de un modelo, o `None` si no lo tiene.

    KNN no lo admite; el SVM lo tiene anidado dentro del `SVC` calibrado.
    """
    try:
        return get_spec(model_name, task_type).class_weight_path
    except KeyError:
        return None


def metric_label(key: str) -> str:
    """Nombre legible de una métrica (o la clave si no está en el catálogo)."""
    return METRIC_LABELS.get(key, key)


def metrics_for_ranking(task_type: str) -> str:
    """Métrica usada para ordenar y comparar modelos."""
    return RANKING_METRIC.get(task_type, "f1_macro")


def family_label(family: str) -> str:
    """Nombre legible de una familia de modelos."""
    return FAMILIES.get(family, family)


def build_estimator(model_name: str, task_type: Optional[str] = None):
    """Instancia el estimador del modelo indicado."""
    return get_spec(model_name, task_type).build()


def evaluate_compatibility(
    task_type: str,
    profiles,
    target: Optional[str] = None,
    n_rows: Optional[int] = None,
) -> list:
    """Decide qué modelos pueden usarse con esos datos y por qué no los demás.

    Parámetros
    ----------
    task_type : {"classification", "regression"}
    profiles : list[ColumnProfile]
        Perfil de las columnas del dataset.
    target : str | None
        Columna objetivo ya elegida; se excluye de las variables predictors.
    n_rows : int | None
        Número de filas; si es `None` se deduce del perfil.

    Devuelve
    --------
    list[ModelCompatibility]
    """
    features = [p for p in profiles if not target or p.name != target]
    blocking = [p for p in features if not p.usable_as_feature]
    rows = (
        n_rows if n_rows is not None else max((p.n_rows for p in features), default=0)
    )

    results = []
    for name, spec in iter_specs(task_type):
        if blocking:
            nombres = ", ".join(p.name for p in blocking)
            reason = f"requiere convertir o eliminar: {nombres}"
            results.append(ModelCompatibility(name, spec.family, False, reason))
            continue
        needed = MIN_ROWS.get(spec.family, 0)
        if rows and rows < needed:
            reason = f"necesita al menos {needed} filas (hay {rows})"
            results.append(ModelCompatibility(name, spec.family, False, reason))
            continue
        results.append(ModelCompatibility(name, spec.family, True))
    return results


def summarize_compatibility(results: list) -> str:
    """Mensaje para la interfaz con el motivo de los modelos descartados."""
    excluded = [r for r in results if not r.compatible]
    if not excluded:
        return "Todos los modelos son compatibles con estos datos."
    motivos = "; ".join(f"{r.name} ({r.reason})" for r in excluded)
    return f"Modelos no compatibles con estos datos → {motivos}"
