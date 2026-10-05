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
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    ElasticNet,
    HuberRegressor,
    Lasso,
    LinearRegression,
    LogisticRegression,
    Ridge,
    SGDClassifier,
    SGDRegressor,
    TheilSenRegressor,
)
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import (
    KNeighborsClassifier,
    KNeighborsRegressor,
    RadiusNeighborsClassifier,
    RadiusNeighborsRegressor,
)
from sklearn.svm import SVC, SVR, LinearSVC
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
    "naive_bayes": "Naive Bayes",
    "lda": "Análisis discriminante",
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

#: Filas mínimas recomendadas para que una familia de modelos tenga sentido.
MIN_ROWS = {"svm": 50, "vecinos": 30, "naive_bayes": 10, "lda": 20}

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

#: Radio de búsqueda, no número de vecinos: útil cuando la densidad de los datos
#: no es uniforme (los vecinos lejanos dejan de importar). Los valores son
#: grandes a propósito: el radio por defecto de sklearn (1.0) deja muestras sin
#: vecinos en espacio escalado y sus predicciones salen como NaN.
_RADIUS_PARAMS = (
    ParamSpec("radius", "Radio de búsqueda", (2.0, 5.0, 10.0)),
    ParamSpec("weights", "Peso de vecinos", ("uniform", "distance")),
)

#: ExtraTrees comparte casi todo con un bosque, pero su criterio de división es
#: aleatorio: por eso `min_samples_split` es menos determinante.
_EXTRA_TREES_PARAMS = (
    ParamSpec("n_estimators", "Número de árboles", (100, 200, 400)),
    ParamSpec("max_depth", "Profundidad máxima", (None, 10, 20)),
    ParamSpec("min_samples_split", "Mínimo para dividir", (2, 5, 10)),
    ParamSpec("max_features", "Variables por división", ("sqrt", "log2")),
)

_BAGGING_PARAMS = (
    ParamSpec("n_estimators", "Número de estimadores", (10, 20, 50)),
    ParamSpec("max_samples", "Proporción de muestras", (0.5, 0.8, 1.0)),
    ParamSpec("max_features", "Proporción de variables", (0.5, 0.8, 1.0)),
)

_ADABOOST_PARAMS = (
    ParamSpec("n_estimators", "Número de estimadores", (50, 100, 200)),
    ParamSpec("learning_rate", "Tasa de aprendizaje", (0.01, 0.1, 0.5, 1.0)),
)

_BOOST_PARAMS = (
    ParamSpec("n_estimators", "Número de árboles", (100, 200, 400)),
    ParamSpec("learning_rate", "Tasa de aprendizaje", (0.01, 0.05, 0.1, 0.3)),
    ParamSpec("max_depth", "Profundidad máxima", (3, 5, 10)),
    ParamSpec("subsample", "Proporción de muestras", (0.8, 1.0)),
)

#: El boosting por histogramas es el más rápido de la familia y admite
#: `class_weight`, que los otros tres no aceptan.
_HIST_PARAMS = (
    ParamSpec("learning_rate", "Tasa de aprendizaje", (0.05, 0.1, 0.3)),
    ParamSpec("max_iter", "Número de iteraciones", (100, 200, 500)),
    ParamSpec("max_depth", "Profundidad máxima", (None, 3, 5, 10)),
    ParamSpec("min_samples_leaf", "Mínimo en hoja", (20, 40)),
    ParamSpec("l2_regularization", "Regularización L2", (0.0, 0.1, 1.0)),
)

_LINEAR_REG_PARAMS = (
    ParamSpec("alpha", "Fuerza de la regularización", (0.001, 0.01, 0.1, 1.0)),
    ParamSpec("fit_intercept", "Con término independiente", (True, False)),
)

_CLASS_WEIGHT = ParamSpec("class_weight", "Peso de clases", (None, "balanced"))

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
        class_weight_path="model__class_weight",
        notes="Interpretable y rápido. Funciona bien con muchas clases.",
    ),
    "Árbol de Decisión": ModelSpec(
        name="Árbol de Decisión",
        family="arbol",
        task_type=CLASSIFICATION,
        factory=lambda: DecisionTreeClassifier(random_state=42),
        params=_TREE_PARAMS,
        class_weight_path="model__class_weight",
        notes="Muy interpretable, propenso a sobreajustar.",
    ),
    "Random Forest": ModelSpec(
        name="Random Forest",
        family="ensemble",
        task_type=CLASSIFICATION,
        factory=lambda: RandomForestClassifier(n_estimators=200, random_state=42),
        params=_FOREST_PARAMS,
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
        class_weight_path="model__estimator__class_weight",
        notes="Muy potente con muchas variables; lento con muchos datos.",
    ),
    "KNN": ModelSpec(
        name="KNN",
        family="vecinos",
        task_type=CLASSIFICATION,
        factory=lambda: KNeighborsClassifier(),
        params=_KNN_PARAMS,
        notes="Simple; sensible a la escala de las variables.",
    ),
    "SGD": ModelSpec(
        name="SGD",
        family="lineal",
        task_type=CLASSIFICATION,
        factory=lambda: SGDClassifier(max_iter=1000, tol=1e-3, random_state=42),
        params=(
            ParamSpec(
                "loss", "Función de pérdida", ("hinge", "log_loss", "modified_huber")
            ),
            ParamSpec("penalty", "Penalización", ("l2", "l1", "elasticnet")),
            ParamSpec("alpha", "Fuerza de la penalización", (1e-5, 1e-4, 1e-3, 1e-2)),
            _CLASS_WEIGHT,
        ),
        class_weight_path="model__class_weight",
        notes="Lineal pero muy rápido; el punto de partida con muchos datos.",
    ),
    "ExtraTrees": ModelSpec(
        name="ExtraTrees",
        family="ensemble",
        task_type=CLASSIFICATION,
        factory=lambda: ExtraTreesClassifier(n_estimators=200, random_state=42),
        params=_EXTRA_TREES_PARAMS + (_CLASS_WEIGHT,),
        class_weight_path="model__class_weight",
        notes="Bosque con divisiones aleatorias; más rápido que Random Forest.",
    ),
    "Bagging": ModelSpec(
        name="Bagging",
        family="ensemble",
        task_type=CLASSIFICATION,
        factory=lambda: BaggingClassifier(n_estimators=20, random_state=42),
        params=_BAGGING_PARAMS,
        notes="Agrupa estimadores débiles; reduce la varianza.",
    ),
    "AdaBoost": ModelSpec(
        name="AdaBoost",
        family="ensemble",
        task_type=CLASSIFICATION,
        factory=lambda: AdaBoostClassifier(n_estimators=100, random_state=42),
        params=_ADABOOST_PARAMS,
        notes="Boosting sobre árboles poco profundos; sensible al ruido.",
    ),
    "GradientBoosting": ModelSpec(
        name="GradientBoosting",
        family="ensemble",
        task_type=CLASSIFICATION,
        factory=lambda: GradientBoostingClassifier(n_estimators=100, random_state=42),
        params=_BOOST_PARAMS,
        notes="Boosting secuencial; muy preciso, costoso de entrenar.",
    ),
    "HistGradientBoosting": ModelSpec(
        name="HistGradientBoosting",
        family="ensemble",
        task_type=CLASSIFICATION,
        factory=lambda: HistGradientBoostingClassifier(max_iter=200, random_state=42),
        params=_HIST_PARAMS + (_CLASS_WEIGHT,),
        class_weight_path="model__class_weight",
        notes="Boosting por histogramas: rápido incluso con muchos datos.",
    ),
    "LinearSVC": ModelSpec(
        name="LinearSVC",
        family="svm",
        task_type=CLASSIFICATION,
        # `squared_hinge` fijo y `max_iter=3000`: con la `hinge` por defecto y
        # `C=100` liblinear no converge ni con 10000 iteraciones, y en una
        # búsqueda eso se traduce en un modelo entregado a medio entrenar.
        factory=lambda: LinearSVC(loss="squared_hinge", max_iter=3000, random_state=42),
        params=(
            ParamSpec("C", "C (inverso de regularización)", (0.1, 1.0, 10.0, 100.0)),
            _CLASS_WEIGHT,
        ),
        class_weight_path="model__class_weight",
        notes="SVM lineal: mucho más rápido que el SVM con núcleo.",
    ),
    "RadiusNeighbors": ModelSpec(
        name="RadiusNeighbors",
        family="vecinos",
        task_type=CLASSIFICATION,
        factory=lambda: RadiusNeighborsClassifier(radius=5.0),
        params=_RADIUS_PARAMS,
        notes="Vecinos dentro de un radio; maneja bien densidades desiguales.",
    ),
    "GaussianNB": ModelSpec(
        name="GaussianNB",
        family="naive_bayes",
        task_type=CLASSIFICATION,
        factory=lambda: GaussianNB(),
        params=(
            ParamSpec(
                "var_smoothing",
                "Estabilidad de varianzas",
                (1e-11, 1e-9, 1e-7),
            ),
        ),
        notes="Supone que las variables siguen una normal; muy rápido.",
    ),
    "BernoulliNB": ModelSpec(
        name="BernoulliNB",
        family="naive_bayes",
        task_type=CLASSIFICATION,
        factory=lambda: BernoulliNB(),
        params=(
            ParamSpec("alpha", "Suavizado", (0.1, 0.5, 1.0, 2.0)),
            ParamSpec("binarize", "Umbral de binarización", (0.0, 0.5, 1.0)),
        ),
        notes="Para variables binarias o de texto (datos Sparso).",
    ),
    "LinearDiscriminant": ModelSpec(
        name="LinearDiscriminant",
        family="lda",
        task_type=CLASSIFICATION,
        # `solver="lsqr"` fijo: `svd` no acepta `shrinkage`, y un producto
        # cartesiano `solver x shrinkage` llenaría la búsqueda de combinaciones
        # que fallan y puntúan NaN sin avisar.
        factory=lambda: LinearDiscriminantAnalysis(solver="lsqr"),
        params=(ParamSpec("shrinkage", "Regularización", (None, "auto", 0.5)),),
        notes="Proyecta las variables para que las clases se separen mejor.",
    ),
    "QuadraticDiscriminant": ModelSpec(
        name="QuadraticDiscriminant",
        family="lda",
        task_type=CLASSIFICATION,
        factory=lambda: QuadraticDiscriminantAnalysis(),
        params=(ParamSpec("reg_param", "Regularización", (0.0, 0.1, 0.5, 1.0)),),
        notes="Como el lineal, pero con fronteras curvas. Exige muchos datos.",
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
    "Lasso": ModelSpec(
        name="Lasso",
        family="lineal",
        task_type=REGRESSION,
        factory=lambda: Lasso(random_state=42),
        params=_LINEAR_REG_PARAMS,
        notes="Regularización L1: deja a cero las variables inútiles.",
    ),
    "ElasticNet": ModelSpec(
        name="ElasticNet",
        family="lineal",
        task_type=REGRESSION,
        factory=lambda: ElasticNet(random_state=42),
        params=_LINEAR_REG_PARAMS
        + (ParamSpec("l1_ratio", "Proporción de L1", (0.1, 0.3, 0.5, 0.7, 0.9)),),
        notes="Mezcla L1 y L2: seleccionar con estabilidad.",
    ),
    "Huber": ModelSpec(
        name="Huber",
        family="lineal",
        task_type=REGRESSION,
        factory=lambda: HuberRegressor(),
        params=(
            ParamSpec("epsilon", "Umbral de robustez", (1.1, 1.35, 1.5, 2.0)),
            ParamSpec("alpha", "Fuerza de la penalización", (1e-5, 1e-4, 1e-3, 1e-2)),
            ParamSpec("max_iter", "Número de iteraciones", (100, 200, 500)),
        ),
        notes="Como la regresión lineal, pero los valores extremos influyen menos.",
    ),
    "TheilSen": ModelSpec(
        name="TheilSen",
        family="lineal",
        task_type=REGRESSION,
        factory=lambda: TheilSenRegressor(random_state=42),
        params=(
            ParamSpec("fit_intercept", "Con término independiente", (True, False)),
            ParamSpec("max_iter", "Número de iteraciones", (100, 200, 500)),
        ),
        notes="Muy robusto frente a valores atípicos; lenta con muchos datos.",
    ),
    "SGD": ModelSpec(
        name="SGD",
        family="lineal",
        task_type=REGRESSION,
        factory=lambda: SGDRegressor(max_iter=1000, tol=1e-3, random_state=42),
        params=(
            ParamSpec(
                "loss",
                "Función de pérdida",
                ("squared_error", "huber", "epsilon_insensitive"),
            ),
            ParamSpec("penalty", "Penalización", ("l2", "l1", "elasticnet")),
            ParamSpec("alpha", "Fuerza de la penalización", (1e-5, 1e-4, 1e-3, 1e-2)),
        ),
        notes="Regresión lineal por gradiente; escala a datasets enormes.",
    ),
    "ExtraTrees": ModelSpec(
        name="ExtraTrees",
        family="ensemble",
        task_type=REGRESSION,
        factory=lambda: ExtraTreesRegressor(n_estimators=200, random_state=42),
        params=_EXTRA_TREES_PARAMS,
        notes="Bosque con divisiones aleatorias; más rápido que Random Forest.",
    ),
    "Bagging": ModelSpec(
        name="Bagging",
        family="ensemble",
        task_type=REGRESSION,
        factory=lambda: BaggingRegressor(n_estimators=20, random_state=42),
        params=_BAGGING_PARAMS,
        notes="Agrupa estimadores débiles; reduce la varianza.",
    ),
    "AdaBoost": ModelSpec(
        name="AdaBoost",
        family="ensemble",
        task_type=REGRESSION,
        factory=lambda: AdaBoostRegressor(n_estimators=100, random_state=42),
        params=_ADABOOST_PARAMS
        + (
            ParamSpec(
                "loss", "Función de pérdida", ("linear", "square", "exponential")
            ),
        ),
        notes="Boosting sobre árboles poco profundos; sensible al ruido.",
    ),
    "GradientBoosting": ModelSpec(
        name="GradientBoosting",
        family="ensemble",
        task_type=REGRESSION,
        factory=lambda: GradientBoostingRegressor(n_estimators=100, random_state=42),
        params=_BOOST_PARAMS,
        notes="Boosting secuencial; muy preciso, costoso de entrenar.",
    ),
    "HistGradientBoosting": ModelSpec(
        name="HistGradientBoosting",
        family="ensemble",
        task_type=REGRESSION,
        factory=lambda: HistGradientBoostingRegressor(max_iter=200, random_state=42),
        params=_HIST_PARAMS,
        notes="Boosting por histogramas: rápido incluso con muchos datos.",
    ),
    "KernelRidge": ModelSpec(
        name="KernelRidge",
        family="svm",
        task_type=REGRESSION,
        factory=lambda: KernelRidge(),
        params=(
            ParamSpec("alpha", "Regularización", (0.1, 1.0, 10.0)),
            ParamSpec("kernel", "Núcleo", ("linear", "rbf", "poly")),
            # A diferencia de SVC, `gamma` aquí es un float: ni "scale" ni
            # "auto" existen en KernelRidge.
            ParamSpec("gamma", "Gamma", (0.01, 0.1, 1.0)),
        ),
        notes="Regresión con kernel de función; equivalente a SVR pero más rápida.",
    ),
    "RadiusNeighbors": ModelSpec(
        name="RadiusNeighbors",
        family="vecinos",
        task_type=REGRESSION,
        factory=lambda: RadiusNeighborsRegressor(radius=5.0),
        params=_RADIUS_PARAMS,
        notes="Vecinos dentro de un radio; maneja bien densidades desiguales.",
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
