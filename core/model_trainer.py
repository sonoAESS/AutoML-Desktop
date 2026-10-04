# core/model_trainer.py
"""Construcción de pipelines, entrenamiento y métricas.

El catálogo de modelos vive en `core.model_specs`; aquí solo se assemblan los
pipelines, se entrena y se calculan las métricas.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from core import model_specs
from core.model_specs import CLASSIFICATION, REGRESSION
from core.preprocessor import (
    ColumnNormalizer,
    ColumnTyper,
    apply_transformations,
    categorical_columns,
    numeric_columns,
)

#: Modelos por tarea (instancias de referência, se mantienen por compatibilidad).
CLASSIFIERS = {
    name: spec.build() for name, spec in model_specs.CLASSIFICATION_SPECS.items()
}
REGRESSORS = {name: spec.build() for name, spec in model_specs.REGRESSION_SPECS.items()}

#: Espacio de búsqueda por defecto de cada modelo.
PARAM_GRIDS = {name: spec.grid() for name, spec in model_specs.iter_specs()}


def search_grid(model_name, task_type, selections=None):
    """Grid de búsqueda de un modelo según lo elegido en la interfaz."""
    return model_specs.get_spec(model_name, task_type).grid(selections)


def search_size(
    model_name,
    task_type,
    selections=None,
    search_type="grid",
    cv=5,
    n_iter=20,
):
    """Cuántas combinaciones y cuántos ajustes hará una búsqueda.

    Parámetros
    ----------
    selections : dict | None
        Valores elegidos en la interfaz para cada hiperparámetro.
    search_type : {"grid", "random"}
        Estrategia de búsqueda.
    cv : int
        Folds de validación cruzada.
    n_iter : int
        Iteraciones pedidas si `search_type == "random"`.

    Devuelve
    --------
    (n_combos, n_ajustes) : tuple[int, int]
        El número de ajustes es el de modelos que se entrenarán de verdad:
        combinaciones por folds (limitado por `n_iter` en la aleatoria).
    """
    param_grid = search_grid(model_name, task_type, selections)
    if not param_grid:
        return 0, 0
    combos = _grid_size(param_grid)
    if search_type == "random":
        combos = min(n_iter, combos)
    return combos, combos * max(int(cv), 1)


def available_metrics(task_type, model_name=None):
    """Métricas de scoring compatibles con la tarea y el modelo."""
    return model_specs.available_metrics(task_type, model_name)


def compatible_models(task_type, profiles, target=None, n_rows=None):
    """Modelos usables con los datos described en `profiles`."""
    return model_specs.evaluate_compatibility(
        task_type, profiles, target=target, n_rows=n_rows
    )


def build_preprocessor(X):
    num_cols = numeric_columns(X)
    cat_cols = categorical_columns(X)

    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        [
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )


def _onehot_feature_indices(X) -> list:
    """Índices de las columnas categóricas en la salida del preprocesador.

    El preprocesador codifica las categóricas con `OneHotEncoder`, así que en
    la matriz que ve el balancer esas columnas son las que SMOTEN debe tratar
    como categóricas. Se localizan por el prefijo `cat_` que pone
    `ColumnTransformer.get_feature_names_out()`.
    """
    preprocesador = build_preprocessor(X)
    preprocesador.fit(X)
    nombres = [str(nombre) for nombre in preprocesador.get_feature_names_out()]
    return [i for i, nombre in enumerate(nombres) if nombre.startswith("cat_")]


def build_pipeline(
    df,
    model_name,
    task_type,
    target,
    casts=None,
    normalizations=None,
    balancing=None,
):
    """Construye el pipeline (tipos + normalización + preprocesado + modelo).

    Los pasos `ColumnTyper` y `ColumnNormalizer` son los mismos que se aplican
    al dataset para mostrarlo y exportarlo, de modo que un modelo guardado
    reproduce exactamente la transformación sobre datos nuevos.

    Con `balancing` de remuestreo el pipeline pasa a ser de `imblearn`: es el
    único que propaga el `y` ajustado hasta el modelo. `class_weight` no añade
    ningún paso, solo fija el parámetro del estimador.
    """
    X = df.drop(columns=[target])
    steps = []
    if casts:
        steps.append(("typer", ColumnTyper(casts)))
    if normalizations:
        steps.append(("normalizer", ColumnNormalizer(normalizations)))
    steps.append(("preprocessor", build_preprocessor(X)))

    metodo = (balancing or {}).get("method")
    if metodo and metodo not in ("none", "class_weight"):
        from core.balancing import BalancedSampler

        categoricas = balancing.get("categorical_features")
        if metodo == "smoten" and categoricas is None:
            categoricas = _onehot_feature_indices(X)
            if not categoricas:
                raise ValueError(
                    "SMOTEN necesita columnas categóricas y este dataset no "
                    "tiene. Usa SMOTE, que es equivalente con datos ya "
                    "codificados."
                )
        steps.append(
            (
                "balancer",
                BalancedSampler(
                    method=metodo,
                    k_neighbors=balancing.get("k_neighbors", 5),
                    random_state=balancing.get("random_state", 42),
                    task_type=task_type,
                    categorical_features=categoricas,
                ),
            )
        )

    steps.append(("model", model_specs.build_estimator(model_name, task_type)))

    if metodo == "class_weight":
        path = model_specs.class_weight_path(model_name, task_type)
        if path is None:
            raise ValueError(
                f"El modelo {model_name} no admite pesos de clase. Usa "
                "submuestreo, sobremuestreo o SMOTE."
            )
        pipe = Pipeline(steps)
        pipe.set_params(**{path: "balanced"})
        return pipe

    if any(nombre == "balancer" for nombre, _ in steps):
        from imblearn.pipeline import Pipeline as ImbPipeline

        return ImbPipeline(steps)
    return Pipeline(steps)


def transform_dataset(df, casts=None, normalizations=None):
    """Aplica al dataset los mismos pasos previos del pipeline.

    Sirve para obtener el dataset transformado antes de entrenar, con las
    constantes calculadas sobre el dataset completo.
    """
    return apply_transformations(df, casts, normalizations)


#: Pasos del pipeline que transforman el dataset sin codificar variables.
DATASET_STEPS = ("typer", "normalizer")


def transform_with_pipeline(df, pipe):
    """Transforma el dataset con los pasos ya entrenados del pipeline.

    A diferencia de `transform_dataset`, reutiliza las constantes aprendidas
    durante el entrenamiento, de modo que el dataset que se guarda junto al
    modelo es exactamente el que el modelo procesó.
    """
    out = df
    steps = getattr(pipe, "named_steps", {})
    for name in DATASET_STEPS:
        step = steps.get(name)
        if step is not None:
            out = step.transform(out)
    return out


def train_model(
    df,
    target,
    model_name,
    task_type,
    test_size=0.2,
    random_state=42,
    casts=None,
    normalizations=None,
    balancing=None,
):
    """Entrena un modelo sin búsqueda de hiperparámetros.

    Parámetros
    ----------
    casts : dict | None
        Tipos forzados por columna, `{columna: tipo}`.
    normalizations : dict | None
        Normalización por columna, `{columna: método}`.
    balancing : dict | None
        Estrategia de balanceo, `{method, k_neighbors, random_state}`.

    Devuelve
    --------
    pipe : sklearn.pipeline.Pipeline
        Pipeline entrenado.
    metrics : dict[str, float]
        Métricas calculadas por `_compute_metrics`.
    test_data : tuple
        (X_test, y_test, y_pred)
    """
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if task_type == CLASSIFICATION else None,
    )

    pipe = build_pipeline(
        df,
        model_name,
        task_type,
        target,
        casts=casts,
        normalizations=normalizations,
        balancing=balancing,
    )
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    metrics = _compute_metrics(
        task_type=task_type,
        y_test=y_test,
        y_pred=y_pred,
        pipe=pipe,
        X_test=X_test,
    )

    return pipe, metrics, (X_test, y_test, y_pred)


def _scoring_for(task_type: str, metric: str) -> str:
    """Mapea la métrica elegida por el usuario al scoring de sklearn."""
    if task_type == "classification":
        return {
            "accuracy": "accuracy",
            "f1_macro": "f1_macro",
            "roc_auc": "roc_auc",
        }.get(metric, "accuracy")
    else:
        # En regresión, sklearn usa "neg_mean_squared_error" etc.
        return {
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        }.get(metric, "r2")


def _resolve_roc_auc(model, y) -> str:
    """Devuelve un scoring de ROC AUC válido para el modelo y las etiquetas.

    Sin `predict_proba` no se puede calcular ROC AUC, así que se cae a
    `accuracy`. Con más de dos clases hace falta la variante one-vs-rest
    ("roc_auc_ovr"): "roc_auc" solo es válido en clasificación binaria.
    """
    if not hasattr(model, "predict_proba"):
        return "accuracy"
    if y.nunique(dropna=True) > 2:
        return "roc_auc_ovr"
    return "roc_auc"


def _sin_informar(etapa, actual, total):
    """Callback por defecto: descarta los avisos de progreso."""


def tune_model(
    df,
    target,
    model_name,
    task_type,
    search_type="grid",
    metric="accuracy",
    cv=5,
    n_iter=20,
    test_size=0.2,
    random_state=42,
    selections=None,
    casts=None,
    normalizations=None,
    balancing=None,
    progress_callback=None,
):
    """Entrena un modelo con búsqueda de hiperparámetros.

    Parámetros
    ----------
    search_type : {"grid", "random"}
    metric : str
        Métrica a optimizar (depende de la tarea).
    cv : int
        Número de folds de validación cruzada.
    n_iter : int
        Número de combinaciones a probar si search_type == "random".
    selections : dict | None
        Valores elegidos en la interfaz para cada hiperparámetro. Los que no
        aparecen se explorarían en su dominio completo.
    casts : dict | None
        Tipos forzados por columna, `{columna: tipo}`.
    normalizations : dict | None
        Normalización por columna, `{columna: método}`.
    balancing : dict | None
        Estrategia de balanceo. Con `class_weight` el peso se añade al espacio
        de búsqueda en lugar de fijarse.
    progress_callback : callable | None
        Función que se invoca con `(etapa, actual, total)` en cada paso
        relevante: antes de explorar, al reentrenar y al calcular métricas.
        Se usa desde la UI para reportar progreso. Si el modelo no tiene
        espacio de búsqueda solo se avisa del entrenamiento simple.

    Devuelve
    --------
    best_pipe, metrics, (X_test, y_test, y_pred), cv_results
    """
    X = df.drop(columns=[target])
    y = df[target]

    reportar = progress_callback or _sin_informar
    reportar("Preparando los datos y el pipeline…", 0, 0)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if task_type == CLASSIFICATION else None,
    )

    pipe = build_pipeline(
        df,
        model_name,
        task_type,
        target,
        casts=casts,
        normalizations=normalizations,
        balancing=balancing,
    )

    param_grid = search_grid(model_name, task_type, selections)
    metodo = (balancing or {}).get("method")
    if metodo == "class_weight":
        path = model_specs.class_weight_path(model_name, task_type)
        if path is None:
            raise ValueError(
                f"El modelo {model_name} no admite pesos de clase. Usa "
                "submuestreo, sobremuestreo o SMOTE."
            )
        param_grid = {**param_grid, path: ["balanced"]}
    if not param_grid:
        # Sin espacio de búsqueda: entrenamiento normal.
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        metrics = _compute_metrics(task_type, y_test, y_pred, pipe=pipe, X_test=X_test)
        reportar("Entrenando el modelo…", 1, 1)
        return pipe, metrics, (X_test, y_test, y_pred), {}

    combos, ajustes = search_size(
        model_name,
        task_type,
        selections,
        search_type=search_type,
        cv=cv,
        n_iter=n_iter,
    )
    reportar(f"Explorando {combos} combinaciones ({ajustes} ajustes)…", 0, ajustes)

    scoring = _scoring_for(task_type, metric)
    if scoring == "roc_auc":
        scoring = _resolve_roc_auc(pipe.named_steps["model"], y)

    if search_type == "random":
        search = RandomizedSearchCV(
            pipe,
            param_distributions=param_grid,
            n_iter=min(n_iter, _grid_size(param_grid)),
            scoring=scoring,
            cv=cv,
            random_state=random_state,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )
    else:
        search = GridSearchCV(
            pipe,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            refit=True,
            verbose=0,
        )

    search.fit(X_train, y_train)
    reportar("Refinando la mejor configuración…", ajustes, ajustes)

    best_pipe = search.best_estimator_
    y_pred = best_pipe.predict(X_test)
    reportar("Calculando las métricas finales…", ajustes, ajustes)

    metrics = _compute_metrics(
        task_type=task_type,
        y_test=y_test,
        y_pred=y_pred,
        pipe=best_pipe,
        X_test=X_test,
    )

    metrics["best_cv_score"] = float(search.best_score_)
    metrics["best_params"] = search.best_params_

    return best_pipe, metrics, (X_test, y_test, y_pred), search.cv_results_


def _grid_size(param_grid: dict) -> int:
    """Número total de combinaciones del grid (producto cartesiano)."""
    from math import prod

    return prod(len(v) for v in param_grid.values())


AUC_MOTIVO_SIN_DATOS = "No se han proporcionado datos de prueba para calcular el AUC."
AUC_MOTIVO_SIN_PUNTUACIONES = (
    "Este modelo no expone probabilidades ni puntuaciones de decisión, "
    "necesarias para el AUC."
)
AUC_MOTIVO_UNA_CLASE = (
    "El AUC necesita al menos dos clases distintas en los datos de prueba."
)


def _auc_scores(pipe, X_test) -> tuple:
    """Puntuaciones del pipeline para el AUC: `(scores, clases)`.

    Se llama al pipeline completo (no al estimador suelto) para que el AUC siga
    siendo correcto aunque el pipeline tenga typer, normalizador, selector o
    balancer antes del modelo.
    """
    if hasattr(pipe, "predict_proba"):
        proba = np.asarray(pipe.predict_proba(X_test))
        modelo = pipe.named_steps.get("model") if hasattr(pipe, "named_steps") else None
        clases = list(getattr(modelo, "classes_", range(proba.shape[1])))
        if proba.shape[1] == len(clases):
            return proba, clases
    if hasattr(pipe, "decision_function"):
        scores = np.asarray(pipe.decision_function(X_test))
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        return scores, list(range(scores.shape[1]))
    raise AttributeError("el pipeline no expone probabilidades ni decision_function")


def roc_auc_value(y_test, scores, clases) -> tuple:
    """AUC binario (una-vs-resto en multiclase).

    Returns:
        `(valor, motivo)`. `valor` es `None` si no se puede calcular y `motivo`
        explica por qué, en español.

    Raises:
        AttributeError: si `scores` no aporta la información necesaria.
    """
    y_test = np.asarray(y_test)
    if len(np.unique(y_test)) < 2:
        return None, AUC_MOTIVO_UNA_CLASE
    try:
        if len(clases) == 2:
            valor = roc_auc_score(y_test, scores[:, 1])
        else:
            valor = roc_auc_score(
                y_test,
                scores,
                multi_class="ovr",
                average="macro",
                labels=list(clases),
            )
    except ValueError as exc:
        return None, f"No se pudo calcular el AUC: {exc}"
    return float(valor), None


def _compute_metrics(task_type, y_test, y_pred, pipe=None, X_test=None):
    """Calcula el conjunto estándar de métricas para una tarea.

    Para clasificación devuelve **siempre** las claves `accuracy`, `f1_macro`,
    `roc_auc`, `auc_disponible`, `auc_motivo` y `n_test`, de forma que la
    interfaz nunca tenga que decidir qué mostrar.

    Parámetros
    ----------
    task_type : {"classification", "regression"}
    y_test : array-like
        Valores reales del conjunto de prueba.
    y_pred : array-like
        Predicciones del modelo.
    pipe : sklearn.pipeline.Pipeline | None
        Pipeline entrenado. Solo necesario para `roc_auc`.
    X_test : array-like | None
        Features de prueba. Solo necesario para `roc_auc`.

    Devuelve
    --------
    dict[str, float | None | bool | str | int]
    """
    if task_type == CLASSIFICATION:
        auc = None
        motivo = AUC_MOTIVO_SIN_DATOS
        if pipe is not None and X_test is not None:
            try:
                scores, clases = _auc_scores(pipe, X_test)
                auc, motivo = roc_auc_value(y_test, scores, clases)
                motivo = motivo if auc is None else None
            except (AttributeError, ValueError, TypeError, IndexError, KeyError):
                auc, motivo = None, AUC_MOTIVO_SIN_PUNTUACIONES
        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
            "roc_auc": auc,
            "auc_disponible": auc is not None,
            "auc_motivo": motivo,
            "n_test": int(len(y_test)),
        }

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
        "n_test": int(len(y_test)),
    }


def summarize_metrics(metrics: dict) -> list:
    """Métricas como líneas legibles en español.

    Solo aparecen las claves conocidas por el catálogo; las de control
    (`auc_disponible`, `auc_motivo`, `n_test`, `best_params`) las interpreta la
    interfaz por su cuenta. Un `roc_auc` a `None` se muestra como `n/d`.
    """
    lineas = []
    for clave in model_specs.METRIC_LABELS:
        if clave not in metrics:
            continue
        valor = metrics[clave]
        etiqueta = model_specs.metric_label(clave)
        if valor is None:
            lineas.append(f"{etiqueta}: n/d")
        elif isinstance(valor, bool):
            lineas.append(f"{etiqueta}: {'sí' if valor else 'no'}")
        elif isinstance(valor, (int, float)):
            lineas.append(f"{etiqueta}: {valor:.4f}")
        else:
            lineas.append(f"{etiqueta}: {valor}")
    return lineas
