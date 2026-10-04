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
    accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
    r2_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from core import model_specs
from core.model_specs import CLASSIFICATION, REGRESSION
from core.preprocessor import (
    ColumnNormalizer, ColumnTyper, cast_columns, categorical_columns,
    normalization_plan, normalize_columns, numeric_columns,
)

#: Modelos por tarea (instancias de referência, se mantienen por compatibilidad).
CLASSIFIERS = {
    name: spec.build() for name, spec in model_specs.CLASSIFICATION_SPECS.items()
}
REGRESSORS = {
    name: spec.build() for name, spec in model_specs.REGRESSION_SPECS.items()
}

#: Espacio de búsqueda por defecto de cada modelo.
PARAM_GRIDS = {
    name: spec.grid() for name, spec in model_specs.iter_specs()
}


def search_grid(model_name, task_type, selections=None):
    """Grid de búsqueda de un modelo según lo elegido en la interfaz."""
    return model_specs.get_spec(model_name, task_type).grid(selections)


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

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])


def build_pipeline(df, model_name, task_type, target, casts=None, normalizations=None):
    """Construye el pipeline (tipos + normalización + preprocesado + modelo).

    Los pasos `ColumnTyper` y `ColumnNormalizer` son los mismos que se aplican
    al dataset para mostrarlo y exportarlo, de modo que un modelo guardado
    reproduce exactamente la transformación sobre datos nuevos.
    """
    X = df.drop(columns=[target])
    steps = []
    if casts:
        steps.append(("typer", ColumnTyper(casts)))
    if normalizations:
        steps.append(("normalizer", ColumnNormalizer(normalizations)))
    steps.append(("preprocessor", build_preprocessor(X)))
    steps.append(("model", model_specs.build_estimator(model_name, task_type)))
    return Pipeline(steps)


def transform_dataset(df, casts=None, normalizations=None):
    """Aplica al dataset los mismos pasos previos del pipeline.

    Sirve para obtener el dataset transformado que se muestra y se guarda
    junto al modelo.
    """
    out = df
    if casts:
        out = cast_columns(out, casts)
    for method, columns in normalization_plan(normalizations).items():
        out = normalize_columns(out, columns, method)
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
):
    """Entrena un modelo sin búsqueda de hiperparámetros.

    Parámetros
    ----------
    casts : dict | None
        Tipos forzados por columna, `{columna: tipo}`.
    normalizations : dict | None
        Normalización por columna, `{columna: método}`.

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
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if task_type == CLASSIFICATION else None,
    )

    pipe = build_pipeline(
        df, model_name, task_type, target, casts=casts, normalizations=normalizations
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
    progress_callback : callable | None
        Función que se invoca con (combinación_actual, total) si el
        backend lo permite. Se usa desde la UI para reportar progreso.

    Devuelve
    --------
    best_pipe, metrics, (X_test, y_test, y_pred), cv_results
    """
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if task_type == CLASSIFICATION else None,
    )

    pipe = build_pipeline(
        df, model_name, task_type, target, casts=casts, normalizations=normalizations
    )

    param_grid = search_grid(model_name, task_type, selections)
    if not param_grid:
        # Sin espacio de búsqueda: entrenamiento normal.
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        metrics = _compute_metrics(task_type, y_test, y_pred)
        return pipe, metrics, (X_test, y_test, y_pred), {}

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

    best_pipe = search.best_estimator_
    y_pred = best_pipe.predict(X_test)

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


def _compute_metrics(task_type, y_test, y_pred, pipe=None, X_test=None):
    """Calcula el conjunto estándar de métricas para una tarea.

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
    dict[str, float]
    """
    if task_type == "classification":
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
        }
        if pipe is not None and X_test is not None:
            try:
                model = pipe.named_steps.get("model")
                if model is not None and hasattr(model, "predict_proba"):
                    proba = pipe.predict_proba(X_test)
                    if proba.shape[1] == 2:
                        metrics["roc_auc"] = roc_auc_score(y_test, proba[:, 1])
            except Exception:
                # Si falla, simplemente no añadimos roc_auc.
                pass
        return metrics

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": mean_absolute_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }