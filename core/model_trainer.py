# core/model_trainer.py
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
)

CLASSIFIERS = {
    "Regresión Logística": LogisticRegression(max_iter=1000),
    "Árbol de Decisión": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(),
}

REGRESSORS = {
    "Regresión Lineal": LinearRegression(),
    "Ridge": Ridge(),
    "Árbol de Decisión": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "SVR": SVR(),
    "KNN": KNeighborsRegressor(),
}

PARAM_GRIDS = {
    # --- Clasificación ---
    "Regresión Logística": {
        "model__C": [0.01, 0.1, 1.0, 10.0],
        "model__class_weight": [None, "balanced"],
    },
    "Árbol de Decisión": {
        "model__max_depth": [None, 5, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
    },
    "Random Forest": {
        "model__n_estimators": [100, 200, 400],
        "model__max_depth": [None, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
        "model__max_features": ["sqrt", "log2"],
    },
    "SVM": {
        "model__C": [0.1, 1.0, 10.0],
        "model__kernel": ["rbf", "linear"],
        "model__gamma": ["scale", "auto"],
    },
    "KNN": {
        "model__n_neighbors": [3, 5, 7, 11, 15],
        "model__weights": ["uniform", "distance"],
        "model__p": [1, 2],
    },
    # --- Regresión ---
    "Regresión Lineal": {
        "model__fit_intercept": [True, False],
    },
    "Ridge": {
        "model__alpha": [0.1, 1.0, 10.0, 100.0],
        "model__fit_intercept": [True, False],
    },
    "SVR": {
        "model__C": [0.1, 1.0, 10.0],
        "model__kernel": ["rbf", "linear"],
        "model__epsilon": [0.01, 0.1, 0.5],
    },
}

def build_preprocessor(X):
    num_cols = X.select_dtypes("number").columns.tolist()
    cat_cols = X.select_dtypes(["object", "category", "bool"]).columns.tolist()

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

# core/model_trainer.py

def train_model(df, target, model_name, task_type, test_size=0.2, random_state=42):
    """Entrena un modelo sin búsqueda de hiperparámetros.

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
        stratify=y if task_type == "classification" else None,
    )

    model = (CLASSIFIERS if task_type == "classification" else REGRESSORS)[model_name]
    pipe = Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("model", model),
    ])
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
        stratify=y if task_type == "classification" else None,
    )

    base_model = (CLASSIFIERS if task_type == "classification" else REGRESSORS)[model_name]
    pipe = Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("model", base_model),
    ])

    param_grid = PARAM_GRIDS.get(model_name, {})
    if not param_grid:
        # Sin espacio de búsqueda: entrenamiento normal.
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        metrics = _compute_metrics(task_type, y_test, y_pred)
        return pipe, metrics, (X_test, y_test, y_pred), {}

    scoring = _scoring_for(task_type, metric)
    if scoring == "roc_auc":
        scoring = _resolve_roc_auc(base_model, y)

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