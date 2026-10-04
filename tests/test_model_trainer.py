# tests/test_model_trainer.py
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline

from core import model_specs, model_trainer


def test_tune_random_forest_iris():
    data = load_iris(as_frame=True)
    df = data.frame.rename(columns={"target": "y"})
    pipe, metrics, _, cv = model_trainer.tune_model(
        df,
        target="y",
        model_name="Random Forest",
        task_type="classification",
        search_type="random",
        metric="accuracy",
        cv=3,
        n_iter=5,
    )
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert "best_params" in metrics
    assert cv  # cv_results no está vacío


def test_train_and_tune_comparten_metricas():
    data = load_iris(as_frame=True)
    df = data.frame.rename(columns={"target": "y"})

    # Camino normal
    _, metrics_plain, _ = model_trainer.train_model(
        df,
        target="y",
        model_name="Regresión Logística",
        task_type="classification",
    )

    # Camino con tuning (grid pequeño para que sea rápido)
    _, metrics_tuned, _, _ = model_trainer.tune_model(
        df,
        target="y",
        model_name="Regresión Logística",
        task_type="classification",
        search_type="grid",
        metric="accuracy",
        cv=3,
    )

    # Ambos deben compartir las mismas claves base
    claves_comunes = {"accuracy", "f1_macro"}
    assert claves_comunes.issubset(metrics_plain.keys())
    assert claves_comunes.issubset(metrics_tuned.keys())


def test_tune_clasificacion_multiclase_no_deja_scores_nan():
    data = load_iris(as_frame=True)
    df = data.frame.rename(columns={"target": "y"})

    _, metrics, _, cv = model_trainer.tune_model(
        df,
        target="y",
        model_name="Regresión Logística",
        task_type="classification",
        search_type="grid",
        metric="accuracy",
        cv=3,
    )

    assert metrics["best_cv_score"] == metrics["best_cv_score"]
    assert cv["mean_test_score"].size > 0


def test_tune_roc_auc_multiclase_y_sin_predict_proba():
    data = load_iris(as_frame=True)
    df = data.frame.rename(columns={"target": "y"})

    for model_name in model_trainer.CLASSIFIERS:
        _, metrics, _, _ = model_trainer.tune_model(
            df,
            target="y",
            model_name=model_name,
            task_type="classification",
            search_type="grid",
            metric="roc_auc",
            cv=3,
        )
        assert metrics["best_cv_score"] == metrics["best_cv_score"], model_name


def test_resolve_roc_auc():
    df = pd.DataFrame({"y": [0, 1]})
    assert (
        model_trainer._resolve_roc_auc(
            model_trainer.CLASSIFIERS["Regresión Logística"], df["y"]
        )
        == "roc_auc"
    )

    df = pd.DataFrame({"y": [0, 1, 2]})
    assert (
        model_trainer._resolve_roc_auc(
            model_trainer.CLASSIFIERS["Regresión Logística"], df["y"]
        )
        == "roc_auc_ovr"
    )

    assert (
        model_trainer._resolve_roc_auc(
            model_trainer.REGRESSORS["Regresión Lineal"], df["y"]
        )
        == "accuracy"
    )


def test_transform_dataset_aplica_tipos_y_normalizacion():
    df = pd.DataFrame(
        {
            "edad": ["20", "30", "40", "50", "60", "70", "80", "90", "100", "110"],
            "y": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    out = model_trainer.transform_dataset(
        df,
        casts={"edad": "numerico"},
        normalizations={"edad": "minmax"},
    )
    assert out["edad"].dtype.kind in "if"
    assert out["edad"].min() == 0.0
    assert out["edad"].max() == 1.0


def test_el_pipeline_incluye_los_pasos_de_transformacion():
    df = pd.DataFrame(
        {
            "edad": ["20", "30", "40", "50", "60", "70", "80", "90", "100", "110"],
            "y": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    pipe, _, _ = model_trainer.train_model(
        df,
        target="y",
        model_name="Regresión Logística",
        task_type="classification",
        casts={"edad": "numerico"},
        normalizations={"edad": "standard"},
    )
    pasos = list(pipe.named_steps)
    assert pasos[:3] == ["typer", "normalizer", "preprocessor"]


def test_el_pipeline_predice_sobre_datos_crudos():
    df = pd.DataFrame(
        {
            "ciudad": ["a", "b", "a", "b"] * 10,
            "importe": [10.0, 20.0, 30.0, 40.0] * 10,
            "y": [0, 1, 0, 1] * 10,
        }
    )
    pipe, _, _ = model_trainer.train_model(
        df,
        target="y",
        model_name="Random Forest",
        task_type="classification",
        casts={"importe": "numerico"},
        normalizations={"importe": "minmax"},
    )
    nuevos = pd.DataFrame({"importe": ["999999"], "ciudad": ["a"]})
    assert len(pipe.predict(nuevos)) == 1


def test_search_size_cuenta_combinaciones_y_ajustes():
    combos, ajustes = model_trainer.search_size(
        "Random Forest",
        "classification",
        cv=5,
    )
    grid = model_trainer.search_grid("Random Forest", "classification")
    assert combos == model_trainer._grid_size(grid)
    assert ajustes == combos * 5
    assert combos > 1


def test_search_size_respeta_las_selecciones():
    combos, _ = model_trainer.search_size(
        "Random Forest",
        "classification",
        selections={"max_depth": [3, 5, 7], "min_samples_split": 2},
    )
    grid = model_trainer.search_grid(
        "Random Forest",
        "classification",
        selections={"max_depth": [3, 5, 7], "min_samples_split": 2},
    )
    assert combos == model_trainer._grid_size(grid)


def test_search_size_limita_las_iteraciones_de_la_busqueda_aleatoria():
    combos, ajustes = model_trainer.search_size(
        "Random Forest",
        "classification",
        search_type="random",
        cv=3,
        n_iter=7,
    )
    assert combos == 7
    assert ajustes == 21

    specs = model_specs.get_spec("Random Forest", "classification")
    fijados = {p.name: p.values[0] for p in specs.params}
    fijados["max_depth"] = [1, 2]

    combos, ajustes = model_trainer.search_size(
        "Random Forest",
        "classification",
        selections=fijados,
        search_type="random",
        cv=3,
        n_iter=50,
    )
    assert combos == 2
    assert ajustes == 6


def test_search_size_con_un_solo_valor_por_parametro():
    specs = model_specs.get_spec("Random Forest", "classification")
    fijados = {p.name: p.values[0] for p in specs.params}
    assert model_trainer.search_size(
        "Random Forest",
        "classification",
        selections=fijados,
        cv=4,
    ) == (1, 4)


def test_search_size_sin_parametros_devuelve_cero():
    specs = model_specs.get_spec("Random Forest", "classification")
    sin_parametros = model_specs.ModelSpec(
        name=specs.name,
        family=specs.family,
        task_type=specs.task_type,
        factory=specs.factory,
    )
    original = model_specs.SPECS["classification"]["Random Forest"]
    model_specs.SPECS["classification"]["Random Forest"] = sin_parametros
    try:
        assert model_trainer.search_size(
            "Random Forest",
            "classification",
        ) == (0, 0)
    finally:
        model_specs.SPECS["classification"]["Random Forest"] = original


def test_tune_model_sin_espacio_de_busqueda_entrena_directo(monkeypatch):
    monkeypatch.setattr(model_trainer, "search_grid", lambda *a, **k: {})
    df = pd.DataFrame(
        {
            "a": list(range(10)),
            "y": [0, 1] * 5,
        }
    )
    etapas = []
    pipe, metrics, test_data, cv_results = model_trainer.tune_model(
        df,
        target="y",
        model_name="Random Forest",
        task_type="classification",
        cv=2,
        progress_callback=lambda e, a, t: etapas.append(e),
    )
    assert cv_results == {}
    assert "accuracy" in metrics
    assert etapas == ["Preparando los datos y el pipeline…", "Entrenando el modelo…"]


def test_tune_model_informa_de_las_etapas():
    eventos = []
    df = pd.DataFrame(
        {
            "a": list(range(10)),
            "b": [1.0, 0.0] * 5,
            "y": [0, 1] * 5,
        }
    )
    model_trainer.tune_model(
        df,
        target="y",
        model_name="Árbol de Decisión",
        task_type="classification",
        search_type="random",
        cv=2,
        n_iter=3,
        progress_callback=lambda etapa, actual, total: eventos.append(
            (etapa, actual, total)
        ),
    )
    textos = [etapa for etapa, _, _ in eventos]
    assert any("Explorando" in t for t in textos)
    assert any("métricas finales" in t for t in textos)

    total = eventos[[i for i, t in enumerate(textos) if "Explorando" in t][0]][2]
    assert total == 6  # 3 combinaciones x 2 folds
    assert eventos[0][1] == 0


def test_tune_model_sin_callback_no_falla():
    df = pd.DataFrame(
        {
            "a": list(range(10)),
            "y": [0, 1] * 5,
        }
    )
    pipe, metrics, _, _ = model_trainer.tune_model(
        df,
        target="y",
        model_name="Árbol de Decisión",
        task_type="classification",
        cv=2,
        progress_callback=None,
    )
    assert "accuracy" in metrics


# ---------------------------------------------------------------------------
# SDD-003: contrato de métricas de clasificación
# ---------------------------------------------------------------------------

CLAVES_CLASIFICACION = {
    "accuracy",
    "f1_macro",
    "roc_auc",
    "auc_disponible",
    "auc_motivo",
    "n_test",
}


def _iris():
    data = load_iris(as_frame=True)
    return data.frame.rename(columns={"target": "y"})


def test_metricas_de_clasificacion_tienen_siempre_las_mismas_claves():
    _, metrics, _ = model_trainer.train_model(
        _iris(),
        target="y",
        model_name="Regresión Logística",
        task_type="classification",
    )

    assert CLAVES_CLASIFICACION <= set(metrics)
    assert isinstance(metrics["accuracy"], float)
    assert isinstance(metrics["f1_macro"], float)
    assert metrics["auc_disponible"] is True
    assert metrics["auc_motivo"] is None
    assert metrics["n_test"] > 0


def test_auc_multiclase_no_es_none():
    _, metrics, _ = model_trainer.train_model(
        _iris(),
        target="y",
        model_name="Random Forest",
        task_type="classification",
    )

    assert metrics["roc_auc"] is not None
    assert 0.0 <= metrics["roc_auc"] <= 1.0


def test_metricas_de_regresion_no_incluyen_auc():
    from sklearn.datasets import load_diabetes

    data = load_diabetes(as_frame=True)
    df = data.frame.rename(columns={"target": "y"})
    _, metrics, _ = model_trainer.train_model(
        df,
        target="y",
        model_name="Regresión Lineal",
        task_type="regression",
    )

    assert set(metrics) == {"rmse", "mae", "r2", "n_test"}
    assert "auc_disponible" not in metrics


def test_auc_ausente_delve_motivo_en_espanol():
    class SinPuntuaciones:
        """Doble sin `predict_proba` ni `decision_function`."""

        def __init__(self):
            self.classes_ = [0, 1]

        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    pipe = Pipeline([("model", SinPuntuaciones())])
    y_test = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 1]

    metrics = model_trainer._compute_metrics(
        "classification",
        y_test,
        y_pred,
        pipe=pipe,
        X_test=np.zeros((4, 2)),
    )

    assert metrics["roc_auc"] is None
    assert metrics["auc_disponible"] is False
    assert "probabilidades" in metrics["auc_motivo"]
    assert metrics["n_test"] == 4


def test_auc_sin_datos_de_prueba_explica_el_motivo():
    metrics = model_trainer._compute_metrics(
        "classification",
        [0, 1],
        [0, 1],
    )

    assert metrics["roc_auc"] is None
    assert metrics["auc_disponible"] is False
    assert metrics["auc_motivo"] == model_trainer.AUC_MOTIVO_SIN_DATOS


def test_auc_con_una_sola_clase_explica_el_motivo():
    y_test = np.zeros(10, dtype=int)
    y_pred = np.zeros(10, dtype=int)
    scores = np.column_stack([np.full(10, 0.9), np.full(10, 0.1)])

    valor, motivo = model_trainer.roc_auc_value(y_test, scores, [0, 1])

    assert valor is None
    assert motivo == model_trainer.AUC_MOTIVO_UNA_CLASE


def test_roc_auc_value_usa_one_vs_rest_en_multiclase():
    y_test = np.array([0, 1, 2, 0, 1, 2])
    scores = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
            [0.7, 0.2, 0.1],
            [0.2, 0.7, 0.1],
            [0.2, 0.1, 0.7],
        ]
    )

    valor, motivo = model_trainer.roc_auc_value(y_test, scores, [0, 1, 2])

    assert motivo is None
    assert valor == pytest.approx(1.0)


def test_tune_model_calcula_el_auc_sin_espacio_de_busqueda():
    _, metrics, _, _ = model_trainer.tune_model(
        _iris(),
        target="y",
        model_name="Regresión Logística",
        task_type="classification",
        cv=3,
    )

    assert metrics["roc_auc"] is not None
    assert metrics["auc_disponible"] is True


def test_summarize_metrics_omite_las_claves_de_control():
    _, metrics, _ = model_trainer.train_model(
        _iris(),
        target="y",
        model_name="Regresión Logística",
        task_type="classification",
    )
    metrics["auc_disponible"] = True

    lineas = model_trainer.summarize_metrics(metrics)
    texto = "\n".join(lineas)

    assert "Exactitud: " in texto
    assert "AUC (ROC One-vs-Rest): " in texto
    assert "auc_disponible" not in texto
    assert "n_test" not in texto


def test_summarize_metrics_muestra_nd_sin_auc():
    lineas = model_trainer.summarize_metrics(
        {
            "accuracy": 0.9,
            "f1_macro": 0.8,
            "roc_auc": None,
            "auc_disponible": False,
            "auc_motivo": "sin probabilidades",
        }
    )

    texto = "\n".join(lineas)
    assert "AUC (ROC One-vs-Rest): n/d" in texto
    assert "Exactitud: 0.9000" in texto


def test_summarize_metrics_no_imprime_las_claves_de_control():
    """Un `bool` es un `int` en Python: no debe verse como `1.0000`.

    Las claves de control (`auc_disponible`, `auc_motivo`, `n_test`) las
    interpreta la interfaz por su cuenta, así que no salen en el resumen.
    """
    lineas = model_trainer.summarize_metrics(
        {
            "auc_disponible": True,
            "auc_motivo": "motivo",
            "n_test": 42,
            "best_params": {"C": 1.0},
        }
    )

    assert lineas == []


def test_available_metrics_no_filtra_por_modelo():
    for modelo in ("Regresión Logística", "K Vecinos Más Cercanos"):
        assert model_specs.available_metrics("classification", modelo) == [
            "accuracy",
            "f1_macro",
            "roc_auc",
        ]
    assert model_specs.available_metrics("regression") == ["rmse", "mae", "r2"]


def test_metrics_for_ranking():
    assert model_specs.metrics_for_ranking("classification") == "f1_macro"
    assert model_specs.metrics_for_ranking("regression") == "r2"


def test_metric_label():
    assert model_specs.metric_label("roc_auc") == "AUC (ROC One-vs-Rest)"
    assert model_specs.metric_label("r2") == "R²"
    assert model_specs.metric_label("inventada") == "inventada"
