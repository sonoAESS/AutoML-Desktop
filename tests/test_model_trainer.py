# tests/test_model_trainer.py
import pandas as pd
from sklearn.datasets import load_iris
from core import model_specs, model_trainer


def test_tune_random_forest_iris():
    data = load_iris(as_frame=True)
    df = data.frame.rename(columns={"target": "y"})
    pipe, metrics, _, cv = model_trainer.tune_model(
        df, target="y", model_name="Random Forest",
        task_type="classification", search_type="random",
        metric="accuracy", cv=3, n_iter=5,
    )
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert "best_params" in metrics
    assert cv  # cv_results no está vacío


def test_train_and_tune_comparten_metricas():
    data = load_iris(as_frame=True)
    df = data.frame.rename(columns={"target": "y"})

    # Camino normal
    _, metrics_plain, _ = model_trainer.train_model(
        df, target="y", model_name="Regresión Logística",
        task_type="classification",
    )

    # Camino con tuning (grid pequeño para que sea rápido)
    _, metrics_tuned, _, _ = model_trainer.tune_model(
        df, target="y", model_name="Regresión Logística",
        task_type="classification", search_type="grid",
        metric="accuracy", cv=3,
    )

    # Ambos deben compartir las mismas claves base
    claves_comunes = {"accuracy", "f1_macro"}
    assert claves_comunes.issubset(metrics_plain.keys())
    assert claves_comunes.issubset(metrics_tuned.keys())


def test_tune_clasificacion_multiclase_no_deja_scores_nan():
    data = load_iris(as_frame=True)
    df = data.frame.rename(columns={"target": "y"})

    _, metrics, _, cv = model_trainer.tune_model(
        df, target="y", model_name="Regresión Logística",
        task_type="classification", search_type="grid",
        metric="accuracy", cv=3,
    )

    assert metrics["best_cv_score"] == metrics["best_cv_score"]
    assert cv["mean_test_score"].size > 0


def test_tune_roc_auc_multiclase_y_sin_predict_proba():
    data = load_iris(as_frame=True)
    df = data.frame.rename(columns={"target": "y"})

    for model_name in model_trainer.CLASSIFIERS:
        _, metrics, _, _ = model_trainer.tune_model(
            df, target="y", model_name=model_name,
            task_type="classification", search_type="grid",
            metric="roc_auc", cv=3,
        )
        assert metrics["best_cv_score"] == metrics["best_cv_score"], model_name


def test_resolve_roc_auc():
    df = pd.DataFrame({"y": [0, 1]})
    assert model_trainer._resolve_roc_auc(
        model_trainer.CLASSIFIERS["Regresión Logística"], df["y"]
    ) == "roc_auc"

    df = pd.DataFrame({"y": [0, 1, 2]})
    assert model_trainer._resolve_roc_auc(
        model_trainer.CLASSIFIERS["Regresión Logística"], df["y"]
    ) == "roc_auc_ovr"

    assert model_trainer._resolve_roc_auc(
        model_trainer.REGRESSORS["Regresión Lineal"], df["y"]
    ) == "accuracy"


def test_transform_dataset_aplica_tipos_y_normalizacion():
    df = pd.DataFrame({
        "edad": ["20", "30", "40", "50", "60", "70", "80", "90", "100", "110"],
        "y": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })
    out = model_trainer.transform_dataset(
        df,
        casts={"edad": "numerico"},
        normalizations={"edad": "minmax"},
    )
    assert out["edad"].dtype.kind in "if"
    assert out["edad"].min() == 0.0
    assert out["edad"].max() == 1.0


def test_el_pipeline_incluye_los_pasos_de_transformacion():
    df = pd.DataFrame({
        "edad": ["20", "30", "40", "50", "60", "70", "80", "90", "100", "110"],
        "y": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })
    pipe, _, _ = model_trainer.train_model(
        df, target="y", model_name="Regresión Logística",
        task_type="classification",
        casts={"edad": "numerico"},
        normalizations={"edad": "standard"},
    )
    pasos = list(pipe.named_steps)
    assert pasos[:3] == ["typer", "normalizer", "preprocessor"]


def test_el_pipeline_predice_sobre_datos_crudos():
    df = pd.DataFrame({
        "ciudad": ["a", "b", "a", "b"] * 10,
        "importe": [10.0, 20.0, 30.0, 40.0] * 10,
        "y": [0, 1, 0, 1] * 10,
    })
    pipe, _, _ = model_trainer.train_model(
        df, target="y", model_name="Random Forest", task_type="classification",
        casts={"importe": "numerico"}, normalizations={"importe": "minmax"},
    )
    nuevos = pd.DataFrame({"importe": ["999999"], "ciudad": ["a"]})
    assert len(pipe.predict(nuevos)) == 1


def test_search_size_cuenta_combinaciones_y_ajustes():
    combos, ajustes = model_trainer.search_size(
        "Random Forest", "classification", cv=5,
    )
    grid = model_trainer.search_grid("Random Forest", "classification")
    assert combos == model_trainer._grid_size(grid)
    assert ajustes == combos * 5
    assert combos > 1


def test_search_size_respeta_las_selecciones():
    combos, _ = model_trainer.search_size(
        "Random Forest", "classification",
        selections={"max_depth": [3, 5, 7], "min_samples_split": 2},
    )
    grid = model_trainer.search_grid(
        "Random Forest", "classification",
        selections={"max_depth": [3, 5, 7], "min_samples_split": 2},
    )
    assert combos == model_trainer._grid_size(grid)


def test_search_size_limita_las_iteraciones_de_la_busqueda_aleatoria():
    combos, ajustes = model_trainer.search_size(
        "Random Forest", "classification", search_type="random", cv=3, n_iter=7,
    )
    assert combos == 7
    assert ajustes == 21

    specs = model_specs.get_spec("Random Forest", "classification")
    fijados = {p.name: p.values[0] for p in specs.params}
    fijados["max_depth"] = [1, 2]

    combos, ajustes = model_trainer.search_size(
        "Random Forest", "classification", selections=fijados,
        search_type="random", cv=3, n_iter=50,
    )
    assert combos == 2
    assert ajustes == 6


def test_search_size_con_un_solo_valor_por_parametro():
    specs = model_specs.get_spec("Random Forest", "classification")
    fijados = {p.name: p.values[0] for p in specs.params}
    assert model_trainer.search_size(
        "Random Forest", "classification", selections=fijados, cv=4,
    ) == (1, 4)


def test_search_size_sin_parametros_devuelve_cero():
    specs = model_specs.get_spec("Random Forest", "classification")
    sin_parametros = model_specs.ModelSpec(
        name=specs.name, family=specs.family, task_type=specs.task_type,
        factory=specs.factory,
    )
    original = model_specs.SPECS["classification"]["Random Forest"]
    model_specs.SPECS["classification"]["Random Forest"] = sin_parametros
    try:
        assert model_trainer.search_size(
            "Random Forest", "classification",
        ) == (0, 0)
    finally:
        model_specs.SPECS["classification"]["Random Forest"] = original


def test_tune_model_sin_espacio_de_busqueda_entrena_directo(monkeypatch):
    monkeypatch.setattr(model_trainer, "search_grid", lambda *a, **k: {})
    df = pd.DataFrame({
        "a": list(range(10)),
        "y": [0, 1] * 5,
    })
    etapas = []
    pipe, metrics, test_data, cv_results = model_trainer.tune_model(
        df, target="y", model_name="Random Forest",
        task_type="classification", cv=2,
        progress_callback=lambda e, a, t: etapas.append(e),
    )
    assert cv_results == {}
    assert "accuracy" in metrics
    assert etapas == ["Preparando los datos y el pipeline…", "Entrenando el modelo…"]


def test_tune_model_informa_de_las_etapas():
    eventos = []
    df = pd.DataFrame({
        "a": list(range(10)),
        "b": [1.0, 0.0] * 5,
        "y": [0, 1] * 5,
    })
    model_trainer.tune_model(
        df, target="y", model_name="Árbol de Decisión",
        task_type="classification", search_type="random", cv=2, n_iter=3,
        progress_callback=lambda etapa, actual, total: eventos.append(
            (etapa, actual, total)
        ),
    )
    textos = [etapa for etapa, _, _ in eventos]
    assert any("Explorando" in t for t in textos)
    assert any("métricas finales" in t for t in textos)

    total = eventos[[i for i, t in enumerate(textos)
                     if "Explorando" in t][0]][2]
    assert total == 6  # 3 combinaciones x 2 folds
    assert eventos[0][1] == 0


def test_tune_model_sin_callback_no_falla():
    df = pd.DataFrame({
        "a": list(range(10)),
        "y": [0, 1] * 5,
    })
    pipe, metrics, _, _ = model_trainer.tune_model(
        df, target="y", model_name="Árbol de Decisión",
        task_type="classification", cv=2,
        progress_callback=None,
    )
    assert "accuracy" in metrics
