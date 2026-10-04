"""Pruebas de `core/feature_selection.py` (SDD-002)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from core import feature_selection as fs


@pytest.fixture
def dataset_clasificacion():
    """Dos atributos informativos y dieciocho de ruido."""
    rng = np.random.default_rng(0)
    n = 200
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    ruido = rng.normal(size=(n, 18))
    X = pd.DataFrame(
        np.column_stack([x0, x1, ruido]),
        columns=[f"x{i}" for i in range(20)],
    )
    y = pd.Series((x0 + x1 > 0).astype(int), name="objetivo")
    return X, y


@pytest.fixture
def dataset_regresion():
    rng = np.random.default_rng(1)
    n = 150
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    ruido = rng.normal(size=(n, 8))
    X = pd.DataFrame(
        np.column_stack([x0, x1, ruido]), columns=[f"x{i}" for i in range(10)]
    )
    y = pd.Series(3 * x0 - 2 * x1 + 0.1 * ruido[:, 0], name="objetivo")
    return X, y


# ----------------------------------------------------------------------
# T-001 · catálogo
# ----------------------------------------------------------------------
def test_el_catalogo_tiene_los_cuatro_metodos():
    assert list(fs.SELECTION_METHODS) == ["chi2", "anova", "mutual_info", "embedded"]


def test_clasificacion_ofrece_los_cuatro():
    assert fs.available_methods("classification") == [
        "chi2",
        "anova",
        "mutual_info",
        "embedded",
    ]


def test_regresion_descarta_chi2():
    metodos = fs.available_methods("regression")
    assert "chi2" not in metodos
    assert metodos == ["anova", "mutual_info", "embedded"]


def test_chi2_exige_no_negativos():
    assert fs.SELECTION_METHODS["chi2"].requires_non_negative is True


def test_embedded_no_admite_porcentaje():
    assert fs.SELECTION_METHODS["embedded"].supports_percentile is False


def test_method_label_usa_el_etiqueta():
    assert fs.method_label("anova") == "ANOVA F"
    assert fs.method_label("no-existe") == "no-existe"


# ----------------------------------------------------------------------
# T-002 · selectores seguros
# ----------------------------------------------------------------------
def test_safe_select_k_best_recorta_k(dataset_clasificacion):
    X, y = dataset_clasificacion
    selector = fs.SafeSelectKBest(score_func=fs.f_classif, k=10)
    selector.fit(X[["x0", "x1", "x2", "x3"]], y)

    assert selector.k_effective_ == 4
    assert selector.transform(X[["x0", "x1", "x2", "x3"]]).shape[1] == 4


def test_safe_select_k_best_no_deja_k_mutado(dataset_clasificacion):
    X, y = dataset_clasificacion
    selector = fs.SafeSelectKBest(score_func=fs.f_classif, k=10)
    selector.fit(X[["x0", "x1", "x2", "x3"]], y)

    assert selector.k == 10


def test_safe_select_from_model_recorta(dataset_clasificacion):
    X, y = dataset_clasificacion
    estimator = fs._random_forest("classification", 42, n_estimators=5)
    selector = fs.SafeSelectFromModel(estimator=estimator, max_features=9)
    selector.fit(X[["x0", "x1", "x2", "x3"]], y)

    assert selector.max_features_effective_ == 4
    assert selector.n_features_ == 4
    assert selector.max_features == 9


def test_los_selectores_sobreviven_a_pickle(dataset_clasificacion):
    import pickle

    X, y = dataset_clasificacion
    selector = fs.build_selector("anova", "classification", k=2)
    copia = pickle.loads(pickle.dumps(selector))
    copia.fit(X, y)

    assert copia.transform(X).shape[1] == 2


# ----------------------------------------------------------------------
# T-003 · build_selector
# ----------------------------------------------------------------------
@pytest.mark.parametrize("metodo", ["chi2", "anova", "mutual_info"])
@pytest.mark.parametrize("tarea", ["classification", "regression"])
def test_matriz_metodo_por_tarea(
    metodo, tarea, dataset_clasificacion, dataset_regresion
):
    if metodo == "chi2" and tarea == "regression":
        with pytest.raises(ValueError, match="regresión"):
            fs.build_selector(metodo, tarea)
        return

    X, y = dataset_clasificacion if tarea == "classification" else dataset_regresion
    datos = X.to_numpy() - X.to_numpy().min() if metodo == "chi2" else X.to_numpy()
    selector = fs.build_selector(metodo, tarea, k=3, n_estimators=5)
    selector.fit(datos, y)

    assert selector.transform(datos).shape[1] == 3


def test_embedded_devuelve_safe_select_from_model(dataset_clasificacion):
    selector = fs.build_selector("embedded", "classification", k=4, n_estimators=5)
    assert isinstance(selector, fs.SafeSelectFromModel)


def test_porcentaje_devuelve_select_percentile():
    selector = fs.build_selector("anova", "classification", percentile=25)
    assert isinstance(selector, fs.SelectPercentile)
    assert selector.percentile == 25


def test_k_por_defecto_del_catalogo():
    selector = fs.build_selector("anova", "classification")
    assert selector.k == fs.SELECTION_METHODS["anova"].default_k


def test_metodo_desconocido():
    with pytest.raises(ValueError, match="desconocido"):
        fs.build_selector("magia", "classification")


def test_porcentaje_no_soportado():
    with pytest.raises(ValueError, match="porcentaje"):
        fs.build_selector("embedded", "classification", percentile=30)


def test_chi2_falla_con_negativos(dataset_clasificacion):
    X, y = dataset_clasificacion
    selector = fs.build_selector("chi2", "classification", k=2)
    with pytest.raises(ValueError):
        selector.fit(X.to_numpy() - 5, y)


# ----------------------------------------------------------------------
# T-004 · mapa de origen
# ----------------------------------------------------------------------
def _preprocessor_una_hot():
    df = pd.DataFrame({"edad": [20, 30, 40], "ciudad": ["a", "b", "c"]})
    ct = ColumnTransformer(
        [
            ("num", StandardScaler(), ["edad"]),
            ("cat", OneHotEncoder(), ["ciudad"]),
        ]
    ).fit(df)
    return ct, list(ct.get_feature_names_out())


def test_mapa_de_origen_agrupa_el_one_hot():
    ct, nombres = _preprocessor_una_hot()
    origenes = fs.feature_origin_map(ct, nombres)

    assert len(origenes) == 4
    assert origenes == ["edad", "ciudad", "ciudad", "ciudad"]


def test_mapa_de_origen_cae_al_prefijo_si_no_cuadra():
    ct, nombres = _preprocessor_una_hot()
    origenes = fs.feature_origin_map(ct, nombres + ["cat__extra"])

    assert origenes[-1] == "extra"


# ----------------------------------------------------------------------
# T-005 · describe_selection
# ----------------------------------------------------------------------
def _pipe_con_selector(metodo="anova", k=2, tarea="classification"):
    df = pd.DataFrame(
        {
            "edad": [20, 30, 40, 50, 60, 70, 80, 90],
            "ciudad": ["a", "b", "a", "b", "a", "b", "a", "b"],
            "ruido": [1.0, 5.0, 2.0, 9.0, 3.0, 7.0, 4.0, 8.0],
        }
    )
    y = pd.Series([0, 1, 0, 1, 0, 1, 0, 1], name="objetivo")
    ct = ColumnTransformer(
        [
            ("num", StandardScaler(), ["edad", "ruido"]),
            ("cat", OneHotEncoder(), ["ciudad"]),
        ]
    )
    pasos = [("preprocessor", ct)]
    if metodo == "chi2":
        from sklearn.preprocessing import MinMaxScaler

        pasos.append(("scaler_nonneg", MinMaxScaler()))
    pasos.append(("selector", fs.build_selector(metodo, tarea, k=k, n_estimators=5)))
    return Pipeline(pasos)


def test_describe_selection_agrega_por_atributo_original():
    pipe = _pipe_con_selector()
    pipe.fit(
        pd.DataFrame(
            {
                "edad": [20, 30, 40, 50, 60, 70, 80, 90],
                "ciudad": ["a", "b", "a", "b", "a", "b", "a", "b"],
                "ruido": [1.0, 5.0, 2.0, 9.0, 3.0, 7.0, 4.0, 8.0],
            }
        ),
        pd.Series([0, 1, 0, 1, 0, 1, 0, 1]),
    )
    tabla = fs.describe_selection(pipe)

    assert list(tabla.columns) == list(fs.TABLA_SELECCION)
    assert set(tabla["atributo"]) == {"edad", "ruido", "ciudad"}
    ciudad = tabla[tabla["atributo"] == "ciudad"].iloc[0]
    assert ciudad["n_columnas_codificadas"] == 2


def test_describe_selection_ordena_por_puntuacion():
    pipe = _pipe_con_selector()
    X = pd.DataFrame(
        {
            "edad": [20, 30, 40, 50, 60, 70, 80, 90],
            "ciudad": ["a", "b", "a", "b", "a", "b", "a", "b"],
            "ruido": [1.0, 5.0, 2.0, 9.0, 3.0, 7.0, 4.0, 8.0],
        }
    )
    pipe.fit(X, pd.Series([0, 1, 0, 1, 0, 1, 0, 1]))
    tabla = fs.describe_selection(pipe)

    puntuaciones = tabla["puntuacion"].to_numpy()
    assert (puntuaciones[:-1] >= puntuaciones[1:]).all()
    assert tabla.index.tolist() == list(range(len(tabla)))


def test_describe_selection_sin_selector_devuelve_vacio():
    pipe = Pipeline([("preprocessor", StandardScaler())])
    tabla = fs.describe_selection(pipe)

    assert tabla.empty
    assert list(tabla.columns) == list(fs.TABLA_SELECCION)


def test_describe_selection_marca_los_no_seleccionados():
    pipe = _pipe_con_selector(k=1)
    X = pd.DataFrame(
        {
            "edad": [20, 30, 40, 50, 60, 70, 80, 90],
            "ciudad": ["a", "b", "a", "b", "a", "b", "a", "b"],
            "ruido": [1.0, 5.0, 2.0, 9.0, 3.0, 7.0, 4.0, 8.0],
        }
    )
    pipe.fit(X, pd.Series([0, 1, 0, 1, 0, 1, 0, 1]))
    tabla = fs.describe_selection(pipe)

    assert tabla["seleccionado"].sum() >= 1
    assert not tabla["seleccionado"].all()


def test_selected_feature_names(dataset_clasificacion):
    X, y = dataset_clasificacion
    pipe = Pipeline(
        [
            ("preprocessor", StandardScaler()),
            ("selector", fs.build_selector("anova", "classification", k=2)),
        ]
    ).fit(X, y)
    nombres = fs.selected_feature_names(pipe)

    assert len(nombres) == 2
    assert all(n.startswith("x") for n in nombres)


def test_la_seleccion_encuentra_las_columnas_informativas(dataset_clasificacion):
    X, y = dataset_clasificacion
    pipe = Pipeline(
        [
            ("preprocessor", StandardScaler()),
            ("selector", fs.build_selector("anova", "classification", k=2)),
        ]
    ).fit(X, y)
    origen = {n.split("__", 1)[-1] for n in fs.selected_feature_names(pipe)}

    assert origen == {"x0", "x1"}


# ----------------------------------------------------------------------
# T-007 · integración con el pipeline
# ----------------------------------------------------------------------
@pytest.fixture
def df_clasificacion():
    rng = np.random.default_rng(2)
    n = 120
    df = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(5)})
    df["objetivo"] = (df["x0"] + df["x1"] > 0).astype(int)
    return df


def test_build_pipeline_sin_seleccion_no_cambia(df_clasificacion):
    from core import model_trainer

    pipe = model_trainer.build_pipeline(
        df_clasificacion, "Random Forest", "classification", "objetivo"
    )
    assert list(pipe.named_steps) == ["preprocessor", "model"]


def test_build_pipeline_aniade_el_selector(df_clasificacion):
    from core import model_trainer

    pipe = model_trainer.build_pipeline(
        df_clasificacion,
        "Random Forest",
        "classification",
        "objetivo",
        selection={"method": "anova", "k": 3},
    )
    assert list(pipe.named_steps) == ["preprocessor", "selector", "model"]


def test_chi2_aniade_el_escalado_no_negativo(df_clasificacion):
    from core import model_trainer

    pipe = model_trainer.build_pipeline(
        df_clasificacion,
        "Random Forest",
        "classification",
        "objetivo",
        selection={"method": "chi2", "k": 3},
    )
    assert list(pipe.named_steps) == [
        "preprocessor",
        "scaler_nonneg",
        "selector",
        "model",
    ]


def test_el_selector_va_despues_del_balanceo(df_clasificacion):
    from core import model_trainer

    pipe = model_trainer.build_pipeline(
        df_clasificacion,
        "Random Forest",
        "classification",
        "objetivo",
        balancing={"method": "random_under"},
        selection={"method": "anova", "k": 3},
    )
    pasos = list(pipe.named_steps)
    assert pasos.index("balancer") < pasos.index("selector")


def test_train_model_sin_seleccion_deja_el_test_set_intacto(df_clasificacion):
    from core import model_trainer

    _, metrics, (_, y_test, _) = model_trainer.train_model(
        df_clasificacion, "objetivo", "Random Forest", "classification"
    )
    assert metrics["n_test"] == len(y_test) == 24


def test_train_model_con_seleccion_entrena(df_clasificacion):
    from core import model_trainer

    pipe, metrics, _ = model_trainer.train_model(
        df_clasificacion,
        "objetivo",
        "Random Forest",
        "classification",
        selection={"method": "anova", "k": 3},
    )
    assert metrics["accuracy"] is not None
    assert len(fs.selected_feature_names(pipe)) == 3


def test_tune_model_con_seleccion_entrena(df_clasificacion):
    from core import model_trainer

    pipe, metrics, _, _ = model_trainer.tune_model(
        df_clasificacion,
        "objetivo",
        "Random Forest",
        "classification",
        metric="accuracy",
        selection={"method": "anova", "k": 3},
        n_iter=3,
    )
    assert "selector" in pipe.named_steps
    assert metrics["accuracy"] is not None


def test_analyze_selection_devuelve_la_tabla(df_clasificacion):
    from core import model_trainer

    tabla = model_trainer.analyze_selection(
        df_clasificacion,
        "objetivo",
        "Random Forest",
        "classification",
        selection={"method": "anova", "k": 3},
    )
    assert list(tabla.columns) == list(fs.TABLA_SELECCION)
    assert tabla["seleccionado"].sum() >= 1


def test_analyze_selection_sin_metodo_avisa():
    from core import model_trainer

    with pytest.raises(ValueError, match="método de selección"):
        model_trainer.analyze_selection(
            pd.DataFrame({"a": [1, 2], "b": [0, 1]}),
            "b",
            "Random Forest",
            "classification",
        )


# ----------------------------------------------------------------------
# T-008 · estado y persistencia
# ----------------------------------------------------------------------
def test_app_state_guarda_la_seleccion():
    from core.state import AppState

    state = AppState()
    assert state.selection_method is None
    assert state.selection_criteria is None

    state.selection_method = "anova"
    state.selection_criteria = {"k": 5}
    state.reset_data()
    assert state.selection_method is None
    assert state.selection_criteria is None


def test_bundle_metadata_guarda_la_seleccion(df_clasificacion):
    from core import model_trainer, persistence

    pipe, metrics, _ = model_trainer.train_model(
        df_clasificacion,
        "objetivo",
        "Random Forest",
        "classification",
        selection={"method": "anova", "k": 3},
    )
    metadata = persistence.build_metadata(
        pipe,
        target_column="objetivo",
        task_type="classification",
        model_name="Random Forest",
        metrics=metrics,
        df=df_clasificacion,
        selection={"method": "anova", "k": 3},
    )

    assert metadata.selection["method"] == "anova"
    assert metadata.selection["criterion"] == {"k": 3}
    assert len(metadata.selection["selected"]) == 3
    assert "Selección: ANOVA F" in metadata.describe()


def test_bundle_sin_seleccion_no_inventa_nada(df_clasificacion):
    from core import model_trainer, persistence

    pipe, metrics, _ = model_trainer.train_model(
        df_clasificacion, "objetivo", "Random Forest", "classification"
    )
    metadata = persistence.build_metadata(
        pipe,
        target_column="objetivo",
        task_type="classification",
        model_name="Random Forest",
        metrics=metrics,
        df=df_clasificacion,
    )

    assert metadata.selection == {}
    assert "Selección" not in metadata.describe()


def test_un_bundle_antiguo_sin_la_clave_seleccion_sigue_cargando(
    df_clasificacion, tmp_path
):
    """Un `.automl` de antes de la spec no tiene `selection` en `__dict__`."""
    from dataclasses import replace

    from core import model_trainer, persistence

    pipe, metrics, test_data = model_trainer.train_model(
        df_clasificacion,
        "objetivo",
        "Random Forest",
        "classification",
        selection={"method": "anova", "k": 3},
    )
    metadata = persistence.build_metadata(
        pipe,
        target_column="objetivo",
        task_type="classification",
        model_name="Random Forest",
        metrics=metrics,
        df=df_clasificacion,
        selection={"method": "anova", "k": 3},
    )
    antiguo = replace(metadata)
    del antiguo.__dict__["selection"]

    ruta = tmp_path / "viejo.automl"
    # Se simula un bundle anterior a la spec: los metadatos se guardan sin la
    # clave `selection`, como si el campo no existiera todavía.
    persistence.save_bundle(ruta, pipe, metadata, dataset=df_clasificacion)
    import joblib

    payload = joblib.load(ruta)
    payload["metadata"] = antiguo
    joblib.dump(payload, ruta)

    cargado = persistence.load_bundle(ruta)
    assert "Selección" not in cargado.metadata.describe()


def test_selection_summary_para_la_interfaz(df_clasificacion):
    from core import model_trainer, persistence

    pipe, _, _ = model_trainer.train_model(
        df_clasificacion,
        "objetivo",
        "Random Forest",
        "classification",
        selection={"method": "embedded", "k": 3},
    )
    linea = persistence.selection_summary(pipe, {"method": "embedded", "k": 3})

    assert "Selección: Embebido" in linea
    assert "3 atributos" in linea
