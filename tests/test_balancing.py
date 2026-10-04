# tests/test_balancing.py
"""Pruebas del núcleo de balanceo de clases (SDD-004)."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from core import balancing, model_specs, model_trainer

# ---------------------------------------------------------------------------
# class_distribution
# ---------------------------------------------------------------------------


def test_distribution_ordena_de_mayor_a_menor():
    d = balancing.class_distribution(pd.Series(["a"] * 90 + ["b"] * 10))

    assert [r.clase for r in d.rows] == ["a", "b"]
    assert d.majority == "a" and d.minority == "b"
    assert d.majority_n == 90 and d.minority_n == 10
    assert d.imbalance_ratio == pytest.approx(9.0)
    assert d.total == 100 and d.n_classes == 2
    assert d.rows[0].pct == pytest.approx(0.9)


def test_distribution_empate_ordena_por_texto_de_clase():
    d = balancing.class_distribution(pd.Series(["b"] * 5 + ["a"] * 5))

    assert [r.clase for r in d.rows] == ["a", "b"]


def test_distribution_niveles_de_desbalance():
    assert balancing.nivel_de_desbalance(1.0) == "equilibrado"
    assert balancing.nivel_de_desbalance(1.4) == "equilibrado"
    assert balancing.nivel_de_desbalance(1.5) == "moderado"
    assert balancing.nivel_de_desbalance(2.9) == "moderado"
    assert balancing.nivel_de_desbalance(3.0) == "severo"
    assert balancing.nivel_de_desbalance(9.0) == "severo"


def test_distribution_entropia_normalizada():
    equilibrada = balancing.class_distribution(pd.Series(["a"] * 50 + ["b"] * 50))
    sesgada = balancing.class_distribution(pd.Series(["a"] * 90 + ["b"] * 10))

    assert equilibrada.normalized_entropy == pytest.approx(1.0)
    assert 0.0 < sesgada.normalized_entropy < 1.0
    assert equilibrada.shannon_entropy > sesgada.shannon_entropy


def test_distribution_una_sola_clase():
    d = balancing.class_distribution(pd.Series([1, 1, 1]))

    assert d.n_classes == 1
    assert d.imbalance_ratio == 1.0
    assert d.normalized_entropy == 0.0
    assert d.majority == 1 and d.minority == 1
    assert d.is_imbalanced is False


def test_distribution_vacia():
    d = balancing.class_distribution(pd.Series([], dtype=float))

    assert d.total == 0 and d.n_classes == 0
    assert d.rows == ()
    assert "0 instancias" in d.describe()


def test_distribution_ignora_nulos():
    d = balancing.class_distribution(pd.Series(["a"] * 8 + ["b"] * 2 + [None]))

    assert d.total == 10


def test_distribution_con_nulos_de_verdad():
    d = balancing.class_distribution(pd.Series(["a"] * 8 + ["b"] * 2 + [None]))

    assert d.total == 10


def test_distribution_describe_en_espanol():
    d = balancing.class_distribution(pd.Series(["a"] * 90 + ["b"] * 10))

    texto = d.describe()
    assert "2 clases" in texto
    assert "100 instancias" in texto
    assert "IR = 9.00" in texto
    assert "severo" in texto
    assert "entropía normalizada" in texto


def test_distribution_as_dict_es_serializable():
    d = balancing.class_distribution(pd.Series(["a"] * 3 + ["b"] * 1))
    datos = d.as_dict()

    assert datos["rows"][0]["clase"] == "a"
    assert datos["rows"][0]["n"] == 3
    assert isinstance(datos["rows"], (list, tuple))


def test_distribution_is_imbalanced():
    assert balancing.class_distribution(
        pd.Series(["a"] * 90 + ["b"] * 10)
    ).is_imbalanced
    assert not balancing.class_distribution(
        pd.Series(["a"] * 50 + ["b"] * 50)
    ).is_imbalanced


# ---------------------------------------------------------------------------
# available_strategies
# ---------------------------------------------------------------------------


def test_estrategias_en_regresion_solo_none():
    assert balancing.available_strategies("regression") == ["none"]


def test_estrategias_en_clasificacion_son_las_seis():
    assert balancing.available_strategies("classification") == [
        "none",
        "class_weight",
        "random_under",
        "random_over",
        "smote",
        "smoten",
    ]


def test_estrategias_quita_class_weight_si_el_modelo_no_lo_admite():
    assert "class_weight" not in balancing.available_strategies("classification", "KNN")
    assert "class_weight" in balancing.available_strategies(
        "classification", "Random Forest"
    )


def test_estrategias_con_modelo_desconocido_no_fallan():
    assert "class_weight" in balancing.available_strategies(
        "classification", "Modelo Inventado"
    )


def test_catalogo_de_estrategias_tiene_etiquetas_en_espanol():
    for estrategia in balancing.BALANCING_STRATEGIES.values():
        assert estrategia.label
        assert estrategia.description
        assert estrategia.id in balancing.STRATEGY_ORDER


# ---------------------------------------------------------------------------
# class_weight_path
# ---------------------------------------------------------------------------


def test_class_weight_path_por_modelo():
    assert (
        model_specs.class_weight_path("Regresión Logística", "classification")
        == "model__class_weight"
    )
    assert (
        model_specs.class_weight_path("SVM", "classification")
        == "model__estimator__class_weight"
    )
    assert model_specs.class_weight_path("KNN", "classification") is None
    assert model_specs.class_weight_path("Regresión Lineal", "regression") is None


def test_class_weight_path_con_modelo_desconocido():
    assert model_specs.class_weight_path("Inventado") is None


# ---------------------------------------------------------------------------
# BalancedSampler
# ---------------------------------------------------------------------------


@pytest.fixture
def desbalanceado():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "num": rng.normal(size=120),
            "cat": ["x", "y"] * 60,
        }
    )
    y = pd.Series(["mayor"] * 100 + ["menor"] * 20)
    return X, y


def _conteos(y):
    return {clase: int((y == clase).sum()) for clase in np.unique(y)}


def test_fit_no_remuestrea(desbalanceado):
    X, y = desbalanceado
    sampler = balancing.BalancedSampler(method="random_under").fit(X, y)

    assert sampler.n_before == 120
    assert sampler.n_after is None
    assert sampler.classes_before == ("mayor", "menor")


def test_fit_transform_none_es_identidad(desbalanceado):
    X, y = desbalanceado
    sampler = balancing.BalancedSampler(method="none")
    X2, y2 = sampler.fit_resample(X, y)

    assert X2 is X
    assert y2 is y
    assert sampler.n_after == 120


def test_submuestreo_baja_a_la_minoritaria(desbalanceado):
    X, y = desbalanceado
    sampler = balancing.BalancedSampler(method="random_under")
    X2, y2 = sampler.fit_resample(X, y)

    assert _conteos(y2) == {"mayor": 20, "menor": 20}
    assert sampler.n_after == 40
    assert len(X2) == 40


def test_sobremuestreo_sube_a_la_mayoritaria(desbalanceado):
    X, y = desbalanceado
    sampler = balancing.BalancedSampler(method="random_over")
    X2, y2 = sampler.fit_resample(X, y)

    assert _conteos(y2) == {"mayor": 100, "menor": 100}
    assert sampler.n_after == 200
    assert len(X2) == 200


def test_remuestreo_conserva_el_tipo_de_contenedor(desbalanceado):
    X, y = desbalanceado
    sampler = balancing.BalancedSampler(method="random_under")
    X2, y2 = sampler.fit_resample(X, y)

    assert isinstance(X2, pd.DataFrame)
    assert list(X2.columns) == list(X.columns)


def test_remuestreo_funciona_con_matriz_dispersa(desbalanceado):
    X, y = desbalanceado
    denso = sparse.csr_matrix(np.arange(240, dtype=float).reshape(120, 2))
    sampler = balancing.BalancedSampler(method="random_under")
    X2, y2 = sampler.fit_resample(denso, y)

    assert sparse.issparse(X2)
    assert X2.shape[0] == 40
    assert _conteos(y2) == {"mayor": 20, "menor": 20}


def test_balanceo_en_regresion_da_error():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    y = pd.Series([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="clasificación"):
        balancing.BalancedSampler(method="random_under", task_type="regression").fit(
            X, y
        )


def test_smote_necesita_dos_instancias_en_la_minoritaria():
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]})
    y = pd.Series(["a", "a", "a", "b"])

    with pytest.raises(ValueError, match="minoritaria"):
        balancing.BalancedSampler(method="smote").fit_resample(X, y)


def test_smote_equilibra_las_clases():
    pytest.importorskip("imblearn")
    rng = np.random.default_rng(1)
    X = pd.DataFrame({"a": rng.normal(size=120), "b": rng.normal(size=120)})
    y = pd.Series(["mayor"] * 100 + ["menor"] * 20)

    X2, y2 = balancing.BalancedSampler(method="smote").fit_resample(X, y)

    assert _conteos(y2) == {"mayor": 100, "menor": 100}
    assert len(X2) == 200


def test_smoten_equilibra_las_clases():
    pytest.importorskip("imblearn")
    rng = np.random.default_rng(2)
    X = pd.DataFrame({"a": rng.normal(size=120), "cat": ["x", "y"] * 60})
    y = pd.Series(["mayor"] * 100 + ["menor"] * 20)

    sampler = balancing.BalancedSampler(method="smoten", categorical_features=[1])
    X2, y2 = sampler.fit_resample(X, y)

    assert _conteos(y2) == {"mayor": 100, "menor": 100}


def test_smoten_con_matriz_dispersa_la_densifica():
    pytest.importorskip("imblearn")
    rng = np.random.default_rng(3)
    X = sparse.csr_matrix(rng.normal(size=(120, 3)))
    y = pd.Series(["mayor"] * 100 + ["menor"] * 20)

    sampler = balancing.BalancedSampler(method="smoten", categorical_features=[0])
    X2, y2 = sampler.fit_resample(X, y)

    assert _conteos(y2) == {"mayor": 100, "menor": 100}


def test_stats_recoge_lo_necesario_para_el_bundle(desbalanceado):
    X, y = desbalanceado
    sampler = balancing.BalancedSampler(
        method="random_under", k_neighbors=3, random_state=7
    ).fit(X, y)
    sampler.fit_resample(X, y)

    stats = sampler.stats
    assert stats["method"] == "random_under"
    assert stats["k_neighbors"] == 3
    assert stats["n_before"] == 120
    assert stats["n_after"] == 40
    assert stats["classes_before"] == ["mayor", "menor"]
    assert stats["classes_after"] == ["mayor", "menor"]


def test_clone_y_params_funcionan(desbalanceado):
    X, y = desbalanceado
    sampler = balancing.BalancedSampler(
        method="random_over", k_neighbors=4, random_state=9
    )

    assert sampler.get_params()["method"] == "random_over"
    assert sampler.get_params()["k_neighbors"] == 4
    copia = clone(sampler)
    assert copia.get_params()["random_state"] == 9
    sampler.set_params(k_neighbors=2)
    assert sampler.get_params()["k_neighbors"] == 2


def test_metodo_desconocido_da_error_en_espanol(desbalanceado):
    X, y = desbalanceado

    with pytest.raises(ValueError, match="desconocida"):
        balancing.BalancedSampler(method="inventado").fit_resample(X, y)


def test_sampler_dentro_de_un_pipeline(desbalanceado):
    pytest.importorskip("imblearn")
    from imblearn.pipeline import Pipeline

    X, y = desbalanceado
    X = X[["num"]]
    pipe = Pipeline(
        [
            ("balancer", balancing.BalancedSampler(method="random_under")),
            ("model", LogisticRegression(max_iter=200)),
        ]
    )

    pipe.fit(X, y)

    assert pipe.predict(X).shape[0] == len(X)
    assert pipe.named_steps["balancer"].n_after == 40


# ---------------------------------------------------------------------------
# Integración con build_pipeline
# ---------------------------------------------------------------------------


@pytest.fixture
def dataset_clasificacion():
    rng = np.random.default_rng(4)
    n = 120
    return pd.DataFrame(
        {
            "edad": rng.integers(20, 80, size=n),
            "ciudad": rng.choice(["a", "b", "c"], size=n),
            "objetivo": ["no"] * 90 + ["sí"] * 30,
        }
    )


def test_submuestreo_no_toca_el_conjunto_de_prueba(dataset_clasificacion):
    _, _, (X_test, y_test, _) = model_trainer.train_model(
        dataset_clasificacion,
        target="objetivo",
        model_name="Regresión Logística",
        task_type="classification",
        balancing={"method": "random_under"},
    )

    assert len(y_test) == 24  # 20 % de 120, igual que sin balanceo
    assert set(y_test) == {"no", "sí"}


def test_class_weight_no_añade_paso_balancer(dataset_clasificacion):
    pipe, _, _ = model_trainer.train_model(
        dataset_clasificacion,
        target="objetivo",
        model_name="Regresión Logística",
        task_type="classification",
        balancing={"method": "class_weight"},
    )

    assert "balancer" not in pipe.named_steps
    modelo = pipe.named_steps["model"]
    assert modelo.get_params()["class_weight"] == "balanced"


def test_class_weight_con_modelo_que_no_lo_admite(dataset_clasificacion):
    with pytest.raises(ValueError, match="no admite pesos de clase"):
        model_trainer.train_model(
            dataset_clasificacion,
            target="objetivo",
            model_name="KNN",
            task_type="classification",
            balancing={"method": "class_weight"},
        )


def test_balanceo_en_regresion_da_error(dataset_clasificacion):
    with pytest.raises(ValueError, match="clasificación"):
        model_trainer.train_model(
            dataset_clasificacion,
            target="objetivo",
            model_name="Regresión Lineal",
            task_type="regression",
            balancing={"method": "random_under"},
        )


def test_export_df_mantiene_sus_filas_con_balanceo(dataset_clasificacion):
    pipe, _, _ = model_trainer.train_model(
        dataset_clasificacion,
        target="objetivo",
        model_name="Regresión Logística",
        task_type="classification",
        balancing={"method": "random_under"},
    )

    export = model_trainer.transform_with_pipeline(dataset_clasificacion, pipe)

    assert len(export) == len(dataset_clasificacion)


def test_smoten_detecta_las_columnas_categoricas(dataset_clasificacion):
    pytest.importorskip("imblearn")
    pipe = model_trainer.build_pipeline(
        dataset_clasificacion,
        "Regresión Logística",
        "classification",
        "objetivo",
        balancing={"method": "smoten"},
    )

    sampler = pipe.named_steps["balancer"]
    assert sampler.categorical_features
    assert all(indice >= 1 for indice in sampler.categorical_features)


def test_smoten_sin_categóricas_avisa(dataset_clasificacion):
    pytest.importorskip("imblearn")
    solo_numericas = dataset_clasificacion[["edad", "objetivo"]]

    with pytest.raises(ValueError, match="SMOTEN"):
        model_trainer.build_pipeline(
            solo_numericas,
            "Regresión Logística",
            "classification",
            "objetivo",
            balancing={"method": "smoten"},
        )


def test_tune_model_con_class_weight_lo_incluye_en_el_grid(dataset_clasificacion):
    _, metrics, _, cv = model_trainer.tune_model(
        dataset_clasificacion,
        target="objetivo",
        model_name="Regresión Logística",
        task_type="classification",
        cv=2,
        balancing={"method": "class_weight"},
    )

    assert any("class_weight" in str(k) for k in cv["params"][0])


def test_tune_model_con_remuestreo_funciona(dataset_clasificacion):
    pytest.importorskip("imblearn")
    pipe, metrics, _, _ = model_trainer.tune_model(
        dataset_clasificacion,
        target="objetivo",
        model_name="Regresión Logística",
        task_type="classification",
        cv=2,
        balancing={"method": "random_under"},
    )

    assert "balancer" in pipe.named_steps
    assert metrics["auc_disponible"] is True
