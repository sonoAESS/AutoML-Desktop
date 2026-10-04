# tests/test_model_specs.py
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from core import model_specs, profiling


def test_list_models_por_tarea_y_familia():
    assert "Regresión Logística" in model_specs.list_models("classification")
    assert "Regresión Lineal" in model_specs.list_models("regression")
    assert model_specs.list_models("classification", "svm") == ["SVM"]
    assert model_specs.list_models("classification", "lineal") == [
        "Regresión Logística"
    ]


def test_los_modelos_comparten_nombre_entre_tareas():
    comun = set(model_specs.list_models("classification")) & set(
        model_specs.list_models("regression")
    )
    assert {"Árbol de Decisión", "Random Forest", "KNN"} <= comun


def test_build_estimator_devuelve_instancias_nuevas():
    spec = model_specs.get_spec("Regresión Logística", "classification")
    assert isinstance(spec.build(), LogisticRegression)
    assert spec.build() is not spec.build()


def test_svm_usa_calibracion_para_tener_probabilidades():
    model = model_specs.build_estimator("SVM", "classification")
    assert isinstance(model, CalibratedClassifierCV)
    assert hasattr(model, "predict_proba")


def test_param_spec_genera_grid():
    spec = model_specs.get_spec("Regresión Logística", "classification")
    grid = spec.grid()
    assert grid["model__C"] == [0.01, 0.1, 1.0, 10.0, 100.0]

    filtrado = spec.grid({"C": 1.0})
    assert filtrado["model__C"] == [1.0]

    completo = spec.grid({"C": model_specs.ALL_VALUES})
    assert completo["model__C"] == list(spec.param("C").values)


def test_available_metrics_depende_del_modelo():
    assert "roc_auc" in model_specs.available_metrics("classification", "SVM")
    assert "roc_auc" not in model_specs.available_metrics("regression", "SVR")
    assert model_specs.available_metrics("regression", "Ridge") == ["rmse", "mae", "r2"]


def test_compatibilidad_bloquea_columnas_texto():
    df = pd.DataFrame(
        {
            "nota": [f"comentario largo {i} con muchas palabras" for i in range(80)],
            "x": range(80),
        }
    )
    profiles = profiling.profile_dataframe(df)
    results = model_specs.evaluate_compatibility("classification", profiles, target="x")
    assert results and all(not r.compatible for r in results)
    assert "nota" in results[0].reason


def test_compatibilidad_por_numero_de_filas():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [0, 1, 0]})
    profiles = profiling.profile_dataframe(df)
    results = {
        r.name: r
        for r in model_specs.evaluate_compatibility(
            "classification", profiles, target="y"
        )
    }
    assert results["Regresión Logística"].compatible is True
    assert results["SVM"].compatible is False
    assert "50" in results["SVM"].reason
    assert "30" in results["KNN"].reason


def test_compatibilidad_ok_con_dataset_normal():
    df = pd.DataFrame({"x": [float(i % 20) for i in range(100)], "y": [0, 1] * 50})
    profiles = profiling.profile_dataframe(df)
    results = model_specs.evaluate_compatibility("classification", profiles, target="y")
    assert all(r.compatible for r in results)
    assert "compatibles" in model_specs.summarize_compatibility(results)


def test_summarize_compatibility_explica_motivos():
    results = [
        model_specs.ModelCompatibility("SVM", "svm", False, "necesita 50 filas"),
    ]
    texto = model_specs.summarize_compatibility(results)
    assert "SVM" in texto and "50 filas" in texto


def test_get_spec_por_defecto_busca_en_ambas_tareas():
    assert model_specs.get_spec("KNN").task_type == "classification"
    with pytest.raises(KeyError):
        model_specs.get_spec("No existe")


def test_none_es_un_valor_elegido_y_no_el_dominio_completo():
    """Elegir «ninguno» en max_depth debe buscar solo ese valor."""
    spec = model_specs.get_spec("Árbol de Decisión", "classification")
    grid = spec.grid({"max_depth": None})
    assert grid["model__max_depth"] == [None]

    ausente = spec.grid({"min_samples_split": 5})
    assert ausente["model__max_depth"] == list(spec.param("max_depth").values)
    assert ausente["model__min_samples_split"] == [5]
