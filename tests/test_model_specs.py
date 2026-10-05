# tests/test_model_specs.py
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression

from core import model_specs, profiling


def test_list_models_por_tarea_y_familia():
    assert "Regresión Logística" in model_specs.list_models("classification")
    assert "Regresión Lineal" in model_specs.list_models("regression")
    assert "SVM" in model_specs.list_models("classification", "svm")
    assert "SGD" in model_specs.list_models("classification", "lineal")


def test_el_catalogo_tiene_los_modelos_nuevos():
    """C-001, C-002: recuentos y familias tras ampliar el catálogo."""
    assert len(model_specs.list_models("classification")) == 17
    assert len(model_specs.list_models("regression")) == 18
    assert set(model_specs.list_families()) == {
        "lineal",
        "arbol",
        "ensemble",
        "svm",
        "vecinos",
        "naive_bayes",
        "lda",
    }
    for clave, etiqueta in model_specs.FAMILIES.items():
        assert etiqueta, f"la familia {clave} no tiene etiqueta"


def test_class_weight_solo_donde_sklearn_lo_acepta():
    """C-003: el campo se informa donde el estimador acepta el parámetro."""
    con_peso = {
        nombre
        for nombre in model_specs.list_models("classification")
        if model_specs.class_weight_path(nombre) is not None
    }
    assert con_peso >= {
        "HistGradientBoosting",
        "ExtraTrees",
        "SGD",
        "LinearSVC",
    }
    # Bagging, AdaBoost y GradientBoosting no aceptan `class_weight`: si se
    # informara, `build_pipeline` pondría un parámetro que el estimador ignora.
    for nombre in ("Bagging", "AdaBoost", "GradientBoosting", "GaussianNB"):
        assert model_specs.class_weight_path(nombre) is None


def test_las_familias_nuevas_tienen_minimo_de_filas():
    """C-004."""
    assert model_specs.MIN_ROWS == {
        "svm": 50,
        "vecinos": 30,
        "naive_bayes": 10,
        "lda": 20,
    }


def test_ningun_grid_supera_las_500_combinaciones():
    """C-005: un grid enorme hace la búsqueda inmanejable."""
    for nombre, spec in model_specs.iter_specs():
        grid = spec.grid()
        combinaciones = 1
        for valores in grid.values():
            combinaciones *= len(valores)
        assert combinaciones <= 500, f"{nombre}: {combinaciones} combinaciones"
        assert all(clave.startswith("model__") for clave in grid), nombre


def test_los_metadatos_muertos_se_eliminaron():
    """C-006: `supports_probability` no lo leía nadie; el AUC se decide en
    runtime con `hasattr(pipe, "predict_proba")`."""
    assert not hasattr(model_specs.ModelSpec, "supports_probability")
    assert not hasattr(model_specs, "RANKING_METRIC")
    assert not hasattr(model_specs, "metrics_for_ranking")


def test_build_estimator_devuelve_el_tipo_correcto():
    """C-007."""
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import LinearSVC

    assert isinstance(model_specs.build_estimator("GaussianNB"), GaussianNB)
    assert isinstance(
        model_specs.build_estimator("LinearDiscriminant"), LinearDiscriminantAnalysis
    )
    assert isinstance(
        model_specs.build_estimator("LinearSVC", "classification"), LinearSVC
    )
    assert isinstance(
        model_specs.build_estimator("HistGradientBoosting"),
        HistGradientBoostingClassifier,
    )


def test_los_modelos_con_semilla_la_fijan_en_42():
    """C-007: sin `random_state` los resultados no son reproducibles.

    Solo se comprueban los modelos añadidos en esta spec: `Regresión Logística`
    y `Ridge` no fijan semilla en el catálogo preexistente, y cambiarlas está
    fuera de alcance.
    """
    nuevos = {
        "SGD",
        "ExtraTrees",
        "Bagging",
        "AdaBoost",
        "GradientBoosting",
        "HistGradientBoosting",
        "LinearSVC",
        "Lasso",
        "ElasticNet",
        "TheilSen",
    }
    for nombre in nuevos:
        estimador = model_specs.get_spec(nombre).build()
        assert getattr(estimador, "random_state", 42) == 42, nombre


def test_la_familia_lda_se_excluye_con_pocas_filas():
    """C-008."""
    perfiles = [
        profiling.ColumnProfile(
            name="a",
            dtype="float64",
            semantic_type="numeric",
            n_unique=15,
            n_rows=15,
            n_missing=0,
        )
    ]
    resultados = model_specs.evaluate_compatibility(
        "classification", perfiles, target="objetivo", n_rows=15
    )
    por_nombre = {item.name: item for item in resultados}
    assert por_nombre["LinearDiscriminant"].compatible is False
    assert "20 filas" in por_nombre["LinearDiscriminant"].reason
    # GaussianNB sí entra: su mínimo son 10 filas.
    assert por_nombre["GaussianNB"].compatible is True


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


@pytest.mark.slow
@pytest.mark.parametrize(
    "model_name",
    [
        spec.name
        for task in ("classification", "regression")
        for spec in model_specs.SPECS[task].values()
    ],
    ids=lambda n: n,
)
def test_todas_las_combinaciones_del_grid_son_validas(model_name):
    """Un grid con combinaciones que fallan no falla: puntúa NaN en silencio.

    Es lo que pasó con `LinearDiscriminant`: el solver `svd` no acepta
    `shrinkage`, así que la mitad del grid lanzaba `NotImplementedError` y la
    búsqueda acababa eligiendo sobre puntuaciones incompletas, sin un solo error
    visible. Se recorre el grid entero del catálogo con los avisos de convergencia
    convertidos en errores.

    Son 1620 ajustes sobre 30 filas: segundos y detecta combinaciones rotas.
    """
    import itertools
    import warnings

    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.preprocessing import StandardScaler

    X, y = load_iris(return_X_y=True)
    # Subconjunto balanceado: con `X[:30]` sale una sola clase y casi todos los
    # estimadores fallarían por eso, no por su grid.
    keep = np.concatenate([np.flatnonzero(y == clase)[:10] for clase in (0, 1, 2)])
    # Escalado como hace `model_trainer` antes del estimador: sin él, Lasso y
    # ElasticNet no convergen y el aviso sería del fixture, no del modelo.
    X = StandardScaler().fit_transform(X[keep])
    y = y[keep]

    spec = model_specs.get_spec(model_name)
    # Solo se quita el prefijo del pipeline: rutas anidadas como
    # `estimator__C` (SVM dentro de CalibratedClassifierCV) se respetan.
    prefijo = "model__"
    grid = {
        clave[len(prefijo) :]: vals
        for clave, vals in spec.grid().items()
        if clave.startswith(prefijo)
    }
    claves = list(grid)

    for valores in itertools.product(*(grid[k] for k in claves)):
        params = dict(zip(claves, valores))
        estimador = spec.build()
        estimador.set_params(**params)
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=ConvergenceWarning)
            try:
                estimador.fit(X, y)
            except Exception as exc:
                raise AssertionError(
                    f"{model_name} falla con {params}: {type(exc).__name__}: {exc}"
                ) from exc
