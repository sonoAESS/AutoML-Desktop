"""Pruebas de `core/comparison.py` (SDD-005)."""

import numpy as np
import pandas as pd
import pytest

from core import comparison

CLASIFICACION = ["Regresión Logística", "Árbol de Decisión", "Random Forest"]
REGRESION = ["Regresión Lineal", "Ridge", "Random Forest"]


@pytest.fixture
def dataset_clasificacion():
    """Un atributo útil y cuatro de ruido; dos clases equilibradas."""
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    ruido = rng.normal(size=(n, 4))
    df = pd.DataFrame(
        np.column_stack([x, ruido]),
        columns=["x0", "r1", "r2", "r3", "r4"],
    )
    df["objetivo"] = (df["x0"] > 0).astype(int)
    return df


@pytest.fixture
def dataset_regresion():
    rng = np.random.default_rng(1)
    n = 200
    x = rng.normal(size=n)
    ruido = rng.normal(size=(n, 4))
    df = pd.DataFrame(
        np.column_stack([x, ruido]),
        columns=["x0", "r1", "r2", "r3", "r4"],
    )
    # El ruido del objetivo se genera aparte: si fuera una de las columnas, la
    # regresión lineal lo ajustaría exactamente y su RMSE sería cero.
    df["objetivo"] = 3.0 * x + rng.normal(scale=0.5, size=n)
    return df


# ----------------------------------------------------------------------
# Validación
# ----------------------------------------------------------------------
def test_menos_de_tres_modelos_avisa(dataset_clasificacion):
    with pytest.raises(ValueError, match="al menos 3 modelos"):
        comparison.compare_models(
            dataset_clasificacion,
            "objetivo",
            ["Regresión Logística", "Random Forest"],
            "classification",
            "accuracy",
        )


def test_metrica_no_soportada_avisa(dataset_clasificacion):
    with pytest.raises(ValueError, match="Métrica no soportada"):
        comparison.compare_models(
            dataset_clasificacion,
            "objetivo",
            CLASIFICACION,
            "classification",
            "precision",
            n_splits=4,
            n_repeats=1,
        )


# ----------------------------------------------------------------------
# C-001 · estructura del resultado
# ----------------------------------------------------------------------
@pytest.mark.slow
def test_tres_modeles_diez_bloques(dataset_clasificacion):
    resultado = comparison.compare_models(
        dataset_clasificacion,
        "objetivo",
        CLASIFICACION,
        "classification",
        "accuracy",
        n_splits=5,
        n_repeats=2,
    )

    assert resultado.n_observaciones == 10
    assert len(resultado.ranking) == 3
    assert resultado.n_splits == 5
    assert resultado.n_repeats == 2
    # Una fila por modelo y bloque, sin repeticiones perdidas.
    assert len(resultado.scores) == 30
    assert resultado.scores.duplicated(["modelo", "repeticion", "fold"]).sum() == 0


@pytest.mark.slow
def test_el_ranking_va_de_mejor_a_peor(dataset_clasificacion):
    resultado = comparison.compare_models(
        dataset_clasificacion,
        "objetivo",
        CLASIFICACION,
        "classification",
        "accuracy",
        n_splits=5,
        n_repeats=2,
    )
    rangos = resultado.ranking["rango_medio"].to_numpy()

    assert (rangos[:-1] <= rangos[1:]).all()
    assert resultado.mejor_modelo == resultado.ranking.iloc[0]["modelo"]


# ----------------------------------------------------------------------
# C-002 · empate
# ----------------------------------------------------------------------
def test_un_tie_produce_el_mismo_rango_medio():
    """Dos modelos idénticos empatan en rango medio (C-002)."""
    matriz = np.array(
        [
            [0.9, 0.9, 0.7],
            [0.8, 0.8, 0.6],
            [0.95, 0.95, 0.5],
        ]
    )
    rangos = comparison.stats.rankdata(-matriz, axis=0)
    ranking = comparison._ranking(matriz, rangos, ["a", "b", "c"], cd=1.5)

    rangos_medios = dict(zip(ranking["modelo"], ranking["rango_medio"]))
    assert rangos_medios["a"] == rangos_medios["b"]


def test_modelos_iguales_dan_p_valor_alto(dataset_clasificacion, monkeypatch):
    """Con los tres modelos idénticos, Friedman no puede rechazar."""
    monkeypatch.setattr(
        comparison,
        "_score_block",
        lambda metric, pipe, X_test, y_test, task_type: 0.8,
    )
    resultado = comparison.compare_models(
        dataset_clasificacion,
        "objetivo",
        CLASIFICACION,
        "classification",
        "accuracy",
        n_splits=4,
        n_repeats=2,
    )
    assert resultado.friedman_p >= resultado.alpha
    assert resultado.ranking["rango_medio"].nunique() == 1
    assert (resultado.posthoc["p_valor"] == 1.0).all()


# ----------------------------------------------------------------------
# C-003 · diferencia real
# ----------------------------------------------------------------------
@pytest.mark.slow
def test_un_modelo_muy_superior_es_significativo():
    """Un producto interactivo: solo el bosque lo separa del resto."""
    rng = np.random.default_rng(2)
    n = 300
    x0 = rng.normal(size=n)
    x1 = rng.normal(size=n)
    df = pd.DataFrame(
        {"x0": x0, "x1": x1, "r1": rng.normal(size=n), "r2": rng.normal(size=n)}
    )
    df["objetivo"] = ((x0 * x1) > 0.5).astype(int)

    resultado = comparison.compare_models(
        df,
        "objetivo",
        ["Regresión Logística", "Árbol de Decisión", "Random Forest"],
        "classification",
        "accuracy",
        n_splits=5,
        n_repeats=3,
    )

    assert resultado.friedman_p < resultado.alpha
    assert resultado.mejor_modelo == "Random Forest"
    frente_al_mejor = resultado.posthoc[
        resultado.posthoc["modelo_b"] == resultado.mejor_modelo
    ]
    assert frente_al_mejor["significativo"].any()


# ----------------------------------------------------------------------
# C-004 · cinco modelos
# ----------------------------------------------------------------------
@pytest.mark.slow
def test_cinco_modelos_completan_la_tabla(dataset_clasificacion):
    resultado = comparison.compare_models(
        dataset_clasificacion,
        "objetivo",
        ["Regresión Logística", "KNN", "Árbol de Decisión", "Random Forest", "SVM"],
        "classification",
        "accuracy",
        n_splits=4,
        n_repeats=1,
    )
    assert len(resultado.ranking) == 5


# ----------------------------------------------------------------------
# C-006 · métricas de error
# ----------------------------------------------------------------------
@pytest.mark.slow
def test_rmse_ordena_de_menor_a_mayor(dataset_regresion):
    resultado = comparison.compare_models(
        dataset_regresion,
        "objetivo",
        REGRESION,
        "regression",
        "rmse",
        n_splits=4,
        n_repeats=2,
    )
    rmse = resultado.ranking["puntuacion_media"].to_numpy()
    rangos = resultado.ranking["rango_medio"].to_numpy()

    assert (rmse > 0).all()  # el RMSE real, no el orientado
    assert (rangos[:-1] <= rangos[1:]).all()
    assert rangos[0] == rangos.min()
    # Donde el rango medio es estricto, la media de RMSE debe crecer con él:
    # un RMSE más alto no puede tener mejor rango.
    distintos = pd.Series(rangos).diff().fillna(0) > 0
    if distintos.any():
        assert (np.diff(rmse)[distintos.to_numpy()[1:]] >= 0).all()


def test_la_orientacion_invierte_el_rmse():
    assert comparison.orientation("rmse") == -1
    assert comparison.orientation("mae") == -1
    assert comparison.orientation("accuracy") == 1
    assert comparison.orientation("r2") == 1


# ----------------------------------------------------------------------
# C-007 · bloques pareados
# ----------------------------------------------------------------------
@pytest.mark.slow
def test_los_bloques_son_los_mismos_para_todos_los_modelos(dataset_clasificacion):
    resultado = comparison.compare_models(
        dataset_clasificacion,
        "objetivo",
        CLASIFICACION,
        "classification",
        "accuracy",
        n_splits=4,
        n_repeats=2,
        random_state=7,
    )
    # Si los bloques no estuvieran pareados, ningún modelo tendría puntuación en
    # todos ellos: la matriz del ranking se quedaría hueca.
    assert not resultado.scores["puntuacion"].isna().any()
    assert resultado.n_observaciones == 8

    repetido = comparison.compare_models(
        dataset_clasificacion,
        "objetivo",
        CLASIFICACION,
        "classification",
        "accuracy",
        n_splits=4,
        n_repeats=2,
        random_state=7,
    )
    pd.testing.assert_frame_equal(resultado.scores, repetido.scores)


# ----------------------------------------------------------------------
# C-008 · sin fuga entre folds
# ----------------------------------------------------------------------
def test_n_splits_se_recorta_a_la_clase_minoritaria():
    y = pd.Series([0] * 40 + [1] * 3)
    assert comparison._effective_n_splits("classification", y, 5) == 3


def test_clase_minoritaria_minima_avisa():
    y = pd.Series([0] * 40 + [1] * 1)
    with pytest.raises(ValueError, match="clase minoritaria"):
        comparison._effective_n_splits("classification", y, 5)


# ----------------------------------------------------------------------
# R-013 · exclusión
# ----------------------------------------------------------------------
def test_un_modelo_que_falla_se_excluye(dataset_clasificacion, monkeypatch):
    """Si un modelo falla en un fold, se excluye y el resto sigue."""
    from sklearn.tree import DecisionTreeClassifier

    original = comparison._score_block

    def _selectivo(metric, pipe, X_test, y_test, task_type):
        if isinstance(pipe.named_steps["model"], DecisionTreeClassifier):
            return float("nan")
        return original(metric, pipe, X_test, y_test, task_type)

    monkeypatch.setattr(comparison, "_score_block", _selectivo)
    resultado = comparison.compare_models(
        dataset_clasificacion,
        "objetivo",
        CLASIFICACION + ["KNN"],
        "classification",
        "accuracy",
        n_splits=4,
        n_repeats=1,
    )

    assert "Árbol de Decisión" in resultado.excluidos
    assert len(resultado.ranking) == 3
    assert "Árbol de Decisión" not in set(resultado.ranking["modelo"])
    assert "Excluido: Árbol de Decisión" in resultado.describe()


def test_si_sobreviven_dos_modelos_falla(dataset_clasificacion, monkeypatch):
    monkeypatch.setattr(comparison, "_score_block", lambda *a, **k: float("nan"))
    with pytest.raises(ValueError, match="al menos 3"):
        comparison.compare_models(
            dataset_clasificacion,
            "objetivo",
            CLASIFICACION,
            "classification",
            "accuracy",
            n_splits=4,
            n_repeats=1,
        )


# ----------------------------------------------------------------------
# R-014 · progreso
# ----------------------------------------------------------------------
@pytest.mark.slow
def test_el_progreso_llega_de_uno_al_total(dataset_clasificacion):
    avances = []
    resultado = comparison.compare_models(
        dataset_clasificacion,
        "objetivo",
        CLASIFICACION,
        "classification",
        "accuracy",
        n_splits=4,
        n_repeats=1,
        progress=lambda hecho, total: avances.append((hecho, total)),
    )

    assert avances[0][0] == 1
    assert avances[-1][0] == avances[-1][1] == 3 * resultado.n_observaciones
    assert all(total == 12 for _hecho, total in avances)


# ----------------------------------------------------------------------
# T-004 · Holm y diferencia crítica
# ----------------------------------------------------------------------
def test_holm_es_monotono_y_acotado():
    ajustados = comparison._holms_pvalues([0.01, 0.04, 0.03, 0.2])
    ordenados = np.sort(ajustados)

    assert (ordenados[:-1] <= ordenados[1:]).all()
    assert (ajustados <= 1.0).all()
    assert (ajustados >= np.array([0.01, 0.04, 0.03, 0.2])).all()


def test_holm_con_valores_conocidos():
    # m = 3: el menor se multiplica por 3 y el segundo por 2.
    ajustados = comparison._holms_pvalues([0.01, 0.02, 0.03])
    assert ajustados == pytest.approx([0.03, 0.04, 0.04])


def test_holm_acota_a_uno():
    assert comparison._holms_pvalues([0.6, 0.7, 0.9])[-1] == pytest.approx(1.0)


def test_critical_difference_crece_con_el_numero_de_bloques():
    cd_pocos = comparison.critical_difference(3, 5)
    cd_muchos = comparison.critical_difference(3, 30)

    assert cd_pocos > cd_muchos
    assert comparison.critical_difference(2, 10) > comparison.critical_difference(5, 10)


def test_equivalente_usa_la_diferencia_critica(dataset_clasificacion, monkeypatch):
    """Un rango dentro de la CD del mejor se marca como equivalente."""
    monkeypatch.setattr(
        comparison, "critical_difference", lambda k, b, alpha=0.05: 99.0
    )
    resultado = comparison.compare_models(
        dataset_clasificacion,
        "objetivo",
        CLASIFICACION,
        "classification",
        "accuracy",
        n_splits=4,
        n_repeats=1,
    )
    assert resultado.ranking["equivalente"].all()
    assert bool(resultado.ranking["equivalente"].iloc[0])


def test_post_hoc_con_pocos_bloques_usa_t_pareado(dataset_clasificacion):
    resultado = comparison.compare_models(
        dataset_clasificacion,
        "objetivo",
        CLASIFICACION,
        "classification",
        "accuracy",
        n_splits=3,
        n_repeats=1,
    )
    assert set(resultado.posthoc["test"]) == {"t pareado"}
    assert (resultado.posthoc["n_bloques"] == 3).all()


def test_post_hoc_con_muchos_bloques_usa_wilcoxon(dataset_clasificacion):
    resultado = comparison.compare_models(
        dataset_clasificacion,
        "objetivo",
        CLASIFICACION,
        "classification",
        "accuracy",
        n_splits=4,
        n_repeats=2,
    )
    assert set(resultado.posthoc["test"]) == {"Wilcoxon"}


# ----------------------------------------------------------------------
# Resumen
# ----------------------------------------------------------------------
def test_describe_explica_el_p_valor(dataset_clasificacion):
    resultado = comparison.compare_models(
        dataset_clasificacion,
        "objetivo",
        CLASIFICACION,
        "classification",
        "accuracy",
        n_splits=4,
        n_repeats=1,
    )
    texto = resultado.describe()

    assert "Friedman χ²=" in texto
    assert "Mejor modelo:" in texto
    if resultado.friedman_p >= resultado.alpha:
        assert "no hay diferencias" in texto
    else:
        assert "son estadísticamente significativas" in texto


# ----------------------------------------------------------------------
# T-009 · estado
# ----------------------------------------------------------------------
def test_app_state_guarda_la_comparativa():
    from core.state import AppState

    state = AppState()
    assert state.comparison is None
    assert state.comparison_key is None

    state.comparison = "resultado"
    state.comparison_key = ("objetivo", "classification", "accuracy", 3, True)
    state.reset_data()
    assert state.comparison is None
    assert state.comparison_key is None


def test_reset_model_olvida_la_comparativa():
    from core.state import AppState

    state = AppState()
    state.comparison = "resultado"
    state.comparison_key = ("objetivo", "classification", "accuracy", 3, True)
    state.reset_model()

    assert state.comparison is None
    assert state.comparison_key is None
