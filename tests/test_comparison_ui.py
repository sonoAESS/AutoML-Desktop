"""Pruebas headless de la comparativa de modelos en la interfaz (SDD-005)."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from core.state import AppState  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def mensajes(app, monkeypatch):
    registrados = []

    def registrar(categoria):
        def _falso(parent, titulo, texto, *args, **kwargs):
            registrados.append((categoria, titulo, texto))
            return QMessageBox.Yes

        return _falso

    for categoria in ("critical", "warning", "information", "question"):
        monkeypatch.setattr(QMessageBox, categoria, registrar(categoria))
    return registrados


@pytest.fixture
def dataset():
    rng = np.random.default_rng(0)
    n = 120
    x = rng.normal(size=n)
    df = pd.DataFrame(
        np.column_stack([x, rng.normal(size=(n, 2))]), columns=["x0", "r1", "r2"]
    )
    df["objetivo"] = (df["x0"] > 0).astype(int)
    return df


def _comparison_falsa():
    """Un `ModelComparison` construido a mano, sin entrenar nada."""
    from core.comparison import ModelComparison

    return ModelComparison(
        metric="accuracy",
        task_type="classification",
        scores=pd.DataFrame(
            {
                "modelo": ["a", "b", "c"] * 4,
                "repeticion": [1] * 12,
                "fold": [1, 2, 3, 4] * 3,
                "puntuacion": [0.9, 0.8, 0.7] * 4,
            }
        ),
        ranking=pd.DataFrame(
            {
                "modelo": ["Random Forest", "KNN", "SVM"],
                "puntuacion_media": [0.9, 0.8, 0.7],
                "desviacion": [0.03, 0.04, 0.05],
                "rango_medio": [1.0, 2.0, 3.5],
                "equivalente": [True, True, False],
            }
        ),
        friedman_statistic=12.4,
        friedman_p=0.002,
        n_observaciones=12,
        posthoc=pd.DataFrame(
            {
                "modelo_a": ["KNN", "KNN"],
                "modelo_b": ["Random Forest", "SVM"],
                "diferencia_media": [0.1, 0.05],
                "p_valor": [0.001, 0.02],
                "test": ["Wilcoxon", "Wilcoxon"],
                "n_bloques": [12, 12],
                "p_ajustada": [0.003, 0.04],
                "significativo": [True, True],
            }
        ),
        critical_difference=1.65,
        alpha=0.05,
        mejor_modelo="Random Forest",
        excluidos={"KNN raro": "no se pudo puntuar"},
        n_splits=4,
        n_repeats=3,
    )


def _pestana(dataset, con_comparativa=False):
    from ui.results_tab import ResultsTab

    state = AppState()
    state.clean_df = dataset
    state.export_df = dataset
    state.target_column = "objetivo"
    state.task_type = "classification"
    state.model_name = "Random Forest"
    state.family = "ensemble"
    state.metrics = {"accuracy": 0.9, "f1_macro": 0.88, "roc_auc": 0.95, "n_test": 24}
    if con_comparativa:
        state.comparison = _comparison_falsa()
        state.comparison_key = (
            "objetivo",
            "classification",
            "accuracy",
            3,
            True,
        )
    tab = ResultsTab(state)
    tab.refresh()
    return tab, state


# ----------------------------------------------------------------------
# R-015 · visibilidad
# ----------------------------------------------------------------------
def test_el_grupo_aparece_con_datos(app, mensajes, dataset):
    tab, _ = _pestana(dataset)
    assert not tab.gb_comparativa.isHidden()
    assert tab.btn_comparar.isEnabled()


def test_el_grupo_arranca_oculto(app, mensajes):
    """Sin datos cargados el grupo no debe verse ni un instante."""
    from ui.results_tab import ResultsTab

    tab = ResultsTab(AppState())
    assert tab.gb_comparativa.isHidden()


def test_sin_task_type_el_grupo_desaparece(app, mensajes, dataset):
    tab, state = _pestana(dataset)
    state.task_type = None
    tab.refresh()

    assert tab.gb_comparativa.isHidden()
    assert not tab.btn_comparar.isEnabled()


def test_sin_export_df_el_grupo_desaparece(app, mensajes, dataset):
    tab, state = _pestana(dataset)
    state.export_df = None
    tab.refresh()
    assert tab.gb_comparativa.isHidden()


def test_la_metrica_por_defecto_es_una_con_valor(app, mensajes, dataset):
    tab, _ = _pestana(dataset)
    assert tab.cmb_metrica_comparativa.currentText() == "accuracy"


# ----------------------------------------------------------------------
# R-016/R-017/R-018 · tabla, diagrama y resumen
# ----------------------------------------------------------------------
def test_la_tabla_tiene_una_fila_por_modelo(app, mensajes, dataset):
    tab, _ = _pestana(dataset, con_comparativa=True)

    assert tab.tbl_comparativa.rowCount() == 3
    assert tab.tbl_comparativa.item(0, 0).text() == "Random Forest"
    assert tab.tbl_comparativa.item(0, 1).text() == "0.9000"
    assert tab.tbl_comparativa.item(0, 3).text() == "1.00"
    assert tab.tbl_comparativa.item(0, 4).text() == "sí"
    assert tab.tbl_comparativa.item(2, 4).text() == "no"


def test_el_resumen_explica_el_p_valor(app, mensajes, dataset):
    tab, _ = _pestana(dataset, con_comparativa=True)

    assert "Friedman χ²=12.400" in tab.lbl_comparativa.text()
    assert "p=0.0020" in tab.lbl_comparativa.text()
    assert "son estadísticamente significativas" in tab.lbl_comparativa.text()


def test_los_excluidos_se_avisan(app, mensajes, dataset):
    tab, _ = _pestana(dataset, con_comparativa=True)
    assert "KNN raro" in tab.lbl_excluidos.text()


def test_el_diagrama_se_dibuja(app, mensajes, dataset):
    tab, _ = _pestana(dataset, con_comparativa=True)

    assert len(tab.figure_cd.axes) == 1
    eje = tab.figure_cd.axes[0]
    assert len(eje.patches) == 3  # una barra por modelo
    assert eje.get_xlabel() == "Rango medio (menor es mejor)"
    assert "Diferencia crítica" in eje.get_title()


def test_el_diagrama_se_redibuja_sobre_una_comparativa_nueva(app, mensajes, dataset):
    """Redibujar no debe acumular artists de la figura anterior."""
    tab, _ = _pestana(dataset, con_comparativa=True)
    primera = len(tab.figure_cd.axes)
    tab._show_comparison(_comparison_falsa())

    assert len(tab.figure_cd.axes) == primera == 1
    assert len(tab.figure_cd.axes[0].patches) == 3


# ----------------------------------------------------------------------
# R-020 · caché desfasada
# ----------------------------------------------------------------------
def test_avisa_si_la_comparativa_esta_desfasada(app, mensajes, dataset):
    tab, state = _pestana(dataset, con_comparativa=True)
    tab.spn_repeticiones.setValue(7)
    tab._avisar_comparativa_desfasada()

    assert "otra configuración" in tab.lbl_comparativa.text()


def test_no_avisa_si_la_configuracion_sigue_igual(app, mensajes, dataset):
    tab, state = _pestana(dataset, con_comparativa=True)
    tab.spn_repeticiones.setValue(3)
    tab._avisar_comparativa_desfasada()

    assert "otra configuración" not in tab.lbl_comparativa.text()


# ----------------------------------------------------------------------
# R-019 · bloqueo de controles
# ----------------------------------------------------------------------
def test_los_controles_se_bloquean_durante_la_comparativa(app, mensajes, dataset):
    tab, _ = _pestana(dataset)

    tab._set_busy_comparativa(True)
    assert not tab.btn_comparar.isEnabled()
    assert not tab.cmb_metrica_comparativa.isEnabled()
    assert not tab.spn_repeticiones.isEnabled()
    assert not tab.chk_todos_los_modelos.isEnabled()
    assert not tab.progress_comparativa.isHidden()  # visible mientras compara

    tab._set_busy_comparativa(False)
    assert tab.btn_comparar.isEnabled()
    assert tab.spn_repeticiones.isEnabled()
    assert tab.progress_comparativa.isHidden()


def test_el_progreso_muestra_el_avance(app, mensajes, dataset):
    tab, _ = _pestana(dataset)
    tab._on_comparacion_progreso("Comparando modelos: 3/12", 3, 12)

    assert tab.progress_comparativa.maximum() == 12
    assert tab.progress_comparativa.value() == 3


def test_el_fallo_libera_los_controles(app, mensajes, dataset):
    tab, _ = _pestana(dataset)
    tab._set_busy_comparativa(True)

    tab._on_comparacion_fallida("No se pudo comparar")

    assert tab.btn_comparar.isEnabled()
    assert ("critical", "Error al comparar modelos", "No se pudo comparar") in mensajes


# ----------------------------------------------------------------------
# Alcance de los modelos
# ----------------------------------------------------------------------
def test_el_alcance_por_defecto_son_todos_los_compatibles(app, mensajes, dataset):
    tab, state = _pestana(dataset)
    modelos = tab._modelos_a_comparar()

    assert "Random Forest" in modelos
    assert len(modelos) >= 3


def test_sin_la_casilla_se_queda_con_la_familia(app, mensajes, dataset):
    tab, _state = _pestana(dataset)
    tab.chk_todos_los_modelos.setChecked(False)

    modelos = tab._modelos_a_comparar()
    assert modelos, "debe quedar algún modelo de la familia"
    # Solo los de la familia `ensemble` del modelo actual, no los de las demás.
    assert "Regresión Logística" not in modelos
    assert "Random Forest" in modelos


def test_con_menos_de_tres_modelos_avisa(app, mensajes, dataset):
    tab, state = _pestana(dataset)
    state.export_df = dataset[["x0", "objetivo"]]
    tab.chk_todos_los_modelos.setChecked(False)
    state.family = "una-familia-inexistente"

    tab.comparar_modelos()

    assert any(
        categoria == "warning" and titulo == "Modelos insuficientes"
        for categoria, titulo, _texto in mensajes
    )
