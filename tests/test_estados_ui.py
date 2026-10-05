"""Pruebas de los estados de los controles y de la densidad (SDD-009)."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import (  # noqa: E402
    QAbstractButton,
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QListWidget,
    QPushButton,
    QRadioButton,
    QSpinBox,
)

from core import profiling  # noqa: E402
from core.state import AppState  # noqa: E402
from ui.enabled import TIPOS_EDITABLES, habilitar_hijos  # noqa: E402

MAX_TEXTO_BOTON = 28
MAX_WIDGETS_POR_FILA = 5


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def tema(app):
    from ui.theme import apply_theme

    apply_theme(app)


def _perfil(nombre, tipo):
    return profiling.ColumnProfile(nombre, "object", tipo, 3, 3, 0)


def _dataset():
    return pd.DataFrame(
        {"edad": [30.0, None, 45.0, 61.0], "ciudad": ["a", None, "b", "a"]}
    )


def _pestanas():
    from ui.data_tab import DataTab
    from ui.predict_tab import PredictTab
    from ui.preprocess_tab import PreprocessTab
    from ui.results_tab import ResultsTab
    from ui.train_tab import TrainTab

    state = AppState()
    state.raw_df = _dataset()
    state.profiles = [
        _perfil("edad", "numerico"),
        _perfil("ciudad", "categorico"),
    ]
    return {
        "datos": DataTab(state),
        "preprocesamiento": PreprocessTab(state),
        "entrenamiento": TrainTab(state),
        "resultados": ResultsTab(state),
        "prediccion": PredictTab(state),
    }, state


def _controles(pestana):
    """Controles interactivos de una pestaña."""
    tipos = (
        QCheckBox,
        QRadioButton,
        QComboBox,
        QSpinBox,
        QDoubleSpinBox,
        QLineEdit,
        QPushButton,
    )
    encontrados = []
    for tipo in tipos:
        encontrados += pestana.findChildren(tipo)
    return encontrados


# ----------------------------------------------------------------------
# C-001, C-002 · preprocesamiento
# ----------------------------------------------------------------------
def test_sin_datos_no_hay_nada_habilitado(app):
    from ui.preprocess_tab import PreprocessTab

    pestana = PreprocessTab(AppState())
    pestana.refresh()
    encendidos = [c for c in _controles(pestana) if c.isEnabled()]
    assert not encendidos, [type(c).__name__ for c in encendidos]


def test_con_datos_se_habilita_lo_aplicable(app):
    from ui.preprocess_tab import PreprocessTab

    state = AppState()
    state.raw_df = _dataset()
    state.profiles = [_perfil("edad", "numerico"), _perfil("ciudad", "categorico")]
    pestana = PreprocessTab(state)
    pestana.refresh()

    assert pestana.btn_apply.isEnabled()
    assert pestana.chk_dups.isEnabled()
    assert pestana.cmb_num.isEnabled()
    assert pestana.cmb_cat.isEnabled()


def test_sin_numericas_se_apaga_la_estrategia_numerica(app):
    from ui.preprocess_tab import PreprocessTab

    state = AppState()
    state.raw_df = _dataset()
    state.profiles = [_perfil("ciudad", "categorico")]
    pestana = PreprocessTab(state)
    pestana.refresh()

    assert not pestana.cmb_num.isEnabled()
    assert pestana.cmb_cat.isEnabled()


def test_sin_categoricas_se_apaga_la_estrategia_categorica(app):
    from ui.preprocess_tab import PreprocessTab

    state = AppState()
    state.raw_df = _dataset()
    state.profiles = [_perfil("edad", "numerico")]
    pestana = PreprocessTab(state)
    pestana.refresh()

    assert not pestana.cmb_cat.isEnabled()
    assert pestana.cmb_num.isEnabled()


def test_un_control_apagado_explica_por_que(app):
    """R-010: sin motivo, el usuario lo pulsa y no pasa nada."""
    from ui.preprocess_tab import PreprocessTab

    state = AppState()
    state.raw_df = _dataset()
    state.profiles = [_perfil("edad", "numerico")]
    pestana = PreprocessTab(state)
    pestana.refresh()

    assert "categóricas" in pestana.cmb_cat.toolTip()


def test_ningun_control_apagado_se_queda_sin_tooltip(app):
    _, state = _pestanas()
    from ui.preprocess_tab import PreprocessTab

    state.raw_df = None
    pestana = PreprocessTab(state)
    pestana.refresh()
    for control in _controles(pestana):
        if not control.isEnabled():
            assert control.toolTip().strip(), control


# ----------------------------------------------------------------------
# C-003 · idempotencia
# ----------------------------------------------------------------------
@pytest.mark.parametrize("nombre", ["datos", "preprocesamiento", "resultados"])
def test_el_estado_es_idempotente(app, nombre):
    _, state = _pestanas()
    pestana = {
        "datos": _pestanas()[0]["datos"],
        "preprocesamiento": _pestanas()[0]["preprocesamiento"],
        "resultados": _pestanas()[0]["resultados"],
    }[nombre]

    pestana._update_enabled_state()
    antes = [c.isEnabled() for c in _controles(pestana)]
    pestana._update_enabled_state()
    pestana._update_enabled_state()
    despues = [c.isEnabled() for c in _controles(pestana)]

    assert antes == despues


# ----------------------------------------------------------------------
# C-004 · jerarquía
# ----------------------------------------------------------------------
def test_habilitar_hijos_desactiva_en_cascada(app):
    from PySide6.QtWidgets import QGroupBox, QVBoxLayout

    marco = QGroupBox("grupo")
    layout = QVBoxLayout(marco)
    combo = QComboBox()
    casilla = QCheckBox("x")
    layout.addWidget(combo)
    layout.addWidget(casilla)

    habilitar_hijos(marco, False)
    assert not combo.isEnabled()
    assert not casilla.isEnabled()

    habilitar_hijos(marco, True)
    assert combo.isEnabled()
    assert casilla.isEnabled()


def test_habilitar_hijos_no_toca_a_los_que_no_son_editables(app):
    from PySide6.QtWidgets import QGroupBox, QLabel, QVBoxLayout

    marco = QGroupBox("grupo")
    layout = QVBoxLayout(marco)
    boton = QPushButton("acción")
    layout.addWidget(boton)

    habilitar_hijos(marco, False)
    assert boton.isEnabled(), "un botón no es un campo editable"


# ----------------------------------------------------------------------
# C-005 · resultados
# ----------------------------------------------------------------------
def test_sin_modelo_los_tres_botones_estan_apagados(app):
    _, state = _pestanas()
    from ui.results_tab import ResultsTab

    pestana = ResultsTab(state)
    pestana.refresh()

    for boton in (pestana.btn_save, pestana.btn_export, pestana.btn_load_in_predict):
        assert not boton.isEnabled()
        assert "Entrena" in boton.toolTip(), boton.text()


def test_con_modelo_se_habilitan(app):
    _, state = _pestanas()
    from ui.results_tab import ResultsTab

    state.pipeline = object()
    state.metrics = {"accuracy": 0.9}
    pestana = ResultsTab(state)
    pestana.refresh()

    assert pestana.btn_save.isEnabled()
    assert pestana.btn_export.isEnabled()
    assert pestana.btn_load_in_predict.isEnabled()


# ----------------------------------------------------------------------
# C-006 · predicción
# ----------------------------------------------------------------------
def test_prediccion_tres_estados(app):
    from ui.predict_tab import PredictTab

    state = AppState()
    pestana = PredictTab(state)
    pestana.refresh()
    assert not pestana.btn_predict.isEnabled()
    assert "modelo" in pestana.btn_predict.toolTip()

    state.bundle = object()
    pestana.refresh()
    assert not pestana.btn_predict.isEnabled()
    assert "CSV" in pestana.btn_predict.toolTip()

    pestana._input_df = pd.DataFrame({"edad": [1.0]})
    pestana.refresh()
    assert pestana.btn_predict.isEnabled()

    pestana._predictions = pd.DataFrame({"prediccion": [1]})
    pestana.refresh()
    assert pestana.btn_export.isEnabled()
    assert pestana.btn_add_to_data.isEnabled()


# ----------------------------------------------------------------------
# C-007, C-008 · textos y tooltips
# ----------------------------------------------------------------------
def test_ningun_boton_supera_los_28_caracteres(app):
    from ui.main_window import MainWindow

    ventana = MainWindow()
    largos = [
        (b.text(), len(b.text()))
        for b in ventana.findChildren(QPushButton)
        if len(b.text()) > MAX_TEXTO_BOTON
    ]
    assert not largos, largos


def test_el_boton_largo_movio_el_detalle_al_tooltip(app):
    from ui.results_tab import ResultsTab

    state = AppState()
    state.pipeline = object()
    pestana = ResultsTab(state)
    pestana.refresh()

    # El texto visible se acortó y el detalle pasó al tooltip.
    assert pestana.btn_save.text() == "Guardar proyecto"
    assert "automl" in pestana.btn_save.toolTip()
    assert len(pestana.btn_save.text()) <= MAX_TEXTO_BOTON


def test_los_botones_de_accion_explican_su_efecto(app):
    from ui.main_window import MainWindow

    ventana = MainWindow()
    for boton in ventana.findChildren(QPushButton):
        if boton.property("clase") in {"primario", "peligro"}:
            assert boton.toolTip().strip(), boton.text()


def test_toda_casilla_tiene_tooltip(app):
    # Una casilla no lleva etiqueta al lado: sin tooltip el usuario tiene que
    # adivinar qué activa.
    from ui.main_window import MainWindow

    ventana = MainWindow()
    sin_pista = [
        c.text() for c in ventana.findChildren(QCheckBox) if not c.toolTip().strip()
    ]
    assert not sin_pista, sin_pista


# ----------------------------------------------------------------------
# C-009 · densidad de las filas
# ----------------------------------------------------------------------
def test_ninguna_fila_supera_los_5_widgets(app):
    """Filas de seis controles no caben en 1366 px."""
    from PySide6.QtWidgets import QLayout

    from ui.main_window import MainWindow

    ventana = MainWindow()
    excesos = []
    for layout in ventana.findChildren(QLayout):
        total = layout.count()
        if total > MAX_WIDGETS_POR_FILA and layout.__class__.__name__ == "QHBoxLayout":
            # Solo cuentan los controles, no los separadores.
            controles = sum(
                1 for i in range(total) if layout.itemAt(i).widget() is not None
            )
            if controles > MAX_WIDGETS_POR_FILA:
                excesos.append(controles)
    assert not excesos, sorted(excedos)


# ----------------------------------------------------------------------
# utilidades
# ----------------------------------------------------------------------
def test_los_tipos_editables_incluyen_los_campos(app):
    for tipo in (QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit):
        assert tipo in TIPOS_EDITABLES, tipo
    assert QPushButton not in TIPOS_EDITABLES
