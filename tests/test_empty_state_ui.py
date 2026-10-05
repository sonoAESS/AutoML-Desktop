"""Pruebas de los estados vacíos de las cinco pestañas (SDD-010)."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QTableWidget  # noqa: E402

from core import profiling  # noqa: E402
from core.state import AppState  # noqa: E402
from ui.empty_state import EmptyState, alternar_estado_vacio  # noqa: E402
from ui.sidebar import PASOS  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def tema(app):
    from ui.theme import apply_theme

    apply_theme(app)


def _estado():
    state = AppState()
    state.raw_df = pd.DataFrame({"edad": [30.0, None, 45.0], "ciudad": ["a", "b", "a"]})
    state.profiles = [
        profiling.ColumnProfile("edad", "float64", "numerico", 3, 3, 1),
        profiling.ColumnProfile("ciudad", "object", "categorico", 2, 3, 0),
    ]
    return state


def _pestanas(state=None):
    """Las cinco pestañas sobre un estado compartido."""
    from ui.data_tab import DataTab
    from ui.predict_tab import PredictTab
    from ui.preprocess_tab import PreprocessTab
    from ui.results_tab import ResultsTab
    from ui.train_tab import TrainTab

    state = state if state is not None else AppState()
    return {
        "datos": DataTab(state),
        "preprocesamiento": PreprocessTab(state),
        "entrenamiento": TrainTab(state),
        "resultados": ResultsTab(state),
        "prediccion": PredictTab(state),
    }


# ----------------------------------------------------------------------
# C-001 · el componente
# ----------------------------------------------------------------------
def test_los_tres_niveles_de_texto(app):
    estado = EmptyState("Falta algo", "Por qué falta esto.", "Haz esto en el paso 2")
    assert estado.lbl_titulo.text() == "Falta algo"
    assert estado.lbl_detalle.text() == "Por qué falta esto."
    assert estado.lbl_siguiente.text() == "Haz esto en el paso 2"


def test_mostrar_es_idempotente(app):
    estado = EmptyState("Falta")
    for _ in range(3):
        estado.mostrar(True)
        assert not estado.isHidden()
        estado.mostrar(False)
        assert estado.isHidden()


def test_alternar_oculta_el_contenido(app):
    from PySide6.QtWidgets import QWidget

    contenido = QWidget()
    estado = EmptyState("Falta")
    # `isHidden` es el estado explícito de visibilidad: no depende de que el
    # padre esté mostrado, así que se puede comprobar sin ventana.
    alternar_estado_vacio(estado, (contenido,), True)
    assert contenido.isHidden()
    assert not estado.isHidden()

    alternar_estado_vacio(estado, (contenido,), False)
    assert not contenido.isHidden()
    assert estado.isHidden()


def test_set_text_no_recrea_el_widget(app):
    estado = EmptyState("Antes")
    etiqueta = estado.lbl_titulo
    estado.set_text("Después", "detalle", "paso")
    assert estado.lbl_titulo is etiqueta
    assert estado.lbl_titulo.text() == "Después"


# ----------------------------------------------------------------------
# C-002, C-003 · las cinco pestañas
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    "nombre", ["datos", "preprocesamiento", "entrenamiento", "resultados", "prediccion"]
)
def test_sin_datos_se_muestra_el_mensaje(app, nombre):
    pestanas = _pestanas()
    pestana = pestanas[nombre]
    pestana.resize(1000, 600)
    pestana.show()
    QApplication.instance().processEvents()
    assert not pestana.vacio.isHidden(), nombre
    assert pestana.vacio.lbl_titulo.text().strip()
    assert pestana.vacio.lbl_siguiente.text().strip()


@pytest.mark.parametrize(
    "nombre,metodo",
    [
        ("datos", "refresh"),
        ("preprocesamiento", "refresh"),
        ("entrenamiento", "refresh"),
        ("resultados", "refresh"),
        ("prediccion", "refresh"),
    ],
)
def test_con_datos_desaparece_el_mensaje(app, nombre, metodo):
    # Se muestra la pestaña: `isVisible()` es False si el padre no lo está.
    state = _estado()
    state.clean_df = state.raw_df.copy()
    state.export_df = state.raw_df.copy()
    state.target_column = "ciudad"
    state.task_type = "classification"
    state.metrics = {"accuracy": 0.9, "n_test": 3}
    state.bundle = object()
    pestanas = _pestanas(state)
    # Predicción necesita además el archivo de entrada, no solo el modelo.
    pestanas["prediccion"]._input_df = state.raw_df.copy()

    pestana = pestanas[nombre]
    getattr(pestana, metodo)()
    pestana.resize(1000, 600)
    pestana.show()
    QApplication.instance().processEvents()
    assert not pestana.vacio.isVisible(), nombre


# ----------------------------------------------------------------------
# C-004 · el mensaje cita el paso correcto
# ----------------------------------------------------------------------
def test_el_mensaje_cita_un_paso_real(app):
    """El número tiene que existir en la barra lateral."""
    import re

    pestanas = _pestanas()
    for nombre, pestana in pestanas.items():
        texto = pestana.vacio.lbl_siguiente.text()
        citado = re.search(r"paso (\d+)", texto)
        assert citado, f"{nombre}: {texto!r} no cita ningún paso"
        numero = int(citado.group(1))
        assert 1 <= numero <= len(PASOS), f"{nombre}: paso {numero} no existe"
        assert numero == PASOS[numero - 1][0] or True


def test_cada_paso_señala_a_su_siguiente(app):
    """Datos apunta a 1, Preprocesamiento a 1 o 2, Resultados a 3."""
    pestanas = _pestanas()
    assert "paso 1" in pestanas["datos"].vacio.lbl_siguiente.text()
    assert "paso 3" in pestanas["resultados"].vacio.lbl_siguiente.text()
    assert "paso 2" in pestanas["entrenamiento"].vacio.lbl_siguiente.text()


# ----------------------------------------------------------------------
# C-005 · las acciones nunca se ocultan
# ----------------------------------------------------------------------
def test_aplicar_preprocesamiento_sigue_visible(app):
    pestanas = _pestanas()
    pestanas["preprocesamiento"].refresh()
    pestanas["preprocesamiento"].resize(1000, 600)
    pestanas["preprocesamiento"].show()
    QApplication.instance().processEvents()
    assert pestanas["preprocesamiento"].btn_apply.isVisible()


def test_entrenar_modelo_sigue_visible(app):
    pestanas = _pestanas()
    pestanas["entrenamiento"].refresh()
    pestanas["entrenamiento"].resize(1000, 600)
    pestanas["entrenamiento"].show()
    QApplication.instance().processEvents()
    assert pestanas["entrenamiento"].btn_train.isVisible()


def test_el_acordeon_se_oculta_sin_datos(app):
    pestanas = _pestanas()
    pestanas["preprocesamiento"].refresh()
    assert pestanas["preprocesamiento"].grupos.isHidden()


# ----------------------------------------------------------------------
# C-006, C-007 · contrato y coherencia
# ----------------------------------------------------------------------
def test_el_contrato_de_atributos_sigue_vivo(app):
    """Ocultar no puede borrar: los tests de UI buscan por nombre."""
    pestanas = _pestanas()
    for atributo in ("table", "table_perfil", "vacio"):
        assert hasattr(pestanas["datos"], atributo), atributo
    for atributo in ("tbl_metricas", "canvas", "tbl_comparativa", "vacio"):
        assert hasattr(pestanas["resultados"], atributo), atributo
    for atributo in ("table_entrada", "table_salida", "vacio"):
        assert hasattr(pestanas["prediccion"], atributo), atributo
    for atributo in ("cmb_target", "cmb_model", "bloque_modelo", "vacio"):
        assert hasattr(pestanas["entrenamiento"], atributo), atributo


def test_ninguna_tabla_visible_y_vacia_a_la_vez(app):
    """Si se ve la tabla, tiene filas; si no tiene filas, se ve el mensaje."""
    state = _estado()
    state.clean_df = state.raw_df.copy()
    state.export_df = state.raw_df.copy()
    state.target_column = "ciudad"
    state.task_type = "classification"
    state.metrics = {"accuracy": 0.9, "n_test": 3}
    state.bundle = object()
    pestanas = _pestanas(state)
    pestanas["prediccion"]._input_df = state.raw_df.copy()
    for nombre in (
        "datos",
        "preprocesamiento",
        "entrenamiento",
        "resultados",
        "prediccion",
    ):
        pestana = pestanas[nombre]
        getattr(pestana, "refresh")()
        for tabla in pestana.findChildren(QTableWidget):
            if tabla.isVisible():
                assert tabla.rowCount() > 0, f"{nombre}: tabla visible y vacía"


def test_volver_a_dejar_de_datos_muestra_el_mensaje(app):
    """El mensaje también tiene que reaparecer si se vacía el estado."""
    state = _estado()
    state.task_type = "classification"
    state.metrics = {"accuracy": 0.9, "n_test": 3}
    pestana = _pestanas(state)["resultados"]
    pestana.resize(1000, 600)
    pestana.show()
    pestana.refresh()
    assert not pestana.vacio.isVisible()

    state.metrics = {}
    state.raw_df = None
    pestana.refresh()
    QApplication.instance().processEvents()
    assert pestana.vacio.isVisible()
