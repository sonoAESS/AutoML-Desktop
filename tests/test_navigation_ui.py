"""Pruebas de la estructura, navegación y densidad de la interfaz (SDD-008)."""

import os
import re
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QAbstractScrollArea,
    QApplication,
    QFrame,
    QListWidget,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QTableWidget,
    QTabWidget,
    QToolBox,
)

from core.state import AppState  # noqa: E402
from ui.layout import ALTURA_TABLA, acotar_tabla, envolver_scroll  # noqa: E402
from ui.sidebar import ESTADOS, PASOS, StepSidebar  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
NOMBRES_PESTANA = (
    "datos",
    "preprocesamiento",
    "entrenamiento",
    "resultados",
    "prediccion",
)


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def ventana(app):
    from ui.main_window import MainWindow
    from ui.theme import apply_theme

    apply_theme(app)
    v = MainWindow()
    v.resize(1280, 720)
    v.show()
    app.processEvents()
    yield v
    v.close()


# ----------------------------------------------------------------------
# C-001, C-002 · estructura y decoupling
# ----------------------------------------------------------------------
def test_no_hay_pestanas_superiores(ventana):
    """Una barra de pestañas consume alto que hace falta para el contenido."""
    assert isinstance(ventana.pilas, QStackedWidget)
    assert ventana.pilas.count() == len(NOMBRES_PESTANA)


def test_ninguna_pestana_sube_al_padre(ventana):
    """AGENTS §5.1: las pestañas no tocan el widget de la ventana."""
    assert ventana.centralWidget() is not None
    for ruta in sorted((RAIZ / "ui").glob("*.py")):
        if ruta.name == "main_window.py":
            continue
        texto = ruta.read_text(encoding="utf-8")
        assert "self.window()" not in texto, ruta.name
        assert "centralWidget()" not in texto, ruta.name


def test_cada_pestana_va_en_un_scroll(ventana):
    for indice in range(ventana.pilas.count()):
        area = ventana.pilas.widget(indice)
        assert isinstance(area, QScrollArea), indice
        assert area.widgetResizable()


# ----------------------------------------------------------------------
# C-003, C-004 · barra lateral
# ----------------------------------------------------------------------
def test_los_cinco_pasos_estan_en_orden(app):
    lateral = StepSidebar()
    assert len(lateral.buttons) == 5
    textos = [b.text() for b in lateral.buttons]
    for indice, (_nombre, titulo) in enumerate(PASOS):
        assert titulo in textos[indice]
        assert str(indice + 1) in textos[indice]


def test_solo_hay_un_paso_activo(app):
    lateral = StepSidebar()
    lateral.seleccionar(3)
    activos = [b for b in lateral.buttons if b.property("estado") == "activo"]
    assert len(activos) == 1
    assert activos[0].indice == 3


def test_los_pasos_anteriores_quedan_completados(app):
    lateral = StepSidebar()
    lateral.seleccionar(2)
    estados = [b.property("estado") for b in lateral.buttons]
    assert estados[:2] == ["completado", "completado"]
    assert estados[2] == "activo"
    assert estados[3:] == ["pendiente", "pendiente"]


def test_el_estado_completado_no_es_solo_color(app):
    """R-003 exige doble señal: en escala de grises debe seguir leyéndose."""
    lateral = StepSidebar()
    lateral.marcar_completado(0)
    lateral.seleccionar(1)
    boton = lateral.buttons[0]
    assert boton.property("estado") == "completado"
    assert boton.text().startswith("✓"), "falta la señal no cromática"


def test_un_estado_inventado_falla(app):
    lateral = StepSidebar()
    with pytest.raises(ValueError):
        lateral.set_estado(0, "inventado")
    with pytest.raises(ValueError):
        lateral.set_estado(0, "terminado")


def test_los_estados_declarados_existen_en_la_hoja():
    """`completado` necesita regla propia; `pendiente` es el aspecto base."""
    from ui.theme import build_stylesheet

    qss = build_stylesheet()
    assert 'QPushButton#paso[estado="completado"]' in qss
    assert 'QPushButton#paso[estado="pendiente"]' not in qss
    assert set(ESTADOS) == {"pendiente", "activo", "completado"}


def test_la_hoja_distingue_el_paso_activo(app):
    from ui.theme import build_stylesheet

    qss = build_stylesheet()
    assert "QPushButton#paso:checked" in qss
    assert "border-left: 3px solid" in qss


# ----------------------------------------------------------------------
# C-005, C-006 · navegación
# ----------------------------------------------------------------------
def test_ir_a_paso_cambia_el_contenido_y_el_titulo(ventana):
    ventana.ir_a_paso(2)
    ventana.pilas.currentWidget().widget()
    assert ventana.pilas.currentIndex() == 2
    assert ventana.sidebar.paso_actual() == 2
    assert "Entrenamiento" in ventana.lbl_paso.text()


def test_ir_a_paso_no_pierde_los_datos(ventana):
    import pandas as pd

    ventana.state.clean_df = pd.DataFrame({"a": [1, 2, 3]})
    ventana.state.target_column = "a"
    ventana.ir_a_paso(3)
    ventana.ir_a_paso(0)
    assert ventana.state.target_column == "a"
    assert len(ventana.state.clean_df) == 3


def test_ir_a_paso_con_indice_invalido_no_revienta(ventana):
    ventana.ir_a_paso(99)
    ventana.ir_a_paso(-1)
    assert ventana.pilas.currentIndex() in range(5)


def test_el_boton_de_prediccion_navega_por_senal(ventana):
    """El recorrido completo: botón → señal → contenido."""
    ventana.results_tab.btn_load_in_predict.click()
    assert ventana.pilas.currentIndex() == 4


def test_el_atajo_de_teclado_lleva_al_paso(ventana, app):
    ventana.ir_a_paso(0)
    ventana.ir_a_paso(2)
    assert ventana.sidebar.paso_actual() == 2


def test_paso_siguiente_da_la_vuelta(ventana):
    ventana.ir_a_paso(4)
    ventana.paso_siguiente()
    assert ventana.pilas.currentIndex() == 0


def test_ir_a_paso_nombre_desconocido_no_hace_nada(ventana):
    antes = ventana.pilas.currentIndex()
    ventana.ir_a_paso_nombre("no-existe")
    assert ventana.pilas.currentIndex() == antes


# ----------------------------------------------------------------------
# C-008, C-009, C-010, C-011 · densidad
# ----------------------------------------------------------------------
def test_resultados_tiene_tres_subpestanas(ventana):
    assert ventana.results_tab.subpestanas.count() == 3
    titulos = [ventana.results_tab.subpestanas.tabText(i) for i in range(3)]
    assert titulos == ["Métricas", "Comparativa", "Exportar"]


def test_preprocesamiento_y_entrenamiento_son_acordeones(ventana):
    assert isinstance(ventana.preprocess_tab.grupos, QToolBox)
    assert isinstance(ventana.train_tab.grupos, QToolBox)


def test_solo_una_pagina_del_acordeon_visible(ventana, app):
    forGroups = (ventana.preprocess_tab, ventana.train_tab)
    for tab in forGroups:
        visibles = [
            i for i in range(tab.grupos.count()) if tab.grupos.widget(i).isVisible()
        ]
        assert len(visibles) <= 1, tab


def test_toda_tabla_y_lista_tiene_alto_acotado(ventana):
    tablas = ventana.findChildren(QTableWidget) + ventana.findChildren(QListWidget)
    assert tablas, "debería haber tablas en la ventana"
    for tabla in tablas:
        assert tabla.maximumHeight() > 0, tabla.objectName() or tabla
    assert ALTURA_TABLA > 0


def test_los_lienzos_se_expanden(ventana):
    from PySide6.QtWidgets import QSizePolicy

    lienzos = [
        ventana.results_tab.canvas,
        ventana.results_tab.canvas_cd,
    ]
    for lienzo in lienzos:
        assert lienzo.sizePolicy().horizontalPolicy() == QSizePolicy.Expanding
        assert lienzo.sizePolicy().verticalPolicy() == QSizePolicy.Expanding


def test_las_figuras_usan_tight_layout(ventana, monkeypatch):
    """Sin tight_layout las etiquetas largas se cortan al redimensionar."""
    import pandas as pd
    from matplotlib.figure import Figure

    llamadas = []
    original = Figure.tight_layout

    def espia(self, *args, **kwargs):
        llamadas.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "tight_layout", espia)

    ventana.state.metrics = {"accuracy": 0.9, "f1_macro": 0.88, "n_test": 30}
    ventana.state.task_type = "classification"
    ventana.state.target_column = "objetivo"
    ventana.ir_a_paso(3)
    ventana.results_tab.refresh()

    assert llamadas, "el dibujo debe llamar a tight_layout"


# ----------------------------------------------------------------------
# C-007 · geometría real a 1280×720
# ----------------------------------------------------------------------
def _visible(widget) -> bool:
    """Si el widget está dentro del área visible de su ventana."""
    ventana = widget.window()
    punto = widget.mapTo(ventana, widget.rect().topLeft())
    area = ventana.rect()
    return punto.y() + widget.height() <= area.bottom() + 1


def test_el_boton_de_entrenar_cabe_en_720(ventana):
    ventana.ir_a_paso(2)
    ventana.pilas.currentWidget().widget().adjustSize()
    QApplication.instance().processEvents()
    assert _visible(
        ventana.train_tab.btn_train
    ), f"el botón queda en y={ventana.train_tab.btn_train.y()}"


def test_el_boton_de_aplicar_cabe_en_720(ventana):
    ventana.ir_a_paso(1)
    QApplication.instance().processEvents()
    assert _visible(ventana.preprocess_tab.btn_apply)


def test_la_ventana_no_necesita_mas_de_720_de_alto(ventana):
    """Antes la ventana pedía 820 px, más que la altura útil de un portátil."""
    ventana.ir_a_paso(0)
    QApplication.instance().processEvents()
    assert ventana.minimumSizeHint().height() <= 720


# ----------------------------------------------------------------------
# C-013 · contrato de atributos con los tests existentes
# ----------------------------------------------------------------------
CONTRATO = {
    "datos": [],
    "preprocesamiento": [
        "gb_distribucion",
        "tbl_distribucion",
        "lbl_distribucion",
    ],
    "entrenamiento": [
        "cmb_task",
        "cmb_model",
        "cmb_balancing",
        "spn_vecinos",
        "chk_seleccion",
        "cmb_metodo_sel",
        "cmb_criterio_sel",
        "spn_sel",
        "tbl_seleccion",
        "btn_analizar",
        "lbl_sel",
        "gb_balanceo",
        "_param_widgets",
        "_balancing_config",
        "_selection_config",
        "_on_task_changed",
        "btn_train",
    ],
    "resultados": [
        "tbl_comparativa",
        "lbl_comparativa",
        "lbl_excluidos",
        "figure_cd",
        "canvas_cd",
        "btn_comparar",
        "cmb_metrica_comparativa",
        "spn_repeticiones",
        "chk_todos_los_modelos",
        "progress_comparativa",
        "gb_comparativa",
        "tbl_metricas",
        "canvas",
        "navigate_requested",
        "_set_busy_comparativa",
        "_modelos_a_comparar",
        "_avisar_comparativa_desfasada",
        "_show_comparison",
    ],
    "prediccion": ["predict", "export_predictions", "_predictions", "_preview"],
}


def test_los_atributos_del_contrato_siguen_existiendo(ventana):
    """R-017: reorganizar no puede renombrar lo que los tests tocan."""
    faltan = []
    for nombre_pestana, atributos in CONTRATO.items():
        pestaña = ventana.pestanas[nombre_pestana]
        for atributo in atributos:
            if not hasattr(pestaña, atributo):
                faltan.append(f"{nombre_pestana}.{atributo}")
    assert not faltan, faltan


# ----------------------------------------------------------------------
# utilidades
# ----------------------------------------------------------------------
def test_envolver_scroll_no_pone_marco(app):
    from PySide6.QtWidgets import QWidget

    area = envolver_scroll(QWidget())
    assert area.frameShape() == QFrame.NoFrame
    assert area.widgetResizable()


def test_el_contenido_se_puede_desplazar(app):
    # Si el contenido no cabe, aparece la barra en vez de recortarse.
    from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

    contenido = QWidget()
    vertical = QVBoxLayout(contenido)
    for i in range(8):
        etiqueta = QLabel(f"Bloque {i}")
        etiqueta.setFixedHeight(90)
        vertical.addWidget(etiqueta)

    area = envolver_scroll(contenido)
    area.resize(400, 150)
    area.show()
    QApplication.instance().processEvents()
    assert area.verticalScrollBar().maximum() > 0
    area.close()


def test_una_tabla_no_agranda_la_pestana(app):
    # El motivo del scroll: `QTableWidget` crece con las filas y empujaba los
    # botones fuera de la ventana.
    from PySide6.QtWidgets import QVBoxLayout, QWidget

    contenido = QWidget()
    vertical = QVBoxLayout(contenido)
    tabla = QTableWidget(80, 4)
    acotar_tabla(tabla)
    vertical.addWidget(tabla)
    boton = QPushButton("Entrenar modelo")
    vertical.addWidget(boton)

    assert tabla.maximumHeight() <= ALTURA_TABLA
