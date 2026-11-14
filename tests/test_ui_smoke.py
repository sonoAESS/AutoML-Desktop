"""Arranque de la ventana sin pantalla (offscreen).

Este test corre sobre el código fuente; `test_binary_smoke.py` corre sobre el
paquete ya compilado. Los dos hacen falta: el del binario comprueba que los
plugins de Qt se empaquetaron, y este comprueba lo que el otro no puede ver
desde fuera, que es si el icono SVG se carga de verdad.

Es justo el fallo que no da error: si falta el plugin `qsvg`, `QIcon` no
revienta, se queda vacío y la ventana aparece sin icono.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402

from ui.main_window import MainWindow  # noqa: E402
from ui.theme import apply_theme, escudo_pixmap, icono_app  # noqa: E402


def test_la_ventana_arranca_y_los_recursos_cargan():
    app = QApplication.instance() or QApplication([])
    apply_theme(app)

    ventana = MainWindow()
    try:
        assert ventana.windowTitle(), "la ventana se quedó sin título"
        assert not icono_app().isNull(), (
            "el icono SVG no se pudo cargar: ¿falta el plugin imageformats/qsvg "
            "o la librería Qt6Svg?"
        )
        assert (
            not escudo_pixmap().isNull()
        ), "el escudo PNG no se pudo cargar: ¿se empaquetó resources/escudo-mec.png?"
    finally:
        ventana.close()
