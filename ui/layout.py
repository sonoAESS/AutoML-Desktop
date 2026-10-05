# ui/layout.py
"""Utilidades de distribución compartidas por las cinco pestañas.

No es un módulo `catch-all`: solo resuelve el problema que comparten las
pestañas, que es que su contenido no cabe en una ventana de 1366×768.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QListWidget,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
)

#: Alto máximo por defecto de una tabla o lista. Sin tope, `QTableWidget`
#: crece con las filas dentro del layout y empuja los botones fuera de la
#: ventana: es lo que pasaba con «Entrenar modelo».
ALTURA_TABLA = 180


def envolver_scroll(widget: QFrame) -> QScrollArea:
    """Envuelve el contenido de una pestaña en un área con desplazamiento.

    Envuelve el *contenido*, no la pestaña: el `QStackedWidget` recibe el
    área, para que la barra de desplazamiento sea de la pestaña y no de la
    ventana entera.

    Args:
        widget: la pestaña o el contenido a envolver.

    Returns:
        El `QScrollArea`, listo para añadir a un layout.
    """
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    area.setWidget(widget)
    return area


def acotar_tabla(tabla: QTableWidget | QListWidget, alto: int = ALTURA_TABLA) -> None:
    """Fija un alto máximo y configura el comportamiento de selección.

    Args:
        tabla: la tabla o lista a acotar.
        alto: alto máximo en píxeles.
    """
    tabla.setMaximumHeight(alto)
    tabla.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    if isinstance(tabla, QTableWidget):
        tabla.setAlternatingRowColors(True)
        tabla.setEditTriggers(QAbstractItemView.NoEditTriggers)


def expandir_canvas(canvas) -> None:
    """Deja que un lienzo de matplotlib ocupe todo el espacio disponible.

    Sin esto el lienzo hereda la altura del layout padre y la figura queda
    descentrada o recortada al redimensionar la ventana.
    """
    canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    canvas.setMinimumHeight(200)
