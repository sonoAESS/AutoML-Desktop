# ui/enabled.py
"""Utilidades para habilitar y deshabilitar controles en bloque.

Cada pestaña decide su estado leyendo solo `AppState` (AGENTS §5.1). Estas
funciones evitan el error típico de ese patrón: deshabilitar el marco pero
dejar hijos actives, o dejar un control apagado sin decir por qué.
"""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QLineEdit,
    QRadioButton,
    QSpinBox,
    QWidget,
)

#: Tipos de control que tienen sentido habilitar o deshabilitar en bloque.
TIPOS_EDITABLES = (
    QCheckBox,
    QRadioButton,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QLineEdit,
)


def habilitar_hijos(padre: QWidget, habilitado: bool) -> None:
    """Habilita o deshabilita los controles editables que cuelgan de `padre`.

    Se llama «hijos» y no «descendientes» a propósito: no baja hasta los
    nietos, porque eso desactivaría controles de otra sección que solo están
    anidados por el layout.

    Args:
        padre: el marco o grupo que decide.
        habilitado: `True` para activar, `False` para desactivar.
    """
    for hijo in padre.findChildren(QWidget):
        if hijo is padre or not isinstance(hijo, TIPOS_EDITABLES):
            continue
        hijo.setEnabled(habilitado)


def deshabilitar_hijos(padre: QWidget) -> None:
    """Atajo de `habilitar_hijos(padre, False)`."""
    habilitar_hijos(padre, False)


def con_pista(w: QWidget, habilitado: bool, texto: str) -> None:
    """Fija el estado del control y deja el motivo en el tooltip.

    R-010: un control apagado sin explicación parece una aplicación rota, y el
    usuario acaba pulsándolo para descubrir que no hace nada. Por eso el
    motivo se escribe **también cuando queda activo**: así el usuario sabe
    para qué sirve.

    Args:
        w: el control.
        habilitado: si el control puede usarse con el estado actual.
        texto: qué hace y, si está apagado, qué falta para poder usarlo.
    """
    w.setToolTip(texto)
    w.setEnabled(habilitado)
