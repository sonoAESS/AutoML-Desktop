# ui/file_dialogs.py
"""Diálogos de selección de ficheros compartidos por las pestañas.

Solo conoce widgets de Qt y el filtro multiformato; no contiene lógica de
datos ni de ML (ver `AGENTS.md` §3.5).
"""

from __future__ import annotations

import re

from PySide6.QtWidgets import QFileDialog, QInputDialog, QWidget

TABLE_FILTER = (
    "CSV (*.csv *.txt *.tsv);;"
    "Excel (*.xlsx *.xlsm *.xls);;"
    "OpenDocument (*.ods);;"
    "Todos los archivos (*)"
)


def choose_table_file(
    parent: QWidget | None = None,
    caption: str = "Seleccionar fichero de datos",
) -> str | None:
    """Abre el diálogo de apertura con el filtro multiformato.

    Returns:
        La ruta elegida, o `None` si el usuario cancela.
    """
    path, _ = QFileDialog.getOpenFileName(parent, caption, "", TABLE_FILTER)
    return path or None


def choose_sheet(
    parent: QWidget | None,
    sheets: list[str],
    titulo: str = "El libro tiene varias hojas",
    mensaje: str = "¿Qué hoja quieres cargar?",
) -> str | None:
    """Pregunta qué hoja usar si el libro tiene más de una.

    Returns:
        El nombre de la hoja elegida, o `None` si el usuario cancela. Con menos
        de dos hojas devuelve la única sin preguntar.
    """
    hojas = list(sheets)
    if not hojas:
        return None
    if len(hojas) == 1:
        return hojas[0]
    elegido, aceptado = QInputDialog.getItem(parent, titulo, mensaje, hojas, 0, False)
    return elegido if aceptado and elegido else None


def choose_save_path(
    parent: QWidget | None,
    caption: str,
    default_name: str,
    filtros: str,
) -> str | None:
    """Abre el diálogo de guardado y normaliza la extensión.

    Si el usuario escribe un nombre sin extensión, se añade la del primer
    filtro elegido.

    Returns:
        La ruta de destino, o `None` si el usuario cancela.
    """
    destino, filtro = QFileDialog.getSaveFileName(
        parent, caption, default_name, filtros
    )
    if not destino:
        return None
    return destino if _con_extension(destino) else destino + _extension_de(filtro)


def _filtros(filtros: str) -> list[str]:
    """Patrones de un filtro de Qt (`"CSV (*.csv)"` → `[".csv]"`)."""
    encontrados: list[str] = []
    for patron in re.findall(r"\*(\.[A-Za-z0-9]+)", filtros):
        if patron not in encontrados:
            encontrados.append(patron)
    return encontrados


def _extension_de(filtro: str | None) -> str:
    if not filtro:
        return ""
    extensiones = _filtros(filtro)
    return extensiones[0] if extensiones else ""


def _con_extension(ruta: str) -> bool:
    return bool(re.search(r"\.[A-Za-z0-9]+$", ruta))
