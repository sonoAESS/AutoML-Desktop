# ui/empty_state.py
"""Mensaje de «aquí no hay nada todavía», con la acción que falta.

Una tabla vacía con su cabecera no comunica nada: parece una aplicación rota
en lugar de una pestaña a medio hacer. Este componente sustituye al contenido
vacío por un texto que dice **qué falta** y **qué hacer a continuación**.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

from ui.theme import ESPACIOS


class EmptyState(QWidget):
    """Tres niveles de texto: qué falta, por qué, y el paso siguiente."""

    def __init__(
        self,
        titulo: str = "",
        detalle: str = "",
        siguiente: str = "",
        parent=None,
    ):
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, ESPACIOS["xl"], 0, ESPACIOS["xl"])
        layout.setSpacing(ESPACIOS["sm"])
        layout.addStretch(1)

        self.lbl_titulo = self._etiqueta("titulo", titulo)
        layout.addWidget(self.lbl_titulo)

        self.lbl_detalle = self._etiqueta("suave", detalle)
        layout.addWidget(self.lbl_detalle)

        self.lbl_siguiente = self._etiqueta("seccion", siguiente)
        layout.addWidget(self.lbl_siguiente)

        layout.addStretch(1)

        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setVisible(False)

    # ------------------------------------------------------------------
    @staticmethod
    def _etiqueta(rol: str, texto: str) -> QLabel:
        etiqueta = QLabel(texto)
        etiqueta.setProperty("role", rol)
        etiqueta.setWordWrap(True)
        etiqueta.setAlignment(Qt.AlignCenter)
        return etiqueta

    def set_text(self, titulo: str, detalle: str = "", siguiente: str = "") -> None:
        """Cambia los tres textos sin recrear el widget."""
        self.lbl_titulo.setText(titulo)
        self.lbl_detalle.setText(detalle)
        self.lbl_siguiente.setText(siguiente)

    def mostrar(self, visible: bool) -> None:
        """Muestra u oculta el mensaje. Idempotente."""
        self.setVisible(bool(visible))

    def texto(self) -> str:
        """Todo el texto del mensaje, para poder buscarlo en los tests."""
        return " ".join(
            etiqueta.text()
            for etiqueta in (self.lbl_titulo, self.lbl_detalle, self.lbl_siguiente)
        )


def alternar_estado_vacio(estado: EmptyState, widgets, vacio: bool) -> None:
    """Muestra el mensaje y oculta el contenido, o al revés.

    Solo se oculta lo que está **vacío**: quien llama decide, porque el
    componente no sabe si una tabla tiene filas.

    Args:
        estado: el `EmptyState` de la pestaña.
        widgets: los widgets que representa el contenido.
        vacio: si no hay nada que enseñar.
    """
    estado.mostrar(vacio)
    for widget in widgets:
        widget.setVisible(not vacio)


from PySide6.QtCore import Qt  # noqa: E402
