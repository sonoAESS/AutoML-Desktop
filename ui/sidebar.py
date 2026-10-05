# ui/sidebar.py
"""Barra lateral de pasos: la navegación principal de la aplicación.

Sustituye a las pestañas superiores. Cada paso indica en qué está el usuario
y si los anteriores ya están hechos, algo que las pestañas planas no
expresaban.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)

from ui.theme import ESPACIOS, NOMBRE_APP, ORGANIZACION, escudo_pixmap

#: Los cinco pasos, en el orden en que se hacen. El nombre es el que usan las
#: señales de navegación (`navigate_requested`), el título el que ve el usuario.
PASOS: tuple[tuple[str, str], ...] = (
    ("datos", "Datos"),
    ("preprocesamiento", "Preprocesamiento"),
    ("entrenamiento", "Entrenamiento"),
    ("resultados", "Resultados"),
    ("prediccion", "Predicción"),
)

INDICE_POR_NOMBRE: dict[str, int] = {
    nombre: i for i, (nombre, _titulo) in enumerate(PASOS)
}

#: Los tres estados válidos de un paso.
ESTADOS = ("pendiente", "activo", "completado")

#: Ancho fijo de la columna. Estrechar los títulos de los pasos es lo que
#: provoca el salto de línea, así que se reserva el ancho en vez de dejarlo
#: al tamaño del texto.
ANCHO = 210


class StepButton(QPushButton):
    """Un paso de la barra lateral, con su número y su estado.

    El estado cambia el aspecto **y** el texto: un paso completado se marca con
    «✓» al principio. distinguishing el paso activo solo por el color dejaría
    al usuario sin saber dónde está si el monitor va en escala de grises.
    """

    def __init__(self, indice: int, titulo: str, parent=None):
        super().__init__(parent)
        self.indice = indice
        self.titulo = titulo
        self.setObjectName("paso")
        self.setCheckable(True)
        self.setProperty("estado", "pendiente")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(34)
        self.setToolTip(f"Paso {indice + 1} de {len(PASOS)}: {titulo}")
        self._pintar_texto()

    def _pintar_texto(self) -> None:
        """Refleja el estado en el texto del botón."""
        marca = "✓ " if self.property("estado") == "completado" else ""
        self.setText(f"{marca}{self.indice + 1}.  {self.titulo}")

    def set_estado(self, estado: str) -> None:
        """Fija el estado del botón y lo repinta."""
        self.setProperty("estado", estado)
        estilo = self.style()
        estilo.unpolish(self)
        estilo.polish(self)
        self._pintar_texto()


class StepSidebar(QFrame):
    """Columna fija con la marca y los cinco pasos."""

    paso_elicido = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("lateral")
        self.setFixedWidth(ANCHO)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, ESPACIOS["lg"], 0, ESPACIOS["lg"])
        layout.setSpacing(ESPACIOS["xs"])

        # --- marca institucional ---
        marca = QLabel()
        pixmap = escudo_pixmap()
        if not pixmap.isNull():
            marca.setPixmap(pixmap)
        marca.setFixedHeight(72)
        marca.setAlignment(
            Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
        )
        layout.addWidget(marca)

        titulo = QLabel(NOMBRE_APP)
        titulo.setObjectName("marca_titulo")
        titulo.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(titulo)

        layout.addSpacing(ESPACIOS["lg"])

        # --- pasos ---
        self.buttons: list[StepButton] = []
        for indice, (_nombre, titulo_paso) in enumerate(PASOS):
            boton = StepButton(indice, titulo_paso)
            boton.clicked.connect(lambda _checked, i=indice: self.paso_elicido.emit(i))
            self.buttons.append(boton)
            layout.addWidget(boton)

        layout.addStretch()

        self.pie = QLabel(ORGANIZACION)
        self.pie.setObjectName("marca_organizacion")
        self.pie.setWordWrap(True)
        self.pie.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(self.pie)

        self.completados: set[int] = set()
        self._actual = 0
        self.seleccionar(0)

    # ------------------------------------------------------------------
    def set_estado(self, indice: int, estado: str) -> None:
        """Fija el estado visual de **un** paso.

        Args:
            indice: posición del paso, de 0 a 4.
            estado: `pendiente`, `activo` o `completado`.

        Raises:
            ValueError: si el estado no es uno de los tres. Un estado
                inventado dejaría el paso con un aspecto que no existe en la
                hoja de estilo, sin avisar.
        """
        if estado not in ESTADOS:
            raise ValueError(
                f"Estado de paso desconocido: {estado!r}. "
                f"Use uno de {', '.join(ESTADOS)}."
            )
        if not 0 <= indice < len(self.buttons):
            raise IndexError(f"No existe el paso {indice}")
        self.buttons[indice].set_estado(estado)

    def seleccionar(self, indice: int) -> None:
        """Marca un paso como el actual y recalcula el estado de todos.

        Solo hay un paso activo a la vez. Los anteriores quedan completados si
        se ha avanzado más allá de ellos.
        """
        if not 0 <= indice < len(self.buttons):
            raise IndexError(f"No existe el paso {indice}")
        self._actual = indice
        for boton in self.buttons:
            boton.setChecked(boton.indice == indice)
            if boton.indice == indice:
                estado = "activo"
            elif boton.indice < indice or boton.indice in self.completados:
                estado = "completado"
            else:
                estado = "pendiente"
            boton.set_estado(estado)

    def marcar_completado(self, indice: int) -> None:
        """Registra un paso como completado.

        El paso actual no se marca: se está haciendo ahora, no se ha
        terminado.
        """
        if not 0 <= indice < len(self.buttons):
            raise IndexError(f"No existe el paso {indice}")
        self.completados.add(indice)

    def paso_actual(self) -> int:
        """Índice del paso activo."""
        return self._actual
