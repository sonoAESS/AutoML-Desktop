# ui/main_window.py
"""Ventana principal: barra lateral de pasos y panel de contenido."""

from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from core.state import AppState
from ui.data_tab import DataTab
from ui.layout import envolver_scroll
from ui.predict_tab import PredictTab
from ui.preprocess_tab import PreprocessTab
from ui.results_tab import ResultsTab
from ui.sidebar import INDICE_POR_NOMBRE, StepSidebar
from ui.theme import COLORES, ESPACIOS, NOMBRE_APP, icono_app
from ui.train_tab import TrainTab

#: Tamaño inicial y mínimo. 1366×768 es la resolución de portátil más común en
#: el entorno de uso; descontando la barra de tareas quedan unos 700 px útiles,
#: así que la ventana se pide más baja y el contenido se desplaza dentro.
TAMANO_INICIAL = (1240, 720)
TAMANO_MINIMO = (1000, 640)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(NOMBRE_APP)
        self.setWindowIcon(icono_app())
        self.resize(*TAMANO_INICIAL)
        self.setMinimumSize(*TAMANO_MINIMO)

        self.state = AppState()

        self.setCentralWidget(self._build_ui())
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Listo")

        self._conectar()
        self._atajos()

    # ------------------------------------------------------------------
    def _build_ui(self) -> QWidget:
        """Cabecera, barra lateral y panel de contenido."""
        contenedor = QWidget()
        vertical = QVBoxLayout(contenedor)
        vertical.setContentsMargins(0, 0, 0, 0)
        vertical.setSpacing(0)

        vertical.addWidget(self._build_cabecera())

        cuerpo = QHBoxLayout()
        cuerpo.setContentsMargins(0, 0, 0, 0)
        cuerpo.setSpacing(0)

        self.sidebar = StepSidebar()
        self.sidebar.paso_elicido.connect(self.ir_a_paso)
        cuerpo.addWidget(self.sidebar)

        # El contenido va en un `QStackedWidget`, no en pestañas: una barra de
        # pestañas consume alto que aquí hace falta para el contenido.
        self.pilas = QStackedWidget()
        self.data_tab = DataTab(self.state)
        self.preprocess_tab = PreprocessTab(self.state)
        self.train_tab = TrainTab(self.state)
        self.results_tab = ResultsTab(self.state)
        self.predict_tab = PredictTab(self.state)

        self.pestanas = {
            "datos": self.data_tab,
            "preprocesamiento": self.preprocess_tab,
            "entrenamiento": self.train_tab,
            "resultados": self.results_tab,
            "prediccion": self.predict_tab,
        }
        for nombre in (
            "datos",
            "preprocesamiento",
            "entrenamiento",
            "resultados",
            "prediccion",
        ):
            # Se envuelve el contenido, no la pestaña: la barra de
            # desplazamiento pertenece a la pestaña, no a la ventana.
            self.pilas.addWidget(envolver_scroll(self.pestanas[nombre]))

        cuerpo.addWidget(self.pilas, 1)
        vertical.addLayout(cuerpo, 1)
        return contenedor

    def _build_cabecera(self) -> QFrame:
        """Barra superior con el nombre del paso actual."""
        cabecera = QFrame()
        cabecera.setObjectName("cabecera")
        cabecera.setFixedHeight(52)

        layout = QHBoxLayout(cabecera)
        layout.setContentsMargins(
            ESPACIOS["lg"], ESPACIOS["sm"], ESPACIOS["lg"], ESPACIOS["sm"]
        )

        self.lbl_paso = QLabel("Paso 1 de 5 · Datos")
        self.lbl_paso.setObjectName("cabecera_titulo")
        layout.addWidget(self.lbl_paso)

        layout.addStretch()

        self.lbl_contexto = QLabel("")
        self.lbl_contexto.setProperty("role", "suave")
        layout.addWidget(self.lbl_contexto)

        return cabecera

    # ------------------------------------------------------------------
    def _conectar(self) -> None:
        """Cableado entre pestañas. Solo AppState + señales (AGENTS §5.1)."""
        self.data_tab.data_loaded.connect(self._on_data_loaded)
        self.data_tab.profile_changed.connect(self._on_profile_changed)
        self.train_tab.config_changed.connect(self.preprocess_tab.refresh)
        self.preprocess_tab.data_processed.connect(self._on_data_processed)
        self.train_tab.model_trained.connect(self._on_model_trained)
        self.results_tab.model_saved.connect(self._on_model_saved)
        # Las pestañas piden navegar; la ventana decide a dónde.
        self.results_tab.navigate_requested.connect(self.ir_a_paso_nombre)

    def _atajos(self) -> None:
        """Ctrl+1..5 para saltar de paso y Ctrl+Tab para avanzar."""
        for indice, nombre in enumerate(self.pestanas):
            atajo = QShortcut(QKeySequence(f"Ctrl+{indice + 1}"), self)
            atajo.activated.connect(lambda i=indice: self.ir_a_paso(i))

        siguiente = QShortcut(QKeySequence("Ctrl+Tab"), self)
        siguiente.activated.connect(self.paso_siguiente)

    # ------------------------------------------------------------------
    def ir_a_paso(self, indice: int) -> None:
        """Único punto de la aplicación que cambia de vista.

        Args:
            indice: posición del paso, de 0 a 4.
        """
        if not 0 <= indice < self.pilas.count():
            return
        self.pilas.setCurrentIndex(indice)
        self.sidebar.seleccionar(indice)
        self.lbl_paso.setText(f"Paso {indice + 1} de 5 · {self._nombre_paso(indice)}")

    def ir_a_paso_nombre(self, nombre: str) -> None:
        """Navega al paso indicado por su nombre."""
        indice = INDICE_POR_NOMBRE.get(nombre)
        if indice is None:
            return
        self.ir_a_paso(indice)

    def paso_siguiente(self) -> None:
        """Avanza al paso siguiente y vuelve al primero al final."""
        siguiente = (self.sidebar.paso_actual() + 1) % self.pilas.count()
        self.ir_a_paso(siguiente)

    def _nombre_paso(self, indice: int) -> str:
        from ui.sidebar import PASOS

        return PASOS[indice][1]

    def completar_paso(self, indice: int) -> None:
        """Registra que un paso se ha terminado."""
        self.sidebar.marcar_completado(indice)

    # ------------------------------------------------------------------
    def _estado(self, texto: str, ok: bool = True) -> None:
        """Escribe en la barra de estado, distinguiendo éxito de fallo.

        No se usa para errores: esos siguen yendo por `QMessageBox`
        (AGENTS §5.2), que el usuario no puede ignorar por accidente.
        """
        self.statusBar().showMessage(texto)
        color = COLORES["suave"] if ok else COLORES["acento"]
        self.statusBar().setStyleSheet(f"color: {color};")

    def _on_data_loaded(self):
        self.completar_paso(0)
        self.preprocess_tab.refresh()
        self.train_tab.refresh()
        self._estado("Datos cargados. Revisa el perfil y sigue al paso 2.")

    def _on_profile_changed(self):
        self.preprocess_tab.refresh()
        self.train_tab.refresh()
        self._estado("Tipos de datos actualizados.")

    def _on_data_processed(self):
        self.completar_paso(1)
        self.train_tab.refresh()
        self._estado("Dataset preprocesado. Ya puedes entrenar en el paso 3.")

    def _on_model_trained(self):
        self.completar_paso(2)
        self.completar_paso(3)
        self.results_tab.refresh()
        self._estado("Modelo entrenado. Mira las métricas en el paso 4.")

    def _on_model_saved(self, ruta):
        self._estado(f"Proyecto guardado en {ruta}")
