# ui/main_window.py
from PySide6.QtWidgets import QMainWindow, QStatusBar, QTabWidget

from core.state import AppState
from ui.data_tab import DataTab
from ui.predict_tab import PredictTab
from ui.preprocess_tab import PreprocessTab
from ui.results_tab import ResultsTab
from ui.theme import NOMBRE_APP, icono_app
from ui.train_tab import TrainTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(NOMBRE_APP)
        self.setWindowIcon(icono_app())
        self.resize(1180, 820)

        self.state = AppState()

        tabs = QTabWidget()
        self.data_tab = DataTab(self.state)
        self.preprocess_tab = PreprocessTab(self.state)
        self.train_tab = TrainTab(self.state)
        self.results_tab = ResultsTab(self.state)
        self.predict_tab = PredictTab(self.state)

        tabs.addTab(self.data_tab, "1. Datos")
        tabs.addTab(self.preprocess_tab, "2. Preprocesamiento")
        tabs.addTab(self.train_tab, "3. Entrenamiento")
        tabs.addTab(self.results_tab, "4. Resultados")
        tabs.addTab(self.predict_tab, "5. Predicción")
        self.setCentralWidget(tabs)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Listo")

        # Cuando se cargan datos, refrescamos las demás pestañas
        self.data_tab.data_loaded.connect(self.preprocess_tab.refresh)
        self.data_tab.data_loaded.connect(self.train_tab.refresh)
        self.data_tab.profile_changed.connect(self._on_profile_changed)
        self.train_tab.config_changed.connect(self.preprocess_tab.refresh)
        self.preprocess_tab.data_processed.connect(self._on_data_processed)
        self.train_tab.model_trained.connect(self.results_tab.refresh)
        self.results_tab.model_saved.connect(self._on_model_saved)

    # ------------------------------------------------------------------
    def _on_profile_changed(self):
        self.preprocess_tab.refresh()
        self.train_tab.refresh()
        self.statusBar().showMessage("Tipos de datos actualizados")

    def _on_data_processed(self):
        self.train_tab.refresh()
        self.statusBar().showMessage("Dataset preprocesado listo para entrenar")

    def _on_model_saved(self, ruta):
        self.statusBar().showMessage(f"Modelo guardado en {ruta}")
