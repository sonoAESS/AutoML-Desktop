# ui/main_window.py
from PySide6.QtWidgets import QMainWindow, QTabWidget, QStatusBar
from core.state import AppState
from ui.data_tab import DataTab
from ui.preprocess_tab import PreprocessTab
from ui.train_tab import TrainTab
from ui.results_tab import ResultsTab

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AutoML Desktop")
        self.resize(1100, 750)

        self.state = AppState()

        tabs = QTabWidget()
        self.data_tab = DataTab(self.state)
        self.preprocess_tab = PreprocessTab(self.state)
        self.train_tab = TrainTab(self.state)
        self.results_tab = ResultsTab(self.state)

        tabs.addTab(self.data_tab, "1. Datos")
        tabs.addTab(self.preprocess_tab, "2. Preprocesamiento")
        tabs.addTab(self.train_tab, "3. Entrenamiento")
        tabs.addTab(self.results_tab, "4. Resultados")
        self.setCentralWidget(tabs)

        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Listo")

        # Cuando se cargan datos, refrescamos las demás pestañas
        self.data_tab.data_loaded.connect(self.preprocess_tab.refresh)
        self.data_tab.data_loaded.connect(self.train_tab.refresh)
        self.preprocess_tab.data_processed.connect(self.train_tab.refresh)
        self.train_tab.model_trained.connect(self.results_tab.refresh)