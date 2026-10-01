# ui/results_tab.py
import joblib
import numpy as np
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog,
    QMessageBox, QTextEdit,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay

class ResultsTab(QWidget):
    def __init__(self, state):
        super().__init__()
        self.state = state
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        self.lbl = QLabel("Sin resultados todavía.")
        layout.addWidget(self.lbl)

        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas)

        self.btn_save = QPushButton("Guardar modelo (.joblib)")
        self.btn_save.clicked.connect(self.save_model)
        layout.addWidget(self.btn_save)

    def refresh(self):
        metrics = self.state.metrics
        if not metrics:
            return
        self.lbl.setText(
            "Métricas:\n" + "\n".join(f"  {k}: {v:.4f}" for k, v in metrics.items())
        )

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        X_test, y_test, y_pred = self.state._test_data

        if self.state.task_type == "classification":
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
            ax.set_title("Matriz de confusión")
        else:
            ax.scatter(y_test, y_pred, alpha=0.5)
            lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
            ax.plot(lims, lims, "r--")
            ax.set_xlabel("Real")
            ax.set_ylabel("Predicho")
            ax.set_title("Real vs. Predicho")
        self.canvas.draw()

    def save_model(self):
        if self.state.trained_model is None:
            QMessageBox.warning(self, "Sin modelo", "Entrena un modelo primero.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Guardar modelo", "modelo.joblib", "Joblib (*.joblib)"
        )
        if not path:
            return
        joblib.dump(self.state.trained_model, path)
        QMessageBox.information(self, "Guardado", f"Modelo guardado en:\n{path}")