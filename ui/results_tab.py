# ui/results_tab.py
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox,
    QHBoxLayout, QCheckBox,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from sklearn.metrics import ConfusionMatrixDisplay

from core import persistence


class ResultsTab(QWidget):
    model_saved = Signal(str)

    def __init__(self, state):
        super().__init__()
        self.state = state
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        self.lbl = QLabel("Sin resultados todavía.")
        self.lbl.setWordWrap(True)
        layout.addWidget(self.lbl)

        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas)

        opciones = QHBoxLayout()
        self.chk_dataset = QCheckBox("Incluir el dataset transformado")
        self.chk_dataset.setChecked(True)
        self.chk_dataset.setToolTip(
            "Guarda también el dataset ya limpiado y normalizado, en el mismo "
            "archivo que el modelo."
        )
        opciones.addWidget(self.chk_dataset)
        opciones.addStretch()
        layout.addLayout(opciones)

        botones = QHBoxLayout()
        self.btn_save = QPushButton("Guardar modelo y dataset (.automl)")
        self.btn_save.clicked.connect(self.save_bundle)
        self.btn_export = QPushButton("Exportar solo el dataset (CSV)")
        self.btn_export.clicked.connect(self.export_dataset)
        self.btn_load_in_predict = QPushButton("Ir a Predicción")
        self.btn_load_in_predict.clicked.connect(self._go_to_predict)
        botones.addWidget(self.btn_save)
        botones.addWidget(self.btn_export)
        botones.addWidget(self.btn_load_in_predict)
        layout.addLayout(botones)

        self.lbl_saved = QLabel("")
        self.lbl_saved.setWordWrap(True)
        layout.addWidget(self.lbl_saved)

    # ------------------------------------------------------------------
    def refresh(self):
        metrics = self.state.metrics
        if not metrics:
            return
        self.lbl.setText(
            "Métricas:\n" + "\n".join(
                f"  {k}: {v:.4f}" if isinstance(v, (int, float)) else f"  {k}: {v}"
                for k, v in metrics.items()
            )
        )
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        if not self._plot_cv_results(ax):
            self._plot_test(ax)
        self.canvas.draw()

    def _plot_test(self, ax):
        if not self.state._test_data:
            return
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

    def _plot_cv_results(self, ax):
        cv = getattr(self.state, "_cv_results", None)
        if not cv:
            return False
        scores = cv.get("mean_test_score")
        if scores is None:
            return False
        ax.plot(range(1, len(scores) + 1), scores, marker="o")
        ax.set_xlabel("Combinación")
        ax.set_ylabel("Puntuación de la validación cruzada")
        ax.set_title("Resultados de la búsqueda de hiperparámetros")
        return True

    # ------------------------------------------------------------------
    def save_bundle(self):
        if self.state.trained_model is None:
            QMessageBox.warning(self, "Sin modelo", "Entrena un modelo primero.")
            return
        if not self.state.target_column or not self.state.model_name:
            QMessageBox.warning(
                self, "Sin modelo",
                "No se sabe qué variable objetivo ni qué modelo se usó.",
            )
            return

        destino, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar modelo y dataset",
            self._default_name(),
            "Modelo AutoML (*.automl)",
        )
        if not destino:
            return

        try:
            metadata = persistence.build_metadata(
                self.state.trained_model,
                target_column=self.state.target_column,
                task_type=self.state.task_type,
                model_name=self.state.model_name,
                metrics=self.state.metrics,
                df=self.state.export_df,
                casts=self.state.column_types or None,
                normalizations=self.state.normalizations or None,
            )
            dataset = self.state.export_df if self.chk_dataset.isChecked() else None
            ruta = persistence.save_bundle(
                destino, self.state.trained_model, metadata, dataset=dataset
            )
        except Exception as e:
            QMessageBox.critical(self, "Error al guardar", str(e))
            return

        self.state.bundle_path = ruta
        self.lbl_saved.setText(
            f"Guardado en {ruta}\n"
            "Puedes abrirlo en la pestaña 5. Predicción para aplicarlo a "
            "nuevos datos."
        )
        self.model_saved.emit(ruta)

    def export_dataset(self):
        if self.state.export_df is None:
            QMessageBox.warning(
                self, "Sin dataset", "Aplica el preprocesamiento primero."
            )
            return
        destino, _ = QFileDialog.getSaveFileName(
            self, "Exportar dataset transformado", "dataset_preprocesado.csv",
            "CSV (*.csv)",
        )
        if not destino:
            return
        try:
            self.state.export_df.to_csv(destino, index=False)
        except Exception as e:
            QMessageBox.critical(self, "Error al exportar", str(e))
            return
        self.lbl_saved.setText(f"Dataset exportado en {destino}")

    def _go_to_predict(self):
        window = self.window()
        tabs = window.centralWidget()
        if tabs is not None and hasattr(tabs, "setCurrentIndex"):
            indice = tabs.indexOf(window.predict_tab)
            if indice >= 0:
                tabs.setCurrentIndex(indice)

    def _default_name(self):
        base = self.state.source_path.rsplit("/", 1)[-1] if self.state.source_path else ""
        nombre = (self.state.model_name or "modelo").replace(" ", "_")
        if base:
            return f"{base.rsplit('.', 1)[0]}_{nombre}{persistence.BUNDLE_SUFFIX}"
        return f"{nombre}{persistence.BUNDLE_SUFFIX}"