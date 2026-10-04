# ui/results_tab.py
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from sklearn.metrics import ConfusionMatrixDisplay

from core import model_specs, persistence


def _format_metric(valor) -> str:
    """Valor de una métrica para la tabla: número, texto o `n/d`."""
    if valor is None:
        return "n/d"
    if isinstance(valor, bool):
        return "sí" if valor else "no"
    if isinstance(valor, (int, float)):
        return f"{valor:.4f}"
    return str(valor)


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

        self.tbl_metricas = QTableWidget(0, 2)
        self.tbl_metricas.setHorizontalHeaderLabels(["Métrica", "Valor"])
        self.tbl_metricas.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        self.tbl_metricas.verticalHeader().setVisible(False)
        self.tbl_metricas.setMaximumHeight(140)
        layout.addWidget(self.tbl_metricas)

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
        self._show_metrics(metrics)
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        if not self._plot_cv_results(ax):
            self._plot_test(ax)
        self.canvas.draw()

    def _show_metrics(self, metrics):
        """Tabla Métrica/Valor con `n/d` y motivo cuando algo no existe."""
        claves = model_specs.METRIC_KEYS.get(self.state.task_type, ())
        self.tbl_metricas.setRowCount(len(claves))
        motivo = metrics.get("auc_motivo")

        for fila, clave in enumerate(claves):
            etiqueta = model_specs.metric_label(clave)
            item_clave = QTableWidgetItem(etiqueta)
            item_valor = QTableWidgetItem(_format_metric(metrics.get(clave)))
            if metrics.get(clave) is None:
                item_valor.setForeground(QBrush(QColor(Qt.GlobalColor.gray)))
                if motivo:
                    item_valor.setToolTip(motivo)
            self.tbl_metricas.setItem(fila, 0, item_clave)
            self.tbl_metricas.setItem(fila, 1, item_valor)

        resumen = [
            f"Modelo: {self.state.model_name}",
            (
                f"Evaluado sobre {metrics['n_test']} filas de prueba"
                if "n_test" in metrics
                else ""
            ),
            persistence.balancing_summary(
                self.state.pipeline, {"method": self.state.balancing}
            ),
        ]
        if metrics.get("auc_disponible") is False:
            resumen.append(f"Aviso: {motivo}")
        if isinstance(metrics.get("best_params"), dict):
            resumen.append(f"mejores parámetros: {metrics['best_params']}")
        self.lbl.setText("\n".join(texto for texto in resumen if texto))

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
                self,
                "Sin modelo",
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
            self,
            "Exportar dataset transformado",
            "dataset_preprocesado.csv",
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
        base = (
            self.state.source_path.rsplit("/", 1)[-1] if self.state.source_path else ""
        )
        nombre = (self.state.model_name or "modelo").replace(" ", "_")
        if base:
            return f"{base.rsplit('.', 1)[0]}_{nombre}{persistence.BUNDLE_SUFFIX}"
        return f"{nombre}{persistence.BUNDLE_SUFFIX}"
