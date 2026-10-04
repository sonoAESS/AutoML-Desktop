# ui/predict_tab.py
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QMessageBox, QTableWidget, QTableWidgetItem, QGroupBox,
)
from core import data_loader, persistence


class PredictTab(QWidget):
    """Carga un modelo ya construido y lo aplica a datos nuevos."""

    def __init__(self, state):
        super().__init__()
        self.state = state
        self._input_df = None
        self._predictions = None
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        self.btn_load_model = QPushButton("Cargar modelo (.automl)…")
        self.btn_load_model.clicked.connect(self.load_bundle)
        self.btn_load_data = QPushButton("Cargar datos de entrada (CSV)…")
        self.btn_load_data.clicked.connect(self.load_data)
        self.btn_load_data.setEnabled(False)
        top.addWidget(self.btn_load_model)
        top.addWidget(self.btn_load_data)
        top.addStretch()
        layout.addLayout(top)

        gb_modelo = QGroupBox("Modelo cargado")
        v = QVBoxLayout(gb_modelo)
        self.lbl_modelo = QLabel("Ningún modelo cargado.")
        self.lbl_modelo.setWordWrap(True)
        v.addWidget(self.lbl_modelo)
        layout.addWidget(gb_modelo)

        gb_datos = QGroupBox("Datos de entrada")
        v = QVBoxLayout(gb_datos)
        self.lbl_datos = QLabel("Ningún archivo cargado.")
        self.lbl_datos.setWordWrap(True)
        v.addWidget(self.lbl_datos)
        self.table_entrada = QTableWidget()
        self.table_entrada.setMaximumHeight(180)
        v.addWidget(self.table_entrada)
        layout.addWidget(gb_datos)

        self.btn_predict = QPushButton("Aplicar el modelo")
        self.btn_predict.setEnabled(False)
        self.btn_predict.clicked.connect(self.predict)
        layout.addWidget(self.btn_predict)

        self.btn_export = QPushButton("Exportar predicciones (CSV)")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_predictions)
        layout.addWidget(self.btn_export)

        layout.addWidget(QLabel("Predicciones:"))
        self.table_salida = QTableWidget()
        layout.addWidget(self.table_salida)

        self.lbl_resultado = QLabel("")
        self.lbl_resultado.setWordWrap(True)
        layout.addWidget(self.lbl_resultado)

    # ------------------------------------------------------------------
    def load_bundle(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar modelo", "", "Modelo AutoML (*.automl)"
        )
        if not path:
            return
        confirmacion = QMessageBox.question(
            self,
            "Carga de modelo",
            "Los archivos .automl contienen código compilado: solo continúa "
            "si el modelo es de confianza.\n\n¿Quieres cargar este archivo?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirmacion != QMessageBox.Yes:
            return
        try:
            bundle = persistence.load_bundle(path)
        except Exception as e:
            QMessageBox.critical(self, "Error al cargar el modelo", str(e))
            return

        self.state.bundle = bundle
        self.state.bundle_path = path
        self.lbl_modelo.setText(bundle.metadata.describe())
        self.lbl_modelo.setToolTip(
            "Columnas requeridas: " + ", ".join(bundle.metadata.feature_columns)
        )
        self.btn_load_data.setEnabled(True)
        self.btn_predict.setEnabled(True)
        self._clear_predictions()

        if bundle.dataset is not None:
            self.lbl_modelo.setText(
                self.lbl_modelo.text()
                + f"\nEl bundle incluye {bundle.dataset.shape[0]} filas de "
                "entrenamiento ya transformadas."
            )

    def load_data(self):
        if self.state.bundle is None:
            QMessageBox.warning(
                self, "Faltan datos", "Primero carga un modelo guardado."
            )
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar CSV de entrada", "", "CSV (*.csv *.txt *.tsv)"
        )
        if not path:
            return
        try:
            df = data_loader.load_csv(path)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return

        self._input_df = df
        metadata = self.state.bundle.metadata
        report = persistence.check_schema(df, metadata)
        self.lbl_datos.setText(
            f"{path}\n{df.shape[0]} filas × {df.shape[1]} columnas\n"
            f"{report.describe()}"
        )
        if not report.ok:
            self.lbl_datos.setStyleSheet("color: #b00020;")
        else:
            self.lbl_datos.setStyleSheet("")
        self._fill_table(self.table_entrada, df, max_rows=100)
        self._clear_predictions()

    # ------------------------------------------------------------------
    def predict(self):
        if self.state.bundle is None or self._input_df is None:
            QMessageBox.warning(
                self, "Faltan datos", "Carga un modelo y un CSV de entrada."
            )
            return
        try:
            salida, report, _, _ = persistence.predict(
                self.state.bundle, self._input_df
            )
        except ValueError:
            if not self._confirmar_sin_columnas():
                return
            try:
                salida, report, _, _ = persistence.predict(
                    self.state.bundle, self._input_df, strict=False
                )
            except Exception as e:
                QMessageBox.critical(self, "Error al predecir", str(e))
                return
        except Exception as e:
            QMessageBox.critical(self, "Error al predecir", str(e))
            return

        self._predictions = salida
        self._fill_table(self.table_salida, salida, max_rows=500)
        self.btn_export.setEnabled(True)
        self.lbl_resultado.setText(
            f"{len(salida)} filas predichas.\n{report.describe()}"
        )

    def _confirmar_sin_columnas(self):
        """Pregunta si quiere continuar aunque falten columnas obligatorias."""
        metadata = self.state.bundle.metadata
        report = persistence.check_schema(self._input_df, metadata)
        faltan = ", ".join(report.missing)
        respuesta = QMessageBox.question(
            self,
            "Faltan columnas",
            f"Faltan estas columnas que el modelo necesita: {faltan}.\n\n"
            "Se rellenarán con valores vacíos, que el modelo ya sabe "
            "imputar. ¿Continuar?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        return respuesta == QMessageBox.Yes

    def export_predictions(self):
        if self._predictions is None:
            return
        destino, _ = QFileDialog.getSaveFileName(
            self, "Exportar predicciones", "predicciones.csv", "CSV (*.csv)"
        )
        if not destino:
            return
        try:
            self._predictions.to_csv(destino, index=False)
        except Exception as e:
            QMessageBox.critical(self, "Error al exportar", str(e))
            return
        self.lbl_resultado.setText(f"Predicciones exportadas en {destino}")

    # ------------------------------------------------------------------
    def _clear_predictions(self):
        self._predictions = None
        self.table_salida.setRowCount(0)
        self.btn_export.setEnabled(False)
        self.lbl_resultado.setText("")

    def _fill_table(self, table, df, max_rows=200):
        preview = df.head(max_rows)
        table.setRowCount(len(preview))
        table.setColumnCount(len(preview.columns))
        table.setHorizontalHeaderLabels([str(c) for c in preview.columns])
        for i, (_, row) in enumerate(preview.iterrows()):
            for j, value in enumerate(row):
                texto = f"{value:.4f}" if isinstance(value, float) else str(value)
                table.setItem(i, j, QTableWidgetItem(texto))
        table.resizeColumnsToContents()