# ui/predict_tab.py
import importlib.util

from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core import data_loader, persistence
from ui import file_dialogs
from ui.empty_state import EmptyState, alternar_estado_vacio
from ui.enabled import con_pista
from ui.layout import acotar_tabla
from ui.theme import marcar

EXPORT_FILTER = "CSV (*.csv);;Excel (*.xlsx)"
EXPORT_PREFERENCE = (";", ",", "\t", "|")


def export_csv(df, ruta: str):
    """Escribe un CSV que Excel en español abre con un solo clic.

    El separador es el que aparezca en los nombres de las columnas; si ninguno
    aparece se usa `;`, que es el que espera Excel en configuración regional
    española (C-007). El `utf-8-sig` evita que las tildes se vean mal.
    """
    # Se unen con un espacio y no con ",": si se unieran con una coma, la
    # detección encontraría siempre esa coma y elegiría mal el separador.
    cabecera = " ".join(str(columna) for columna in df.columns)
    sep = data_loader.detect_separator(cabecera, preference=EXPORT_PREFERENCE)
    df.to_csv(ruta, index=False, sep=sep or ";", encoding="utf-8-sig")


def export_excel(df, ruta: str):
    """Escribe un XLSX; exige `openpyxl` instalado."""
    if importlib.util.find_spec("openpyxl") is None:
        raise RuntimeError(
            "Falta el paquete «openpyxl» para exportar a Excel. "
            "Instálalo con:  pip install openpyxl\n\n"
            "Mientras tanto, elige la opción CSV."
        )
    df.to_excel(ruta, index=False, engine="openpyxl")


class PredictTab(QWidget):
    """Carga un modelo ya construido y lo aplica a datos nuevos."""

    def __init__(self, state):
        super().__init__()
        self.state = state
        self._input_df = None
        self._preview = None
        self._predictions = None
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        self.btn_load_model = QPushButton("Cargar modelo (.automl)…")
        self.btn_load_model.clicked.connect(self.load_bundle)
        self.btn_load_data = QPushButton("Cargar datos de entrada…")
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
        acotar_tabla(self.table_entrada)
        self.table_entrada.setMaximumHeight(180)
        v.addWidget(self.table_entrada)
        layout.addWidget(gb_datos)

        self.vacio = EmptyState(
            "No hay ningún modelo cargado",
            "Para predecir hace falta un proyecto .automl con el modelo "
            "entrenado, sus metadatos y el preprocesamiento que usó.",
            "Carga un modelo en el paso 5, o entrena uno en el paso 3",
        )
        layout.addWidget(self.vacio, 1)

        self.btn_predict = QPushButton("Aplicar el modelo")
        self.btn_predict.setEnabled(False)
        self.btn_predict.clicked.connect(self.predict)
        layout.addWidget(self.btn_predict)

        self.btn_export = QPushButton("Exportar resultados")
        self.btn_export.setToolTip(
            "Guarda un CSV con los datos de entrada y las columnas de "
            "predicción, para poder revisar el resultado."
        )
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self.export_predictions)
        layout.addWidget(self.btn_export)

        self.btn_add_to_data = QPushButton("Añadir al dataset")
        marcar(self.btn_add_to_data, "peligro")
        self.btn_add_to_data.setToolTip(
            "Añade la predicción como una columna más del dataset cargado. "
            "Después ya no podrás deshacerlo sin volver a cargar el archivo."
        )
        self.btn_add_to_data.setEnabled(False)
        self.btn_add_to_data.clicked.connect(self.add_to_data)
        layout.addWidget(self.btn_add_to_data)

        layout.addWidget(QLabel("Predicciones:"))
        self.table_salida = QTableWidget()
        acotar_tabla(self.table_salida)
        layout.addWidget(self.table_salida)

        self.lbl_resultado = QLabel("")
        self.lbl_resultado.setWordWrap(True)
        layout.addWidget(self.lbl_resultado)

        # Sin modelo cargado no se puede predecir.
        self._update_enabled_state()
        self._alternar_vacio()

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
        self._clear_predictions()
        # El estado de los botones se recalcula entero aquí: mantener un
        # `setEnabled(True)` suelto duplicaba la regla de `_update_enabled_state`.
        self.refresh()

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
        path = file_dialogs.choose_table_file(self, "Seleccionar datos de entrada")
        if not path:
            return
        try:
            hojas = data_loader.list_sheets(path)
            hoja = file_dialogs.choose_sheet(self, hojas)
            if len(hojas) > 1 and hoja is None:
                return
            df = data_loader.load_table(path, sheet=hoja if hoja else 0)
        except Exception as e:
            QMessageBox.critical(self, "Error al cargar los datos", str(e))
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
        self.refresh()

    # ------------------------------------------------------------------
    def refresh(self):
        """Reevalúa qué se puede hacer con el modelo y los datos cargados."""
        self._update_enabled_state()
        self._alternar_vacio()

    def _alternar_vacio(self):
        """El mensaje cambia según falte el modelo o los datos."""
        if self.state.bundle is None:
            self.vacio.set_text(
                "No hay ningún modelo cargado",
                "Para predecir hace falta un proyecto .automl con el modelo "
                "entrenado, sus metadatos y el preprocesamiento que usó.",
                "Carga un modelo en el paso 5, o entrena uno en el paso 3",
            )
            vacio = True
        elif self._input_df is None:
            self.vacio.set_text(
                "Faltan los datos a predecir",
                "El modelo ya está cargado. Falta el archivo con las filas "
                "que quieres clasificar o de las que quieres obtener el valor.",
                "Carga un CSV en el paso 5",
            )
            vacio = True
        else:
            vacio = False
        alternar_estado_vacio(
            self.vacio, (self.table_entrada, self.table_salida), vacio
        )

    def _update_enabled_state(self):
        """Aplicar el modelo exige modelo **y** datos de entrada.

        Exportar exige predicciones ya calculadas. Antes los dos botones se
        pulsaban sin nada y el error salía como diálogo.
        """
        hay_modelo = self.state.bundle is not None
        hay_datos = self._input_df is not None
        hay_predicciones = self._predictions is not None

        con_pista(
            self.btn_predict,
            hay_modelo and hay_datos,
            (
                "Aplica el modelo cargado a los datos de entrada de arriba."
                if hay_modelo and hay_datos
                else (
                    "Carga primero un modelo guardado."
                    if not hay_modelo
                    else "Carga un CSV con los datos a predecir."
                )
            ),
        )
        con_pista(
            self.btn_export,
            hay_predicciones,
            (
                "Guarda los datos de entrada junto con la predicción, para "
                "revisar el resultado."
                if hay_predicciones
                else "Primero aplica el modelo para tener una predicción que exportar."
            ),
        )
        self.btn_add_to_data.setEnabled(hay_predicciones)
        self.table_salida.setEnabled(hay_predicciones)

    # ------------------------------------------------------------------
    def predict(self):
        if self.state.bundle is None or self._input_df is None:
            QMessageBox.warning(
                self,
                "Faltan datos",
                "Carga un modelo y un fichero de datos de entrada.",
            )
            return
        try:
            frame, report = self._aplicar(strict=True)
        except ValueError:
            if not self._confirmar_sin_columnas():
                return
            try:
                frame, report = self._aplicar(strict=False)
            except Exception as e:
                QMessageBox.critical(self, "Error al predecir", str(e))
                return
        except Exception as e:
            QMessageBox.critical(self, "Error al predecir", str(e))
            return

        columnas_nuevas = [
            c for c in frame.columns if c not in set(self._input_df.columns)
        ]
        self._preview = frame[columnas_nuevas]
        self._predictions = frame
        self._fill_table(self.table_salida, self._preview, max_rows=500)
        self.refresh()
        self.lbl_resultado.setText(
            f"{len(frame)} filas predichas · {len(self._input_df.columns)} columnas "
            f"de entrada + {len(columnas_nuevas)} de predicción "
            f"({', '.join(columnas_nuevas)}), todas exportables juntas.\n"
            f"{report.describe()}"
        )

    def _aplicar(self, strict=True):
        """Devuelve `(frame entrada+predicción, report)`."""
        return persistence.predict_frame(
            self.state.bundle, self._input_df, strict=strict
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

    def add_to_data(self):
        """Añade la predicción como columna del dataset cargado.

        Es la acción que más puede costar deshacer, así que pide confirmación
        antes de escribir.
        """
        if self._predictions is None:
            return
        respuesta = QMessageBox.question(
            self,
            "Añadir al dataset",
            "Se añadirá la predicción como una columna del dataset cargado. "
            "Es un cambio que no se puede deshacer.\n\n¿Continuar?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if respuesta != QMessageBox.Yes:
            return

        entrada = self._input_df.copy()
        for columna in self._predictions.columns:
            if columna in (self.state.target_column,):
                continue
            entrada[columna] = self._predictions[columna].values
        self._input_df = entrada
        self._fill_table(self.table_entrada, entrada, max_rows=100)
        self.refresh()
        self.lbl_resultado.setText(
            "Predicción añadida al dataset. Ya puedes exportarlo con "
            "«Exportar resultados»."
        )

    # ------------------------------------------------------------------
    def export_predictions(self):
        """Guarda el CSV de entrada junto con las columnas de predicción."""
        if self._predictions is None:
            return
        destino = file_dialogs.choose_save_path(
            self,
            "Exportar resultados (datos + predicción)",
            "predicciones.csv",
            EXPORT_FILTER,
        )
        if not destino:
            return
        try:
            if destino.lower().endswith(".xlsx"):
                export_excel(self._predictions, destino)
            else:
                export_csv(self._predictions, destino)
        except Exception as e:
            QMessageBox.critical(self, "Error al exportar", str(e))
            return
        self.lbl_resultado.setText(
            f"Datos y predicciones exportados en {destino}\n"
            f"({len(self._predictions)} filas · "
            f"{len(self._predictions.columns)} columnas)"
        )

    # ------------------------------------------------------------------
    def _clear_predictions(self):
        self._preview = None
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
