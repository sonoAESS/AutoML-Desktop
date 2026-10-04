# ui/preprocess_tab.py
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QCheckBox, QComboBox, QPushButton,
    QLabel, QMessageBox, QListWidget, QListWidgetItem, QHBoxLayout,
    QTableWidget, QTableWidgetItem, QHeaderView,
)
from core import preprocessor, profiling

METHOD_LABELS = {
    "none": "sin normalizar",
    "standard": "estándar (z-score)",
    "minmax": "mín-máx (0-1)",
    "robust": "robusta (mediana-IQR)",
    "log": "logaritmo",
}


class PreprocessTab(QWidget):
    data_processed = Signal()

    def __init__(self, state):
        super().__init__()
        self.state = state
        self._norm_widgets = {}
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)

        gb_dups = QGroupBox("Duplicados")
        v = QVBoxLayout(gb_dups)
        self.chk_dups = QCheckBox("Eliminar filas duplicadas")
        self.chk_dups.setChecked(True)
        v.addWidget(self.chk_dups)
        layout.addWidget(gb_dups)

        gb_null = QGroupBox("Valores nulos")
        v = QVBoxLayout(gb_null)
        self.chk_high_missing = QCheckBox("Eliminar columnas con >50% nulos")
        self.chk_high_missing.setChecked(True)
        self.cmb_num = QComboBox()
        self.cmb_num.addItems(["median", "mean", "drop"])
        self.cmb_cat = QComboBox()
        self.cmb_cat.addItems(["mode", "constant", "drop"])
        v.addWidget(self.chk_high_missing)
        v.addWidget(QLabel("Estrategia numérica:"))
        v.addWidget(self.cmb_num)
        v.addWidget(QLabel("Estrategia categórica:"))
        v.addWidget(self.cmb_cat)
        layout.addWidget(gb_null)

        gb_tipos = QGroupBox("Tipos de datos y fechas")
        v = QVBoxLayout(gb_tipos)
        self.lbl_tipos = QLabel("")
        self.lbl_tipos.setWordWrap(True)
        v.addWidget(self.lbl_tipos)
        self.chk_dates = QCheckBox("Convertir las fechas en año, mes y día")
        v.addWidget(self.chk_dates)
        layout.addWidget(gb_tipos)

        gb_norm = QGroupBox("Normalización de variables numéricas")
        v = QVBoxLayout(gb_norm)
        self.table_norm = QTableWidget()
        self.table_norm.setColumnCount(3)
        self.table_norm.setHorizontalHeaderLabels(
            ["Columna", "Rango actual", "Normalización"]
        )
        self.table_norm.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self.table_norm.verticalHeader().setVisible(False)
        v.addWidget(self.table_norm)
        layout.addWidget(gb_norm)

        gb_cols = QGroupBox("Columnas a eliminar")
        v = QVBoxLayout(gb_cols)
        self.list_cols = QListWidget()
        self.list_cols.setSelectionMode(QListWidget.MultiSelection)
        v.addWidget(self.list_cols)
        layout.addWidget(gb_cols)

        self.btn_apply = QPushButton("Aplicar preprocesamiento")
        self.btn_apply.clicked.connect(self.apply)
        layout.addWidget(self.btn_apply)

        self.lbl_result = QLabel("")
        self.lbl_result.setWordWrap(True)
        layout.addWidget(self.lbl_result)

    # ------------------------------------------------------------------
    def refresh(self):
        self._fill_columns()
        self._show_casts()
        self._fill_normalizations()

    def _fill_columns(self):
        self.list_cols.clear()
        if self.state.raw_df is None:
            return
        for col in self.state.raw_df.columns:
            self.list_cols.addItem(QListWidgetItem(str(col)))

    def _show_casts(self):
        casts = self.state.column_types
        if not casts:
            self.lbl_tipos.setText(
                "No hay tipos forzados: se usan los tipos detectados "
                "automáticamente en la pestaña 1."
            )
            return
        textos = ", ".join(
            f"{columna} → {profiling.SEMANTIC_LABELS.get(tipo, tipo)}"
            for columna, tipo in casts.items()
        )
        self.lbl_tipos.setText(f"Tipos forzados: {textos}")

    def _fill_normalizations(self):
        self.table_norm.blockSignals(True)
        self.table_norm.setRowCount(0)
        self._norm_widgets.clear()

        if self.state.raw_df is None:
            self.table_norm.blockSignals(False)
            return

        numericas = [
            profile
            for profile in profiling.profile_dataframe(self.state.raw_df)
            if profile.semantic_type == "numerico"
        ]
        sugeridas = preprocessor.suggested_normalizations(
            self.state.raw_df, numericas
        )

        for fila, profile in enumerate(numericas):
            self.table_norm.insertRow(fila)
            self.table_norm.setItem(fila, 0, QTableWidgetItem(profile.name))
            self.table_norm.setItem(
                fila, 1, QTableWidgetItem(profile.value_range)
            )

            combo = QComboBox()
            for metodo in preprocessor.NORMALIZATION_METHODS:
                combo.addItem(METHOD_LABELS[metodo], metodo)
            elegido = self.state.normalizations.get(
                profile.name, sugeridas.get(profile.name, "none")
            )
            combo.setCurrentIndex(max(combo.findData(elegido), 0))
            combo.currentIndexChanged.connect(self._on_norm_changed)
            self.table_norm.setCellWidget(fila, 2, combo)
            self._norm_widgets[profile.name] = combo

        self.table_norm.blockSignals(False)

    def _on_norm_changed(self):
        for nombre, combo in self._norm_widgets.items():
            metodo = combo.currentData()
            if metodo == "none":
                self.state.normalizations.pop(nombre, None)
            else:
                self.state.normalizations[nombre] = metodo

    # ------------------------------------------------------------------
    def apply(self):
        if self.state.raw_df is None:
            QMessageBox.warning(self, "Sin datos", "Carga un CSV primero.")
            return
        df = self.state.raw_df.copy()
        try:
            casts = dict(self.state.column_types)
            fecha_cols = [c for c, t in casts.items() if t == "fecha"]
            if self.chk_dates.isChecked():
                df = preprocessor.date_features(df, fecha_cols)
                casts = {c: t for c, t in casts.items() if t != "fecha"}
            if casts:
                df = preprocessor.cast_columns(df, casts)

            if self.chk_high_missing.isChecked():
                df = preprocessor.drop_high_missing(df, 0.5)
            if self.chk_dups.isChecked():
                df = preprocessor.drop_duplicates(df)

            df = preprocessor.fill_missing(
                df,
                numeric_strategy=self.cmb_num.currentText(),
                categorical_strategy=self.cmb_cat.currentText(),
            )

            cols_to_drop = [i.text() for i in self.list_cols.selectedItems()]
            if cols_to_drop:
                df = preprocessor.drop_columns(df, cols_to_drop)

            df = preprocessor.apply_transformations(
                df, casts=None, normalizations=self.state.normalizations or None
            )

            self.state.clean_df = df
            self.state.export_df = df.copy()
            self.state.profiles = profiling.profile_dataframe(df)
            self.state.reset_model()

            self.lbl_result.setText(
                f"Listo: {df.shape[0]} filas × {df.shape[1]} columnas "
                "(este es el dataset que se usará para entrenar y se guardará "
                "junto al modelo)."
            )
            self._fill_normalizations()
            self.data_processed.emit()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
