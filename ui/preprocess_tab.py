# ui/preprocess_tab.py
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QCheckBox, QComboBox,
    QPushButton, QLabel, QMessageBox, QListWidget, QListWidgetItem,
    QHBoxLayout,
)
from core import preprocessor

class PreprocessTab(QWidget):
    data_processed = Signal()

    def __init__(self, state):
        super().__init__()
        self.state = state
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # --- Duplicados
        gb_dup = QGroupBox("Duplicados")
        v = QVBoxLayout(gb_dup)
        self.chk_dups = QCheckBox("Eliminar filas duplicadas")
        v.addWidget(self.chk_dups)
        layout.addWidget(gb_dup)

        # --- Nulos
        gb_null = QGroupBox("Valores nulos")
        v = QVBoxLayout(gb_null)
        self.chk_high_missing = QCheckBox("Eliminar columnas con >50% nulos")
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

        # --- Columnas a eliminar
        gb_cols = QGroupBox("Columnas a eliminar")
        v = QVBoxLayout(gb_cols)
        self.list_cols = QListWidget()
        self.list_cols.setSelectionMode(QListWidget.MultiSelection)
        v.addWidget(self.list_cols)
        layout.addWidget(gb_cols)

        # --- Botón aplicar
        self.btn_apply = QPushButton("Aplicar preprocesamiento")
        self.btn_apply.clicked.connect(self.apply)
        layout.addWidget(self.btn_apply)

        self.lbl_result = QLabel("")
        layout.addWidget(self.lbl_result)

    def refresh(self):
        self.list_cols.clear()
        if self.state.raw_df is None:
            return
        for col in self.state.raw_df.columns:
            self.list_cols.addItem(QListWidgetItem(col))

    def apply(self):
        if self.state.raw_df is None:
            QMessageBox.warning(self, "Sin datos", "Carga un CSV primero.")
            return
        df = self.state.raw_df.copy()
        try:
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

            self.state.clean_df = df
            self.lbl_result.setText(
                f"Listo: {df.shape[0]} filas × {df.shape[1]} columnas"
            )
            self.data_processed.emit()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))