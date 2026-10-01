# ui/data_tab.py
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFileDialog, QTableWidget, QTableWidgetItem,
    QMessageBox,
)
from PySide6.QtGui import QAction
from core import data_loader

class DataTab(QWidget):
    data_loaded = Signal()

    def __init__(self, state):
        super().__init__()
        self.state = state
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        self.btn_load = QPushButton("Cargar CSV…")
        self.btn_load.clicked.connect(self.load_csv)
        self.lbl_info = QLabel("Ningún archivo cargado")
        top.addWidget(self.btn_load)
        top.addWidget(self.lbl_info)
        top.addStretch()
        layout.addLayout(top)

        self.table = QTableWidget()
        layout.addWidget(self.table)

    def load_csv(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Seleccionar CSV", "", "CSV (*.csv *.txt *.tsv)"
        )
        if not path:
            return
        try:
            df = data_loader.load_csv(path)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            return

        self.state.raw_df = df
        self.state.clean_df = df.copy()
        self._show_df(df)

        info = data_loader.summarize(df)
        self.lbl_info.setText(
            f"{info['n_rows']} filas · {info['n_cols']} columnas · "
            f"{info['missing']} valores nulos"
        )
        self.data_loaded.emit()

    def _show_df(self, df, max_rows=200):
        preview = df.head(max_rows)
        self.table.setRowCount(len(preview))
        self.table.setColumnCount(len(preview.columns))
        self.table.setHorizontalHeaderLabels([str(c) for c in preview.columns])
        for i, (_, row) in enumerate(preview.iterrows()):
            for j, val in enumerate(row):
                self.table.setItem(i, j, QTableWidgetItem(str(val)))
        self.table.resizeColumnsToContents()