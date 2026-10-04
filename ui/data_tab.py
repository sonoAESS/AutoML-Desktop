# ui/data_tab.py
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QTableWidget, QTableWidgetItem, QMessageBox, QComboBox, QGroupBox,
    QHeaderView,
)
from core import data_loader, profiling

TASK_LABELS = {
    profiling.CLASSIFICATION: "clasificación",
    profiling.REGRESSION: "regresión",
    None: "no es buen objetivo",
}


class DataTab(QWidget):
    data_loaded = Signal()
    profile_changed = Signal()

    def __init__(self, state):
        super().__init__()
        self.state = state
        self._type_widgets = {}
        self._build_ui()

    # ------------------------------------------------------------------
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

        self.lbl_sugerencia = QLabel("")
        self.lbl_sugerencia.setWordWrap(True)
        layout.addWidget(self.lbl_sugerencia)

        gb_perfil = QGroupBox("Perfil de las columnas (puedes corregir el tipo)")
        v = QVBoxLayout(gb_perfil)
        self.table_perfil = QTableWidget()
        self.table_perfil.setColumnCount(6)
        self.table_perfil.setHorizontalHeaderLabels(
            ["Columna", "Tipo detectado", "Dtype real", "Únicos",
             "% nulos", "Tarea sugerida"]
        )
        self.table_perfil.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self.table_perfil.verticalHeader().setVisible(False)
        v.addWidget(self.table_perfil)

        botones = QHBoxLayout()
        self.btn_usar_objetivo = QPushButton("Usar objetivo sugerido")
        self.btn_usar_objetivo.clicked.connect(self._use_suggested_target)
        self.btn_reperfil = QPushButton("Volver a detectar tipos")
        self.btn_reperfil.clicked.connect(self._reset_types)
        botones.addWidget(self.btn_usar_objetivo)
        botones.addWidget(self.btn_reperfil)
        botones.addStretch()
        v.addLayout(botones)
        layout.addWidget(gb_perfil)

        self.lbl_avisos = QLabel("")
        self.lbl_avisos.setWordWrap(True)
        layout.addWidget(self.lbl_avisos)

        layout.addWidget(QLabel("Vista previa de los datos:"))
        self.table = QTableWidget()
        layout.addWidget(self.table)

    # ------------------------------------------------------------------
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

        self.state.reset_data()
        self.state.raw_df = df
        self.state.clean_df = df.copy()
        self.state.export_df = df.copy()
        self.state.source_path = path
        self.state.profiles = profiling.profile_dataframe(df)
        self._show_df(df)
        self._show_profile()
        self._show_summary()

        info = data_loader.summarize(df)
        self.lbl_info.setText(
            f"{info['n_rows']} filas · {info['n_cols']} columnas · "
            f"{info['missing']} valores nulos"
        )
        self.data_loaded.emit()

    def refresh(self):
        """Repinta el perfil sin volver a leer el archivo."""
        self._show_profile()
        self._show_summary()

    # ------------------------------------------------------------------
    def _show_summary(self):
        sugerencia = profiling.suggest_target(self.state.profiles)
        if sugerencia is None:
            self.lbl_sugerencia.setText(
                "No se detecta una variable objetivo clara: revisa los tipos de "
                "las columnas o elige el objetivo a mano en la pestaña 3."
            )
            self.btn_usar_objetivo.setEnabled(False)
        else:
            tarea = TASK_LABELS.get(sugerencia.target_task, "no es buen objetivo")
            self.lbl_sugerencia.setText(
                f"Sugerencia: usar «{sugerencia.name}» como objetivo "
                f"({sugerencia.n_unique} valores distintos → {tarea})."
            )
            self.btn_usar_objetivo.setEnabled(True)

        avisos = profiling.dataset_warnings(self.state.profiles)
        self.lbl_avisos.setText("\n".join(avisos))

    def _show_profile(self):
        self.table_perfil.blockSignals(True)
        self.table_perfil.setRowCount(0)
        self._type_widgets.clear()

        for fila, profile in enumerate(self.state.profiles):
            self.table_perfil.insertRow(fila)
            self.table_perfil.setItem(
                fila, 0, QTableWidgetItem(profile.name)
            )

            combo = QComboBox()
            for tipo in profiling.SEMANTIC_TYPES:
                combo.addItem(profiling.SEMANTIC_LABELS[tipo], tipo)
            elegido = self.state.column_types.get(
                profile.name, profile.semantic_type
            )
            indice = combo.findData(elegido)
            combo.setCurrentIndex(max(indice, 0))
            self.table_perfil.setCellWidget(fila, 1, combo)
            self._type_widgets[fila] = (profile.name, combo)
            combo.currentIndexChanged.connect(
                lambda _index, nombre=profile.name, box=combo:
                self._on_type_selected(nombre, box)
            )

            self.table_perfil.setItem(fila, 2, QTableWidgetItem(profile.dtype))
            self.table_perfil.setItem(fila, 3, QTableWidgetItem(str(profile.n_unique)))
            self.table_perfil.setItem(
                fila, 4, QTableWidgetItem(f"{profile.missing_pct:.0%}")
            )
            self.table_perfil.setItem(
                fila, 5,
                QTableWidgetItem(TASK_LABELS.get(profile.target_task, "—")),
            )
            if profile.warnings:
                item = QTableWidgetItem(profile.name)
                item.setToolTip("\n".join(profile.warnings))
                self.table_perfil.setItem(fila, 0, item)

        self.table_perfil.blockSignals(False)

    def _on_type_selected(self, nombre, combo):
        elegido = combo.currentData()
        detectado = self._detected_type(nombre)
        if elegido == detectado:
            self.state.column_types.pop(nombre, None)
        else:
            self.state.column_types[nombre] = elegido
        self.profile_changed.emit()

    def _detected_type(self, nombre):
        for profile in self.state.profiles:
            if profile.name == nombre:
                return profile.semantic_type
        return None

    def _reset_types(self):
        self.state.column_types.clear()
        self._show_profile()
        self.profile_changed.emit()

    def _use_suggested_target(self):
        sugerencia = profiling.suggest_target(self.state.profiles)
        if sugerencia is None:
            return
        self.state.target_column = sugerencia.name
        self.state.task_type = sugerencia.target_task
        self.profile_changed.emit()

    # ------------------------------------------------------------------
    def _show_df(self, df, max_rows=200):
        preview = df.head(max_rows)
        self.table.setRowCount(len(preview))
        self.table.setColumnCount(len(preview.columns))
        self.table.setHorizontalHeaderLabels([str(c) for c in preview.columns])
        for i, (_, row) in enumerate(preview.iterrows()):
            for j, val in enumerate(row):
                self.table.setItem(i, j, QTableWidgetItem(str(val)))
        self.table.resizeColumnsToContents()