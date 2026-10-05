# ui/preprocess_tab.py
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QToolBox,
    QVBoxLayout,
    QWidget,
)

from core import balancing, preprocessor, profiling
from ui.empty_state import EmptyState, alternar_estado_vacio
from ui.enabled import con_pista
from ui.layout import acotar_tabla
from ui.theme import ESPACIOS, marcar

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
        """Acordeón de bloques + acciones siempre visibles.

        Los seis grupos medían unos 800 px seguidos, así que había que hacer
        scroll para llegar al botón «Aplicar». En un acordeón cabe un bloque y
        el botón, que es lo que cierra el paso.
        """
        layout = QVBoxLayout(self)

        self.grupos = QToolBox()
        layout.addWidget(self.grupos, 1)

        # Página «Calidad»: duplicados y nulos van juntos porque los dos
        # limpian el dataset y seleienden a la vez.
        self.calidad = QWidget()
        v_calidad = QVBoxLayout(self.calidad)
        v_calidad.setContentsMargins(0, 0, 0, 0)
        v_calidad.setSpacing(ESPACIOS["sm"])

        gb_dups = QGroupBox("Duplicados")
        v = QVBoxLayout(gb_dups)
        self.chk_dups = QCheckBox("Eliminar filas duplicadas")
        self.chk_dups.setChecked(True)
        v.addWidget(self.chk_dups)

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
        v_calidad.addWidget(gb_dups)
        v_calidad.addWidget(gb_null)

        gb_tipos = QGroupBox("Tipos de datos y fechas")
        v = QVBoxLayout(gb_tipos)
        self.lbl_tipos = QLabel("")
        self.lbl_tipos.setWordWrap(True)
        v.addWidget(self.lbl_tipos)
        self.chk_dates = QCheckBox("Convertir las fechas en año, mes y día")
        v.addWidget(self.chk_dates)

        gb_norm = QGroupBox("Normalización de variables numéricas")
        v = QVBoxLayout(gb_norm)
        self.table_norm = QTableWidget()
        self.table_norm.setColumnCount(3)
        self.table_norm.setHorizontalHeaderLabels(
            ["Columna", "Rango actual", "Normalización"]
        )
        self.table_norm.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table_norm.verticalHeader().setVisible(False)
        acotar_tabla(self.table_norm)
        v.addWidget(self.table_norm)

        gb_cols = QGroupBox("Columnas a eliminar")
        v = QVBoxLayout(gb_cols)
        self.list_cols = QListWidget()
        acotar_tabla(self.list_cols)
        self.list_cols.setSelectionMode(QListWidget.MultiSelection)
        v.addWidget(self.list_cols)

        self.gb_distribucion = QGroupBox("Distribución de clases")
        v = QVBoxLayout(self.gb_distribucion)
        self.lbl_distribucion = QLabel("Carga un dataset y elige un objetivo.")
        self.lbl_distribucion.setWordWrap(True)
        v.addWidget(self.lbl_distribucion)
        self.tbl_distribucion = QTableWidget()
        self.tbl_distribucion.setColumnCount(3)
        self.tbl_distribucion.setHorizontalHeaderLabels(["Clase", "Instancias", "%"])
        self.tbl_distribucion.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self.tbl_distribucion.verticalHeader().setVisible(False)
        acotar_tabla(self.tbl_distribucion)
        v.addWidget(self.tbl_distribucion)

        self.btn_apply = QPushButton("Aplicar preprocesamiento")
        marcar(self.btn_apply, "primario")
        self.btn_apply.setMaximumWidth(260)
        self.btn_apply.clicked.connect(self.apply)
        layout.addWidget(self.btn_apply)

        self.lbl_result = QLabel("")
        self.lbl_result.setWordWrap(True)
        self.lbl_result.setProperty("role", "suave")
        layout.addWidget(self.lbl_result)

        # --- montaje del acordeón ---
        self.grupos.addItem(self.calidad, "Calidad (duplicados y nulos)")
        self.grupos.addItem(gb_tipos, "Tipos de datos y fechas")
        self.grupos.addItem(gb_norm, "Normalización de variables")
        self.grupos.addItem(gb_cols, "Columnas a eliminar")
        # La distribución informa en lugar de configurar: se deja visible,
        # pero dentro del acordeón para no romper el flujo de lectura.
        self.grupos.addItem(self.gb_distribucion, "Distribución de clases")
        self._pagina_distribucion = self.grupos.indexOf(self.gb_distribucion)

        self.vacio = EmptyState(
            "No hay datos que limpiar",
            "Aquí se quitan duplicados, se rellenan huecos y se ajustan las "
            "variables, pero hace falta un dataset de partida.",
            "Vuelve al paso 1 y carga un archivo",
        )
        layout.insertWidget(0, self.vacio, 1)

        # Sin datos, ninguno de los grupos sirve todavía.
        self._update_enabled_state()
        self._alternar_vacio()

    # ------------------------------------------------------------------
    def _alternar_vacio(self):
        """Cinco grupos con todo apagado no dicen qué pasa: el mensaje sí.

        `btn_apply` queda fuera a propósito: es la acción que cierra el paso y
        su sitio no depende de si hay contenido que limpiar.
        """
        alternar_estado_vacio(self.vacio, (self.grupos,), self.state.raw_df is None)

    # ------------------------------------------------------------------
    def refresh(self):
        self._fill_columns()
        self._show_casts()
        self._fill_normalizations()
        self._show_distribution()
        self._update_enabled_state()
        self._alternar_vacio()

    # ------------------------------------------------------------------
    def _update_enabled_state(self):
        """Enciende solo los controles que pueden usarse con estos datos.

        Antes los trece controles quedaban habilitados siempre, y el usuario
        veía casillas activas para un dataset que ni siquiera había cargado.

        Idempotente: solo lee `self.state`, así que llamarla dos veces no
        cambia nada.
        """
        hay_datos = self.state.raw_df is not None

        # El perfil semántico refina la decisión: si no hay variables
        # numéricas, la estrategia numérica no tiene a qué aplicarse. Sin esto
        # la única alternativa sería apagar todo en bloque, que deja ruido.
        tipos = {p.semantic_type for p in (self.state.profiles or ())}
        hay_numericas = "numerico" in tipos
        hay_categoricas = bool(tipos & {"categorico", "booleano", "fecha", "texto"})

        sin_datos = "Carga un archivo para usar esta opción."
        sin_numericas = "No hay variables numéricas en este dataset."
        sin_categoricas = "No hay variables categóricas en este dataset."

        con_pista(
            self.chk_dups,
            hay_datos,
            (
                "Quita las filas repetidas, que empujan las métricas hacia el "
                "grupo sobrerrepresentado."
                if hay_datos
                else sin_datos
            ),
        )
        con_pista(
            self.chk_high_missing,
            hay_datos,
            (
                "Elimina las columnas que tienen más de la mitad de valores vacíos."
                if hay_datos
                else sin_datos
            ),
        )
        con_pista(
            self.chk_dates,
            hay_datos,
            (
                "Convierte cada fecha en tres columnas numéricas: año, mes y día."
                if hay_datos
                else sin_datos
            ),
        )
        con_pista(
            self.cmb_num,
            hay_datos and hay_numericas,
            (
                "Cómo se rellenan los huecos de las variables numéricas."
                if hay_datos and hay_numericas
                else (sin_datos if not hay_datos else sin_numericas)
            ),
        )
        con_pista(
            self.cmb_cat,
            hay_datos and hay_categoricas,
            (
                "Cómo se rellenan los huecos de las variables categóricas."
                if hay_datos and hay_categoricas
                else (sin_datos if not hay_datos else sin_categoricas)
            ),
        )
        con_pista(
            self.list_cols,
            hay_datos,
            (
                "Marca aquí las columnas que quieras quitar del dataset."
                if hay_datos
                else sin_datos
            ),
        )
        self.table_norm.setEnabled(hay_datos)
        con_pista(
            self.btn_apply,
            hay_datos,
            (
                "Aplica lo de arriba y deja el dataset listo para entrenar."
                if hay_datos
                else "Carga un archivo para poder aplicar el preprocesamiento."
            ),
        )

    def _show_distribution(self):
        """Tabla con la distribución de clases del objetivo elegido."""
        objetivo = self.state.target_column
        if (
            self.state.task_type != "classification"
            or not objetivo
            or self.state.clean_df is None
            or objetivo not in self.state.clean_df.columns
        ):
            self.gb_distribucion.setVisible(False)
            return

        self.gb_distribucion.setVisible(True)
        serie = self.state.clean_df[objetivo]
        distribucion = balancing.class_distribution(serie)
        self.state.class_distribution = distribucion.as_dict()

        self.lbl_distribucion.setText(distribucion.describe())
        if distribucion.is_imbalanced:
            self.lbl_distribucion.setStyleSheet("font-weight: bold;")
        else:
            self.lbl_distribucion.setStyleSheet("")

        self.tbl_distribucion.setRowCount(len(distribucion.rows))
        for fila, fila_datos in enumerate(distribucion.rows):
            self.tbl_distribucion.setItem(
                fila, 0, QTableWidgetItem(str(fila_datos.clase))
            )
            self.tbl_distribucion.setItem(fila, 1, QTableWidgetItem(str(fila_datos.n)))
            self.tbl_distribucion.setItem(
                fila, 2, QTableWidgetItem(f"{fila_datos.pct:.1%}")
            )

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
        sugeridas = preprocessor.suggested_normalizations(self.state.raw_df, numericas)

        for fila, profile in enumerate(numericas):
            self.table_norm.insertRow(fila)
            self.table_norm.setItem(fila, 0, QTableWidgetItem(profile.name))
            self.table_norm.setItem(fila, 1, QTableWidgetItem(profile.value_range))

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
