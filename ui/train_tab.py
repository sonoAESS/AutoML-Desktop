# ui/train_tab.py
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QToolBox,
    QVBoxLayout,
    QWidget,
)

from core import balancing, feature_selection, model_specs, model_trainer, profiling
from ui.enabled import con_pista
from ui.layout import acotar_tabla
from ui.theme import ESPACIOS, marcar
from ui.workers import SelectionWorker, TuneWorker, WorkerThread

TASK_LABELS = {
    "classification": "Clasificación",
    "regression": "Regresión",
}


class TrainTab(QWidget):
    model_trained = Signal()
    config_changed = Signal()

    def __init__(self, state):
        super().__init__()
        self.state = state
        self._thread = None
        self._param_widgets = {}
        self._run_context = None  # ajustes con los que se lanzó la búsqueda
        self._elapsed = 0
        self._status = ""
        self._done = 0
        self._total = 0
        self._fallo = False
        self._build_ui()
        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._on_tick)

    # ------------------------------------------------------------------
    def _build_ui(self):
        """Acordeón de configuración + acción principal siempre visible.

        Los cinco bloques sumaban más de 900 px seguidos y el botón
        «Entrenar modelo» quedaba por debajo del borde de la ventana. Con el
        acordeón el botón está siempre a la vista, que es lo que importa.
        """
        layout = QVBoxLayout(self)

        self.grupos = QToolBox()
        layout.addWidget(self.grupos, 1)

        # --- página «Modelo» ---
        self.pagina_modelo = QWidget()
        v_modelo = QVBoxLayout(self.pagina_modelo)
        v_modelo.setContentsMargins(0, 0, 0, 0)
        v_modelo.setSpacing(ESPACIOS["sm"])

        form = QFormLayout()

        self.cmb_target = QComboBox()
        self.cmb_task = QComboBox()
        self.cmb_task.addItem("Clasificación", "classification")
        self.cmb_task.addItem("Regresión", "regression")
        self.cmb_family = QComboBox()
        self.cmb_model = QComboBox()
        self.cmb_metric = QComboBox()

        form.addRow("Variable objetivo:", self.cmb_target)
        form.addRow("Tipo de tarea:", self.cmb_task)
        form.addRow("Familia de modelo:", self.cmb_family)
        form.addRow("Modelo:", self.cmb_model)
        v_modelo.addLayout(form)

        self.lbl_compatibilidad = QLabel("")
        self.lbl_compatibilidad.setWordWrap(True)
        v_modelo.addWidget(self.lbl_compatibilidad)

        self.lbl_modelo = QLabel("")
        self.lbl_modelo.setWordWrap(True)
        self.lbl_modelo.setProperty("role", "suave")
        v_modelo.addWidget(self.lbl_modelo)
        v_modelo.addStretch()

        gb_params = QGroupBox("Hiperparámetros del modelo")
        grid = QGridLayout(gb_params)
        self.grid_params = grid
        self.gb_params = gb_params

        gb_tune = QGroupBox("Búsqueda de hiperparámetros")
        form_tune = QFormLayout(gb_tune)
        self.chk_tune = QCheckBox("Buscar la mejor combinación")
        self.chk_tune.setToolTip(
            "Prueba varias combinaciones de hiperparámetros y quédate con "
            "la mejor. Tarda bastante más que un entrenamiento normal."
        )
        self.cmb_search = QComboBox()
        self.cmb_search.addItem("Todas (grid)", "grid")
        self.cmb_search.addItem("Aleatoria (random)", "random")
        self.spn_cv = QSpinBox()
        self.spn_cv.setRange(2, 10)
        self.spn_cv.setValue(5)
        self.spn_iter = QSpinBox()
        self.spn_iter.setRange(5, 200)
        self.spn_iter.setValue(20)
        form_tune.addRow(self.chk_tune)
        form_tune.addRow("Estrategia:", self.cmb_search)
        form_tune.addRow("Folds (validación cruzada):", self.spn_cv)
        form_tune.addRow("Iteraciones (si es aleatoria):", self.spn_iter)
        form_tune.addRow("Métrica a optimizar:", self.cmb_metric)
        self.gb_tune = gb_tune

        self.pagina_ajustes = QWidget()
        v_ajustes = QVBoxLayout(self.pagina_ajustes)
        v_ajustes.setContentsMargins(0, 0, 0, 0)
        v_ajustes.setSpacing(ESPACIOS["sm"])
        v_ajustes.addWidget(gb_params)
        v_ajustes.addWidget(gb_tune)
        v_ajustes.addStretch()

        gb_balanceo = QGroupBox("Balanceo de clases")
        form_balanceo = QFormLayout(gb_balanceo)
        self.cmb_balancing = QComboBox()
        self.spn_vecinos = QSpinBox()
        self.spn_vecinos.setRange(2, 10)
        self.spn_vecinos.setValue(5)
        self.spn_vecinos.setToolTip(
            "Vecinos que usa SMOTE para crear los ejemplos sintéticos."
        )
        form_balanceo.addRow("Estrategia:", self.cmb_balancing)
        form_balanceo.addRow("Vecinos (SMOTE):", self.spn_vecinos)
        self.gb_balanceo = gb_balanceo

        gb_seleccion = QGroupBox("Selección de atributos")
        v_seleccion = QVBoxLayout(gb_seleccion)
        self.chk_seleccion = QCheckBox("Seleccionar solo los atributos más útiles")
        self.chk_seleccion.setToolTip(
            "Analiza qué atributos aportan al modelo y descarta los "
            "débiles. Acelera el entrenamiento y suele mejorar laexactitud."
        )
        self.chk_seleccion.toggled.connect(self._on_seleccion_toggled)
        v_seleccion.addWidget(self.chk_seleccion)

        form_seleccion = QFormLayout()
        self.cmb_metodo_sel = QComboBox()
        self.cmb_criterio_sel = QComboBox()
        self.cmb_criterio_sel.addItem("Número de atributos (k)", "k")
        self.cmb_criterio_sel.addItem("Porcentaje", "percentile")
        self.spn_sel = QSpinBox()
        self.spn_sel.setRange(1, 10000)
        self.spn_sel.setValue(10)
        self.lbl_sel = QLabel("")
        self.lbl_sel.setWordWrap(True)
        form_seleccion.addRow("Método:", self.cmb_metodo_sel)
        form_seleccion.addRow("Criterio:", self.cmb_criterio_sel)
        form_seleccion.addRow("Valor:", self.spn_sel)
        form_seleccion.addRow(QLabel(""), self.lbl_sel)
        v_seleccion.addLayout(form_seleccion)

        self.btn_analizar = QPushButton("Analizar selección")
        self.btn_analizar.clicked.connect(self.analizar_seleccion)
        self.btn_analizar.setEnabled(False)
        v_seleccion.addWidget(self.btn_analizar)

        self.tbl_seleccion = QTableWidget()
        self.tbl_seleccion.setColumnCount(4)
        self.tbl_seleccion.setHorizontalHeaderLabels(
            ["Atributo", "Columnas codificadas", "Puntuación", "Seleccionado"]
        )
        self.tbl_seleccion.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self.tbl_seleccion.verticalHeader().setVisible(False)
        acotar_tabla(self.tbl_seleccion)
        v_seleccion.addWidget(self.tbl_seleccion)
        self.gb_seleccion = gb_seleccion

        self.btn_train = QPushButton("Entrenar modelo")
        marcar(self.btn_train, "primario")
        self.btn_train.setMaximumWidth(240)
        self.btn_train.clicked.connect(self.train)
        layout.addWidget(self.btn_train)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # modo "ocupado": no hay porcentaje
        self.progress.setFormat("Trabajando…")
        self.progress.hide()
        layout.addWidget(self.progress)

        # Estado de la búsqueda: qué está pasando y cuánto lleva. No se
        # sobrescribe con las métricas, eso es cosa de `lbl_result`.
        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

        self.lbl_result = QLabel("")
        self.lbl_result.setWordWrap(True)
        layout.addWidget(self.lbl_result)

        # --- montaje del acordeón ---
        # Solo se muestran las páginas con contenido real: una página de
        # «balanceo» vacía en un problema de regresión no informa de nada.
        self._paginas_grupo = (
            (self.pagina_modelo, "Qué modelo entrenar"),
            (self.pagina_ajustes, "Hiperparámetros"),
            (gb_balanceo, "Balanceo de clases"),
            (gb_seleccion, "Selección de atributos"),
        )
        for widget, titulo in self._paginas_grupo:
            self.grupos.addItem(widget, titulo)
        self._pagina_balanceo = self.grupos.indexOf(gb_balanceo)
        self._pagina_seleccion = self.grupos.indexOf(gb_seleccion)

        self.cmb_target.currentIndexChanged.connect(self._on_target_changed)
        self.cmb_task.currentIndexChanged.connect(self._on_task_changed)
        self.cmb_family.currentIndexChanged.connect(self._refresh_models)
        self.cmb_model.currentIndexChanged.connect(self._on_model_changed)
        self.chk_tune.toggled.connect(self._toggle_tune_options)
        self.cmb_balancing.currentIndexChanged.connect(self._on_balancing_changed)
        self.cmb_metodo_sel.currentIndexChanged.connect(self._on_metodo_sel_changed)
        self.cmb_criterio_sel.currentIndexChanged.connect(self._on_criterio_sel_changed)

        self._refresh_families()
        self._refresh_models()
        self._toggle_tune_options(False)

        # Sin dataset limpio todavía no se puede entrenar.
        self._update_enabled_state()

    # ------------------------------------------------------------------
    def refresh(self):
        self._reload_targets()
        self._refresh_balancing()
        self._refresh_seleccion()
        self._update_enabled_state()

    # ------------------------------------------------------------------
    def _update_enabled_state(self):
        """Activa lo que se puede usar con el estado actual.

        «Entrenar modelo» exige dataset limpio, objetivo y modelo: antes se
        podía pulsar sin modelo y el error salía como diálogo.

        La búsqueda de hiperparámetros y los vecinos de SMOTE ya tenían su
        propia lógica en `_toggle_tune_options` y `_refresh_balancing`; aquí
        solo se recuerda el criterio para que las tres condiciones vivan en el
        mismo sitio.
        """
        listo = bool(
            self.state.clean_df is not None
            and self.state.target_column
            and self.state.model_name
        )
        con_pista(
            self.btn_train,
            listo,
            (
                "Entrena el modelo elegido con los datos ya preprocesados."
                if listo
                else (
                    "Carga un dataset y preprocésalo para poder entrenar."
                    if self.state.clean_df is None
                    else "Elige la variable objetivo y el modelo que quieres entrenar."
                )
            ),
        )
        con_pista(
            self.btn_analizar,
            bool(self.state.export_df is not None and self.chk_seleccion.isChecked()),
            (
                "Calcula qué atributos son más útiles, sin tocar los datos."
                if self.state.export_df is not None and self.chk_seleccion.isChecked()
                else (
                    "Activa la selección de atributos para poder analizarlos."
                    if not self.chk_seleccion.isChecked()
                    else "Aplica el preprocesamiento para tener el dataset transformado."
                )
            ),
        )

    def _reload_targets(self):
        anterior = self.state.target_column
        self.cmb_target.blockSignals(True)
        self.cmb_target.clear()
        if self.state.clean_df is not None:
            self.cmb_target.addItems(list(self.state.clean_df.columns))
            indice = self.cmb_target.findText(anterior or "")
            if indice < 0 and self.state.profiles:
                sugerida = profiling.suggest_target(self.state.profiles)
                if sugerida is not None:
                    indice = self.cmb_target.findText(sugerida.name)
                    self.state.target_column = sugerida.name
            if indice >= 0:
                self.cmb_target.setCurrentIndex(indice)
        self.cmb_target.blockSignals(False)
        self._on_target_changed()

    def _on_target_changed(self):
        columna = self.cmb_target.currentText()
        self.state.target_column = columna or None
        sugerida = self._suggested_task(columna)
        self.config_changed.emit()
        indice = self.cmb_task.findData(sugerida) if sugerida else -1
        if indice >= 0 and self.cmb_task.currentData() != sugerida:
            self.cmb_task.blockSignals(True)
            self.cmb_task.setCurrentIndex(indice)
            self.cmb_task.blockSignals(False)
        self._on_task_changed()

    def _suggested_task(self, columna):
        if not columna or self.state.clean_df is None:
            return None
        if columna not in self.state.clean_df.columns:
            return None
        return profiling.suggest_task_type(self.state.clean_df[columna])

    def _on_task_changed(self):
        self.state.task_type = self.cmb_task.currentData()
        self._refresh_families()
        self._refresh_models()
        self._refresh_balancing()
        self._refresh_seleccion()
        self.config_changed.emit()

    # ------------------------------------------------------------------
    def _refresh_families(self):
        task = self.cmb_task.currentData()
        self.cmb_family.blockSignals(True)
        self.cmb_family.clear()
        self.cmb_family.addItem("Todas", model_specs.ALL_FAMILIES)
        for familia in model_specs.list_families(task):
            self.cmb_family.addItem(model_specs.family_label(familia), familia)
        self.cmb_family.blockSignals(False)

    def _refresh_models(self):
        task = self.cmb_task.currentData()
        familia = self.cmb_family.currentData()
        self.cmb_model.blockSignals(True)
        self.cmb_model.clear()

        resultados = self._compatibility(task)
        if resultados is None:
            self.cmb_model.blockSignals(False)
            self.lbl_compatibilidad.setText(
                "Carga y preprocesa datos para ver los modelos disponibles."
            )
            self._on_model_changed()
            return

        compatibles = {item.name for item in resultados if item.compatible}
        for nombre in model_specs.list_models(task, familia):
            if nombre in compatibles:
                self.cmb_model.addItem(nombre, nombre)
        self.cmb_model.blockSignals(False)

        if not self.cmb_model.count():
            self.lbl_compatibilidad.setText(
                "Ningún modelo puede usarse con estos datos: "
                + model_specs.summarize_compatibility(resultados)
            )
            self.btn_train.setEnabled(False)
        else:
            self.lbl_compatibilidad.setText(
                model_specs.summarize_compatibility(resultados)
            )
            self.btn_train.setEnabled(True)

        self._on_model_changed()

    def _compatibility(self, task):
        if self.state.clean_df is None:
            return None
        profiles = self._feature_profiles()
        return model_specs.evaluate_compatibility(
            task,
            profiles,
            target=self.state.target_column,
            n_rows=len(self.state.clean_df),
        )

    def _feature_profiles(self):
        """Perfil de las columnas que se usarán como predictoras."""
        if self.state.clean_df is None:
            return []
        nombres = [
            column
            for column in self.state.clean_df.columns
            if column != self.state.target_column
        ]
        return [
            profiling.profile_column(self.state.clean_df[column]) for column in nombres
        ]

    def _on_model_changed(self):
        model_name = self.cmb_model.currentData()
        self.state.model_name = model_name
        self.state.family = (
            model_specs.get_spec(model_name, self.cmb_task.currentData()).family
            if model_name
            else None
        )
        self._build_param_widgets(model_name)
        self._refresh_metrics()
        self._refresh_balancing()

        if model_name:
            spec = model_specs.get_spec(model_name, self.cmb_task.currentData())
            self.lbl_modelo.setText(
                f"{spec.notes}  (familia: {model_specs.family_label(spec.family)})"
            )
        else:
            self.lbl_modelo.setText("")

    def _build_param_widgets(self, model_name):
        while self.grid_params.count():
            item = self.grid_params.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._param_widgets.clear()

        if not model_name:
            return
        spec = model_specs.get_spec(model_name, self.cmb_task.currentData())
        # Con balanceo activo el peso de clases lo pone el pipeline, así que el
        # control manual estorbaría: podría contradecir la estrategia elegida.
        balanco_activo = (self._balancing_config() or {}).get(
            "method"
        ) == "class_weight"
        for fila, param in enumerate(spec.params):
            if param.name == "class_weight" and balanco_activo:
                continue
            combo = QComboBox()
            combo.addItem(
                f"Probar {model_specs.ALL_VALUES}…",
                model_specs.ALL_VALUES,
            )
            for valor in param.values:
                combo.addItem(param.display(valor), valor)
            self.grid_params.addWidget(QLabel(param.label), fila, 0)
            self.grid_params.addWidget(combo, fila, 1)
            self._param_widgets[param.name] = combo

    def _selections(self):
        return {
            nombre: combo.currentData() for nombre, combo in self._param_widgets.items()
        }

    def _refresh_metrics(self):
        task = self.cmb_task.currentData()
        metricas = model_trainer.available_metrics(task, self.cmb_model.currentData())
        self.cmb_metric.blockSignals(True)
        self.cmb_metric.clear()
        self.cmb_metric.addItems(metricas)
        default = {
            "classification": "accuracy",
            "regression": "r2",
        }.get(task, metricas[0] if metricas else "")
        indice = self.cmb_metric.findText(default)
        if indice >= 0:
            self.cmb_metric.setCurrentIndex(indice)
        self.cmb_metric.blockSignals(False)

    def _refresh_balancing(self):
        """Rellena el desplegable según la tarea y el modelo elegidos."""
        task = self.cmb_task.currentData()
        model_name = self.cmb_model.currentData()
        estrategias = balancing.available_strategies(task, model_name)

        self.cmb_balancing.blockSignals(True)
        self.cmb_balancing.clear()
        for estrategia in estrategias:
            spec = balancing.BALANCING_STRATEGIES[estrategia]
            self.cmb_balancing.addItem(spec.label, estrategia)
        self.cmb_balancing.blockSignals(False)

        self._balancing_active = (
            task == "classification" and self.cmb_target.count() > 0
        )
        self.gb_balanceo.setVisible(self._balancing_active)
        if not self._balancing_active:
            return

        metodo = self.cmb_balancing.currentData()
        self.spn_vecinos.setEnabled(metodo in ("smote", "smoten"))
        self.state.balancing = metodo

    def _on_balancing_changed(self):
        """Solo SMOTE piden vecinos; el resto de estrategias no los usan."""
        metodo = self.cmb_balancing.currentData()
        self.spn_vecinos.setEnabled(metodo in ("smote", "smoten"))
        self.state.balancing = metodo
        # El peso de clases puede desaparecer de los hiperparámetros.
        self._build_param_widgets(self.cmb_model.currentData())
        self.config_changed.emit()

    def _balancing_config(self):
        """Lo que espera `build_pipeline`, o `None` si no hay balanceo."""
        if not getattr(self, "_balancing_active", False):
            return None
        metodo = self.cmb_balancing.currentData()
        if not metodo or metodo == "none":
            return None
        config = {"method": metodo}
        if metodo in ("smote", "smoten"):
            config["k_neighbors"] = self.spn_vecinos.value()
        return config

    # ------------------------------------------------------------------
    # Selección de atributos
    # ------------------------------------------------------------------
    def _refresh_seleccion(self):
        """Puebla el desplegable de métodos con los válidos para la tarea."""
        task = self.cmb_task.currentData()
        metodos = feature_selection.available_methods(task)
        activo = bool(metodos) and self.cmb_target.count() > 0
        self.gb_seleccion.setVisible(activo)
        if not activo:
            self.chk_seleccion.setChecked(False)
            return

        self.cmb_metodo_sel.blockSignals(True)
        self.cmb_metodo_sel.clear()
        for metodo in metodos:
            self.cmb_metodo_sel.addItem(feature_selection.method_label(metodo), metodo)
        self.cmb_metodo_sel.blockSignals(False)

        # "ANOVA F" es el punto de partida: rápido y sin supuestos raros.
        indice = self.cmb_metodo_sel.findData("anova")
        if indice >= 0:
            self.cmb_metodo_sel.setCurrentIndex(indice)
        self._on_seleccion_toggled(self.chk_seleccion.isChecked())

    def _on_seleccion_toggled(self, activo: bool):
        """La casilla gobierna si el selector entra en el pipeline."""
        self.cmb_metodo_sel.setEnabled(activo)
        self.cmb_criterio_sel.setEnabled(activo)
        self.btn_analizar.setEnabled(activo and self.cmb_target.count() > 0)
        self._on_metodo_sel_changed()
        self._guardar_seleccion_en_estado()

    def _on_metodo_sel_changed(self):
        """`embedded` no admite porcentaje: se avisa y no se ofrece."""
        metodo = self.cmb_metodo_sel.currentData()
        spec = feature_selection.SELECTION_METHODS.get(metodo)
        admite = spec.supports_percentile if spec else False
        porcentaje = self.cmb_criterio_sel.currentData() == "percentile"

        indice_pct = self.cmb_criterio_sel.findData("percentile")
        if not admite:
            self.cmb_criterio_sel.model().item(indice_pct).setEnabled(False)
            if porcentaje:
                self.cmb_criterio_sel.setCurrentIndex(
                    self.cmb_criterio_sel.findData("k")
                )
        else:
            self.cmb_criterio_sel.model().item(indice_pct).setEnabled(True)

        self.lbl_sel.setText(spec.docstring if spec else "")
        self._on_criterio_sel_changed()
        self._guardar_seleccion_en_estado()

    def _on_criterio_sel_changed(self):
        """El mismo spin box sirve para `k` (entero) y porcentaje (1-100)."""
        spec = feature_selection.SELECTION_METHODS.get(
            self.cmb_metodo_sel.currentData()
        )
        porcentaje = self.cmb_criterio_sel.currentData() == "percentile"
        if porcentaje and spec is not None and not spec.supports_percentile:
            # Si el porcentaje llega con un método que no lo admite, se vuelve a
            # `k`: si no, el mismo número significaría cosas distintas.
            self.cmb_criterio_sel.setCurrentIndex(self.cmb_criterio_sel.findData("k"))
            return
        self.spn_sel.setRange(1, 100 if porcentaje else 10000)
        self.spn_sel.setValue(30 if porcentaje else 10)

    def _guardar_seleccion_en_estado(self):
        """Deja en `AppState` lo que el usuario haya elegido."""
        config = self._selection_config()
        self.state.selection_method = (config or {}).get("method")
        self.state.selection_criteria = (
            {clave: valor for clave, valor in config.items() if clave != "method"}
            if config
            else None
        )

    def _selection_config(self):
        """Lo que espera `build_pipeline`, o `None` si no se selecciona."""
        if not self.chk_seleccion.isChecked():
            return None
        metodo = self.cmb_metodo_sel.currentData()
        spec = feature_selection.SELECTION_METHODS.get(metodo)
        if not metodo or spec is None:
            return None
        # `embedded` no admite porcentaje: si llega de algún modo, se corta por
        # número de atributos en lugar de dejar que `core` avise al entrenar.
        admite_pct = spec.supports_percentile
        if admite_pct and self.cmb_criterio_sel.currentData() == "percentile":
            return {
                "method": metodo,
                "percentile": float(self.spn_sel.value()),
            }
        return {"method": metodo, "k": int(self.spn_sel.value())}

    def analizar_seleccion(self):
        """Ajusta el selector en un hilo aparte y pinta su tabla."""
        config = self._selection_config()
        if config is None:
            QMessageBox.warning(
                self, "Sin selección", "Marca la casilla para analizar la selección."
            )
            return
        if self.cmb_model.currentData() is None:
            QMessageBox.warning(self, "Sin modelo", "Elige un modelo válido.")
            return

        self.btn_analizar.setEnabled(False)
        self.progress.setVisible(True)
        self.lbl_sel.setText("Analizando la selección…")

        worker = SelectionWorker(
            df=self.state.clean_df,
            target=self.cmb_target.currentText(),
            model_name=self.cmb_model.currentData(),
            task_type=self.cmb_task.currentData(),
            selection=config,
            casts=self.state.column_types or None,
            normalizations=self.state.normalizations or None,
        )
        self._seleccion_thread = WorkerThread(worker, self)
        worker.finished.connect(self._on_seleccion_lista)
        worker.failed.connect(self._on_seleccion_fallida)
        self._seleccion_thread.finished.connect(self._on_seleccion_terminada)
        self._seleccion_thread.start()

    def _on_seleccion_lista(self, tabla):
        self._pintar_seleccion(tabla)

    def _on_seleccion_fallida(self, mensaje):
        self.progress.setVisible(False)
        self.lbl_sel.setText("No se pudo analizar la selección.")
        QMessageBox.critical(self, "Error al analizar la selección", mensaje)

    def _on_seleccion_terminada(self):
        self.progress.setVisible(False)
        self.btn_analizar.setEnabled(self.chk_seleccion.isChecked())

    def _pintar_seleccion(self, tabla):
        """Rellena la tabla; lo no seleccionado se muestra atenuado."""
        self.tbl_seleccion.setRowCount(len(tabla))
        atenuado = QBrush(QColor(Qt.GlobalColor.gray))
        normal = QBrush(QColor(Qt.GlobalColor.black))
        for fila, datos in enumerate(tabla.itertuples(index=False)):
            self.tbl_seleccion.setItem(fila, 0, QTableWidgetItem(str(datos.atributo)))
            self.tbl_seleccion.setItem(
                fila, 1, QTableWidgetItem(str(int(datos.n_columnas_codificadas)))
            )
            puntuacion = datos.puntuacion
            texto = "n/d" if puntuacion != puntuacion else f"{puntuacion:.4f}"  # NaN
            self.tbl_seleccion.setItem(fila, 2, QTableWidgetItem(texto))
            self.tbl_seleccion.setItem(
                fila, 3, QTableWidgetItem("Sí" if datos.seleccionado else "No")
            )
            for columna in range(4):
                item = self.tbl_seleccion.item(fila, columna)
                item.setForeground(normal if datos.seleccionado else atenuado)

        self._guardar_seleccion_en_estado()
        self.lbl_sel.setText(
            f"{int(tabla['seleccionado'].sum())} de {len(tabla)} atributos "
            "seleccionados."
        )

    def _toggle_tune_options(self, enabled):
        for w in (self.cmb_search, self.spn_cv, self.spn_iter, self.cmb_metric):
            w.setEnabled(enabled)

    # ------------------------------------------------------------------
    def train(self):
        if self.state.clean_df is None or self.cmb_target.count() == 0:
            QMessageBox.warning(self, "Sin datos", "Carga y preprocesa datos primero.")
            return
        if self.cmb_model.currentData() is None:
            QMessageBox.warning(self, "Sin modelo", "Elige un modelo válido.")
            return
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(self, "En curso", "Ya hay un entrenamiento activo.")
            return

        target = self.cmb_target.currentText()
        task = self.cmb_task.currentData()
        model_name = self.cmb_model.currentData()
        transform = {
            "casts": self.state.column_types or None,
            "normalizations": self.state.normalizations or None,
        }
        balancing_config = self._balancing_config()
        selection_config = self._selection_config()
        # Se guarda con qué se lanzó: al terminar se usan estos valores y no
        # los de los desplegables, que durante la búsqueda están bloqueados.
        self._run_context = {
            "target": target,
            "task": task,
            "model_name": model_name,
            "balancing": balancing_config,
            "selection": selection_config,
        }

        if self.chk_tune.isChecked():
            search_type = self.cmb_search.currentData()
            cv = self.spn_cv.value()
            n_iter = self.spn_iter.value()
            combos, ajustes = model_trainer.search_size(
                model_name,
                task,
                self._selections(),
                search_type=search_type,
                cv=cv,
                n_iter=n_iter,
            )
            worker = TuneWorker(
                df=self.state.clean_df,
                target=target,
                model_name=model_name,
                task_type=task,
                search_type=search_type,
                metric=self.cmb_metric.currentText(),
                cv=cv,
                n_iter=n_iter,
                selections=self._selections(),
                balancing=balancing_config,
                selection=selection_config,
                **transform,
            )
            self._thread = WorkerThread(worker, self)
            worker.finished.connect(self._on_finished)
            worker.failed.connect(self._on_failed)
            worker.progress.connect(self._on_progress)
            self._thread.finished.connect(self._on_thread_done)
            self._set_busy(True, total=ajustes, combos=combos)
            self._thread.start()
        else:
            try:
                pipe, metrics, test_data = model_trainer.train_model(
                    self.state.clean_df,
                    target,
                    model_name,
                    task,
                    balancing=balancing_config,
                    selection=selection_config,
                    **transform,
                )
            except Exception as e:
                QMessageBox.critical(self, "Error al entrenar", str(e))
                return
            self._store_result(pipe, metrics, test_data, {})
            self.lbl_status.setText("Entrenamiento completado.")

    # ------------------------------------------------------------------
    def _on_progress(self, etapa, actual, total):
        """Etapas que van llegando desde el hilo de la búsqueda."""
        self._status = etapa
        self._total = total or self._total
        self._done = actual
        self._update_status()

    def _on_tick(self):
        """Cada segundo: cuenta el tiempo transcurrido."""
        self._elapsed += 1
        self._update_status()

    def _update_status(self):
        partes = [self._status]
        if self._total:
            partes.append(f"{self._done}/{self._total} ajustes")
        if self._elapsed:
            partes.append(f"{self._elapsed} s")
        self.lbl_status.setText(" · ".join(p for p in partes if p))

    def _on_finished(self, pipe, metrics, test_data, cv_results):
        self._store_result(pipe, metrics, test_data, cv_results)

    def _on_failed(self, message):
        self._fallo = True
        self._set_busy(False)
        self.lbl_status.setText(f"Búsqueda fallida: {message}")
        QMessageBox.critical(self, "Error al buscar hiperparámetros", message)

    def _on_thread_done(self):
        self._set_busy(False)
        self._thread = None
        if not self._fallo:
            tiempo = f"en {self._elapsed} s" if self._elapsed else "en menos de 1 s"
            self.lbl_status.setText(
                f"Búsqueda terminada {tiempo}. " "Los mejores parámetros están arriba."
            )
        self._fallo = False

    def _set_busy(self, busy: bool, total: int = 0, combos: int = 0):
        """Muestra u oculta el indicador y bloquea los controles mientras tanto."""
        self.btn_train.setEnabled(not busy)
        self.progress.setVisible(busy)
        self._lock_controls(not busy)

        if busy:
            self._elapsed = 0
            self._done = 0
            self._total = total
            self._status = (
                f"Buscando: {combos} combinaciones, {total} ajustes"
                if combos
                else "Buscando la mejor configuración"
            )
            self._update_status()
            self._timer.start()
        else:
            self._timer.stop()

    def _lock_controls(self, enabled: bool):
        """Impide cambiar la configuración mientras dura la búsqueda."""
        for widget in (
            self.cmb_target,
            self.cmb_task,
            self.cmb_family,
            self.cmb_model,
            self.gb_params,
            self.gb_tune,
            self.chk_seleccion,
        ):
            widget.setEnabled(enabled)
        if enabled:
            # `_lock_controls(False)` no debe dejar la selección activa ni el
            # análisis disponible: se restauran con la configuración actual.
            self._on_seleccion_toggled(self.chk_seleccion.isChecked())
        else:
            # Al desbloquear, el estado depende de `AppState`, no de que
            # acaba de terminar un entrenamiento: si el objetivo no está
            # elegido, «Entrenar» debe seguir apagado.
            self._update_enabled_state()

    def _store_result(self, pipe, metrics, test_data, cv_results):
        contexto = self._run_context or {}
        self.state.pipeline = pipe
        self.state.trained_model = pipe
        self.state.metrics = metrics
        self.state.task_type = contexto.get("task", self.cmb_task.currentData())
        self.state.target_column = contexto.get("target", self.cmb_target.currentText())
        self.state.model_name = contexto.get("model_name", self.cmb_model.currentData())
        self.state._test_data = test_data
        self.state._cv_results = cv_results

        # El dataset que se exportará se transforma con el pipeline ya
        # ajustado, para que use las mismas constantes que el modelo.
        try:
            self.state.export_df = model_trainer.transform_with_pipeline(
                self.state.clean_df, pipe
            )
        except Exception:
            self.state.export_df = self.state.clean_df.copy()

        lineas = model_trainer.summarize_metrics(metrics)
        if isinstance(metrics.get("best_params"), dict):
            lineas.append(f"mejores parámetros: {metrics['best_params']}")
        self.lbl_result.setText(" · ".join(lineas))
        self.model_trained.emit()
