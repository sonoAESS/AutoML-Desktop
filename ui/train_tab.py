# ui/train_tab.py
from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core import balancing, model_specs, model_trainer, profiling
from ui.workers import TuneWorker, WorkerThread

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
        layout = QVBoxLayout(self)
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
        layout.addLayout(form)

        self.lbl_compatibilidad = QLabel("")
        self.lbl_compatibilidad.setWordWrap(True)
        layout.addWidget(self.lbl_compatibilidad)

        self.lbl_modelo = QLabel("")
        self.lbl_modelo.setWordWrap(True)
        layout.addWidget(self.lbl_modelo)

        gb_params = QGroupBox("Hiperparámetros del modelo")
        grid = QGridLayout(gb_params)
        self.grid_params = grid
        self.gb_params = gb_params
        layout.addWidget(gb_params)

        gb_tune = QGroupBox("Búsqueda de hiperparámetros")
        form_tune = QFormLayout(gb_tune)
        self.chk_tune = QCheckBox("Buscar la mejor combinación")
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
        layout.addWidget(gb_tune)

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
        layout.addWidget(gb_balanceo)

        self.btn_train = QPushButton("Entrenar modelo")
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

        self.cmb_target.currentIndexChanged.connect(self._on_target_changed)
        self.cmb_task.currentIndexChanged.connect(self._on_task_changed)
        self.cmb_family.currentIndexChanged.connect(self._refresh_models)
        self.cmb_model.currentIndexChanged.connect(self._on_model_changed)
        self.chk_tune.toggled.connect(self._toggle_tune_options)
        self.cmb_balancing.currentIndexChanged.connect(self._on_balancing_changed)

        self._refresh_families()
        self._refresh_models()
        self._toggle_tune_options(False)

    # ------------------------------------------------------------------
    def refresh(self):
        self._reload_targets()
        self._refresh_balancing()

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
        # Se guarda con qué se lanzó: al terminar se usan estos valores y no
        # los de los desplegables, que durante la búsqueda están bloqueados.
        self._run_context = {
            "target": target,
            "task": task,
            "model_name": model_name,
            "balancing": balancing_config,
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
        ):
            widget.setEnabled(enabled)

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
