# ui/train_tab.py  (reemplaza el archivo)
from PySide6.QtCore import Signal, QThread
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QComboBox, QPushButton,
    QLabel, QMessageBox, QCheckBox, QSpinBox, QProgressBar,
)
from core import model_trainer
from ui.workers import TuneWorker, WorkerThread


class TrainTab(QWidget):
    model_trained = Signal()

    def __init__(self, state):
        super().__init__()
        self.state = state
        self._thread: WorkerThread | None = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.cmb_target = QComboBox()
        self.cmb_task = QComboBox()
        self.cmb_task.addItems(["classification", "regression"])
        self.cmb_model = QComboBox()

        form.addRow("Variable objetivo:", self.cmb_target)
        form.addRow("Tipo de tarea:", self.cmb_task)
        form.addRow("Modelo:", self.cmb_model)

        # --- Opciones de búsqueda de hiperparámetros ---
        self.chk_tune = QCheckBox("Buscar hiperparámetros")
        self.cmb_search = QComboBox()
        self.cmb_search.addItems(["grid", "random"])
        self.spn_cv = QSpinBox()
        self.spn_cv.setRange(2, 10)
        self.spn_cv.setValue(5)
        self.spn_iter = QSpinBox()
        self.spn_iter.setRange(5, 200)
        self.spn_iter.setValue(20)
        self.cmb_metric = QComboBox()

        form.addRow(self.chk_tune)
        form.addRow("Estrategia:", self.cmb_search)
        form.addRow("Folds (CV):", self.spn_cv)
        form.addRow("Iteraciones (random):", self.spn_iter)
        form.addRow("Métrica a optimizar:", self.cmb_metric)
        layout.addLayout(form)

        self.btn_train = QPushButton("Entrenar modelo")
        self.btn_train.clicked.connect(self.train)
        layout.addWidget(self.btn_train)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)   # indeterminado
        self.progress.hide()
        layout.addWidget(self.progress)

        self.lbl_result = QLabel("")
        self.lbl_result.setWordWrap(True)
        layout.addWidget(self.lbl_result)

        self.cmb_task.currentTextChanged.connect(self._refresh_models)
        self.cmb_task.currentTextChanged.connect(self._refresh_metrics)
        self.chk_tune.toggled.connect(self._toggle_tune_options)
        self._refresh_models("classification")
        self._refresh_metrics("classification")
        self._toggle_tune_options(False)

    # ------------------------------------------------------------------
    def refresh(self):
        self.cmb_target.clear()
        if self.state.clean_df is None:
            return
        self.cmb_target.addItems(list(self.state.clean_df.columns))

    def _refresh_models(self, task):
        self.cmb_model.clear()
        models = (model_trainer.CLASSIFIERS if task == "classification"
                  else model_trainer.REGRESSORS)
        self.cmb_model.addItems(list(models.keys()))

    def _refresh_metrics(self, task):
        self.cmb_metric.clear()
        if task == "classification":
            self.cmb_metric.addItems(["accuracy", "f1_macro", "roc_auc"])
        else:
            self.cmb_metric.addItems(["rmse", "mae", "r2"])

    def _toggle_tune_options(self, enabled):
        for w in (self.cmb_search, self.spn_cv, self.spn_iter, self.cmb_metric):
            w.setEnabled(enabled)

    # ------------------------------------------------------------------
    def train(self):
        if self.state.clean_df is None or self.cmb_target.count() == 0:
            QMessageBox.warning(self, "Sin datos", "Carga y preprocesa datos primero.")
            return
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(self, "En curso", "Ya hay un entrenamiento activo.")
            return

        target = self.cmb_target.currentText()
        task = self.cmb_task.currentText()
        model_name = self.cmb_model.currentText()

        if self.chk_tune.isChecked():
            worker = TuneWorker(
                df=self.state.clean_df,
                target=target,
                model_name=model_name,
                task_type=task,
                search_type=self.cmb_search.currentText(),
                metric=self.cmb_metric.currentText(),
                cv=self.spn_cv.value(),
                n_iter=self.spn_iter.value(),
            )
            self._thread = WorkerThread(worker, self)
            worker.finished.connect(self._on_finished)
            worker.failed.connect(self._on_failed)
            worker.progress.connect(self.lbl_result.setText)
            self._thread.finished.connect(self._on_thread_done)
            self._set_busy(True)
            self._thread.start()
        else:
            # Entrenamiento normal (sigue siendo síncrono o muévelo también
            # a un hilo si prefieres uniformidad; aquí lo dejamos directo).
            try:
                pipe, metrics, test_data = model_trainer.train_model(
                    self.state.clean_df, target, model_name, task
                )
            except Exception as e:
                QMessageBox.critical(self, "Error al entrenar", str(e))
                return
            self._store_result(pipe, metrics, test_data, {})

    # ------------------------------------------------------------------
    def _on_finished(self, pipe, metrics, test_data, cv_results):
        self._store_result(pipe, metrics, test_data, cv_results)

    def _on_failed(self, message):
        QMessageBox.critical(self, "Error al buscar hiperparámetros", message)
        self._set_busy(False)

    def _on_thread_done(self):
        self._set_busy(False)
        self._thread = None

    def _set_busy(self, busy: bool):
        self.btn_train.setEnabled(not busy)
        self.progress.setVisible(busy)

    def _store_result(self, pipe, metrics, test_data, cv_results):
        self.state.pipeline = pipe
        self.state.trained_model = pipe
        self.state.metrics = metrics
        self.state.task_type = self.cmb_task.currentText()
        self.state.target_column = self.cmb_target.currentText()
        self.state._test_data = test_data
        self.state._cv_results = cv_results

        lines = [f"{k}: {v:.4f}" for k, v in metrics.items()
                 if isinstance(v, (int, float))]
        if "best_params" in metrics:
            lines.append(f"mejores params: {metrics['best_params']}")
        self.lbl_result.setText(" · ".join(lines))
        self.model_trained.emit()