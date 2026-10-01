# ui/train_tab.py
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFormLayout, QComboBox, QPushButton,
    QLabel, QMessageBox, QProgressBar,
)
from core import model_trainer

class TrainTab(QWidget):
    model_trained = Signal()

    def __init__(self, state):
        super().__init__()
        self.state = state
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
        layout.addLayout(form)

        self.cmb_task.currentTextChanged.connect(self._refresh_models)
        self._refresh_models("classification")

        self.btn_train = QPushButton("Entrenar modelo")
        self.btn_train.clicked.connect(self.train)
        layout.addWidget(self.btn_train)

        self.lbl_result = QLabel("")
        layout.addWidget(self.lbl_result)

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

    def train(self):
        if self.state.clean_df is None or self.cmb_target.count() == 0:
            QMessageBox.warning(self, "Sin datos", "Carga y preprocesa datos primero.")
            return
        target = self.cmb_target.currentText()
        task = self.cmb_task.currentText()
        model_name = self.cmb_model.currentText()

        try:
            pipe, metrics, (X_test, y_test, y_pred) = model_trainer.train_model(
                self.state.clean_df, target, model_name, task
            )
        except Exception as e:
            QMessageBox.critical(self, "Error al entrenar", str(e))
            return

        self.state.pipeline = pipe
        self.state.trained_model = pipe
        self.state.metrics = metrics
        self.state.task_type = task
        self.state.target_column = target
        self.state._test_data = (X_test, y_test, y_pred)

        self.lbl_result.setText(
            " · ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
        )
        self.model_trained.emit()