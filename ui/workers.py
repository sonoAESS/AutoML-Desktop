# ui/workers.py  (archivo nuevo)
from PySide6.QtCore import QObject, QThread, Signal

from core import model_trainer


class TuneWorker(QObject):
    """Ejecuta tune_model en un hilo aparte y reporta resultados."""

    finished = Signal(object, dict, tuple, object)   # pipe, metrics, test_data, cv_results
    failed = Signal(str)
    progress = Signal(str)                            # mensajes de estado

    def __init__(self, df, target, model_name, task_type,
                 search_type, metric, cv, n_iter,
                 selections=None, casts=None, normalizations=None):
        super().__init__()
        self._kwargs = dict(
            df=df, target=target, model_name=model_name,
            task_type=task_type, search_type=search_type,
            metric=metric, cv=cv, n_iter=n_iter,
            selections=selections, casts=casts, normalizations=normalizations,
        )

    def run(self):
        try:
            self.progress.emit("Entrenando y buscando hiperparámetros…")
            pipe, metrics, test_data, cv_results = model_trainer.tune_model(
                **self._kwargs
            )
            self.finished.emit(pipe, metrics, test_data, cv_results)
        except Exception as e:
            self.failed.emit(str(e))


class WorkerThread(QThread):
    """QThread reutilizable que ejecuta un TuneWorker."""

    def __init__(self, worker: TuneWorker, parent=None):
        super().__init__(parent)
        self._worker = worker
        worker.moveToThread(self)
        self.started.connect(worker.run)

    @property
    def worker(self) -> TuneWorker:
        return self._worker