# ui/workers.py  (archivo nuevo)
from PySide6.QtCore import QObject, QThread, Signal

from core import comparison, model_trainer


class TuneWorker(QObject):
    """Ejecuta tune_model en un hilo aparte y reporta resultados.

    Las señales cruzan de este hilo al de la interfaz de forma encolada, así
    que `progress` puede invocarse desde dentro de `tune_model` sin cuidado.
    """

    finished = Signal(
        object, dict, tuple, object
    )  # pipe, metrics, test_data, cv_results
    failed = Signal(str)
    # Etapa actual, ajustes completados y ajustes totales (0 = sin total).
    progress = Signal(str, int, int)

    def __init__(
        self,
        df,
        target,
        model_name,
        task_type,
        search_type,
        metric,
        cv,
        n_iter,
        selections=None,
        casts=None,
        normalizations=None,
        balancing=None,
        selection=None,
    ):
        super().__init__()
        self._kwargs = dict(
            df=df,
            target=target,
            model_name=model_name,
            task_type=task_type,
            search_type=search_type,
            metric=metric,
            cv=cv,
            n_iter=n_iter,
            selections=selections,
            casts=casts,
            normalizations=normalizations,
            balancing=balancing,
            selection=selection,
        )

    def run(self):
        try:
            self.progress.emit(
                "Explorando hiperparámetros…",
                0,
                0,
            )
            pipe, metrics, test_data, cv_results = model_trainer.tune_model(
                progress_callback=self._reportar,
                **self._kwargs,
            )
            self.finished.emit(pipe, metrics, test_data, cv_results)
        except Exception as e:
            self.failed.emit(str(e))

    def _reportar(self, etapa, actual, total):
        """Callback que usa `tune_model` para ir informando."""
        self.progress.emit(etapa, actual, total)


class WorkerThread(QThread):
    """QThread que ejecuta un `TuneWorker` una vez y termina.

    Se sobrescribe `run()` a propósito: si no, la implementación por defecto
    de `QThread` abre un event loop y el hilo nunca termina, con lo que la
    señal `finished` no llegaría nunca y la interfaz se quedaría bloqueada
    aunque el modelo ya esté entrenado.
    """

    def __init__(self, worker: TuneWorker, parent=None):
        super().__init__(parent)
        self._worker = worker
        worker.moveToThread(self)

    def run(self):
        self._worker.run()

    @property
    def worker(self) -> TuneWorker:
        return self._worker


class SelectionWorker(QObject):
    """Ejecuta el análisis de selección en un hilo aparte.

    Ajustar el selector puede tardar (el método embebido entrena un bosque por
    dentro), así que no puede bloquear la interfaz. Solo orquesta: la lógica
    está en `core.model_trainer.analyze_selection`.
    """

    finished = Signal(object)  # DataFrame de la selección
    failed = Signal(str)

    def __init__(
        self,
        df,
        target,
        model_name,
        task_type,
        selection=None,
        casts=None,
        normalizations=None,
    ):
        super().__init__()
        self._kwargs = dict(
            df=df,
            target=target,
            model_name=model_name,
            task_type=task_type,
            selection=selection,
            casts=casts,
            normalizations=normalizations,
        )

    def run(self):
        try:
            tabla = model_trainer.analyze_selection(**self._kwargs)
            self.finished.emit(tabla)
        except Exception as e:  # noqa: BLE001 - el mensaje va a la interfaz
            self.failed.emit(str(e))


class ComparisonWorker(QObject):
    """Ejecuta la comparativa de modelos en un hilo aparte.

    Ajustar `modelos × bloques` pipelines es de largo por definición, así que va
    siempre en segundo plano. Solo orquesta: el análisis vive en
    `core.comparison.compare_models`.
    """

    finished = Signal(object)  # ModelComparison
    failed = Signal(str)
    progress = Signal(str, int, int)  # etapa, hecho, total

    def __init__(
        self,
        df,
        target,
        model_names,
        task_type,
        metric,
        casts=None,
        normalizations=None,
        selection=None,
        balancing=None,
        n_splits=5,
        n_repeats=3,
        random_state=42,
    ):
        super().__init__()
        self._kwargs = dict(
            df=df,
            target=target,
            model_names=model_names,
            task_type=task_type,
            metric=metric,
            casts=casts,
            normalizations=normalizations,
            selection=selection,
            balancing=balancing,
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=random_state,
        )

    def _reportar(self, hecho, total):
        self.progress.emit(f"Comparando modelos: {hecho}/{total}", hecho, total)

    def run(self):
        try:
            self.progress.emit("Preparando los folds…", 0, 0)
            resultado = comparison.compare_models(
                progress=self._reportar, **self._kwargs
            )
            self.finished.emit(resultado)
        except Exception as e:  # noqa: BLE001 - el mensaje va a la interfaz
            self.failed.emit(str(e))
