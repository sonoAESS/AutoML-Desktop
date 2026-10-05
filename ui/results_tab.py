# ui/results_tab.py
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from sklearn.metrics import ConfusionMatrixDisplay

from core import model_specs, model_trainer, persistence
from ui.empty_state import EmptyState, alternar_estado_vacio
from ui.enabled import con_pista
from ui.layout import acotar_tabla, expandir_canvas
from ui.theme import marcar
from ui.workers import ComparisonWorker, WorkerThread


def _family_of(model_name):
    """Familia de un modelo, o `None` si no está en el catálogo."""
    from core import model_specs as specs

    for task in (specs.CLASSIFICATION, specs.REGRESSION):
        try:
            return specs.get_spec(model_name, task).family
        except KeyError:
            continue
    return None


def _first_metric(metrics, metricas):
    """Primera métrica disponible que tenga valor en el diccionario."""
    for clave in metricas:
        if (metrics or {}).get(clave) is not None:
            return clave
    return metricas[0] if metricas else ""


def _format_metric(valor) -> str:
    """Valor de una métrica para la tabla: número, texto o `n/d`."""
    if valor is None:
        return "n/d"
    if isinstance(valor, bool):
        return "sí" if valor else "no"
    if isinstance(valor, (int, float)):
        return f"{valor:.4f}"
    return str(valor)


class ResultsTab(QWidget):
    model_saved = Signal(str)
    #: Pide cambiar de paso. La ventana decide a dónde ir: una pestaña no
    #: puede tocar el `QTabWidget` del padre (AGENTS §5.1).
    navigate_requested = Signal(str)

    def __init__(self, state):
        super().__init__()
        self.state = state
        self._hilo_comparativa = None
        self._build_ui()

    def _build_ui(self):
        """Tres sub-pestañas, porque apiladas no cabían en una pantalla.

        Métricas y gráficos medían ~330 px de lienzo más la tabla; la
        comparativa sumaba otros ~340. Apiladas daban más de 1100 px, así que
        solo se veía el primer gráfico.
        """
        layout = QVBoxLayout(self)
        self.subpestanas = QTabWidget()
        layout.addWidget(self.subpestanas)

        # --- Métricas ---
        pagina_metricas = QWidget()
        v_metricas = QVBoxLayout(pagina_metricas)
        v_metricas.setContentsMargins(0, 0, 0, 0)

        self.lbl = QLabel("Sin resultados todavía.")
        self.lbl.setWordWrap(True)
        v_metricas.addWidget(self.lbl)

        self.vacio = EmptyState(
            "Todavía no hay un modelo entrenado",
            "Aquí aparecerán las métricas del modelo, la curva ROC, la matriz "
            "de confusión y la importancia de cada variable.",
            "Entrena un modelo en el paso 3",
        )
        v_metricas.addWidget(self.vacio, 1)

        self.tbl_metricas = QTableWidget(0, 2)
        self.tbl_metricas.setHorizontalHeaderLabels(["Métrica", "Valor"])
        self.tbl_metricas.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        self.tbl_metricas.verticalHeader().setVisible(False)
        acotar_tabla(self.tbl_metricas, 140)
        v_metricas.addWidget(self.tbl_metricas)

        self.figure = Figure(figsize=(5, 4))
        self.canvas = FigureCanvasQTAgg(self.figure)
        expandir_canvas(self.canvas)
        v_metricas.addWidget(self.canvas, 1)

        self.subpestanas.addTab(pagina_metricas, "Métricas")

        # --- Comparativa ---
        pagina_comparativa = QWidget()
        v_comparativa = QVBoxLayout(pagina_comparativa)
        v_comparativa.setContentsMargins(0, 0, 0, 0)
        self._pagina_comparativa = pagina_comparativa

        # Se construye el grupo ya escondido y se oculta la página entera
        # después de añadirla: reparentar borra el estado de visibilidad.
        v_comparativa.addWidget(self._build_comparison_group())
        v_comparativa.addStretch()
        self.subpestanas.addTab(pagina_comparativa, "Comparativa")
        self.pagina_comparativa = self.subpestanas.indexOf(pagina_comparativa)
        # Sin datos no hay nada que comparar: la página se oculta desde el
        # principio, no solo cuando `refresh()` decidelo.
        pagina_comparativa.setVisible(False)

        # --- Exportar ---
        pagina_exportar = QWidget()
        v_exportar = QVBoxLayout(pagina_exportar)
        v_exportar.setContentsMargins(0, 0, 0, 0)

        opciones = QHBoxLayout()
        self.chk_dataset = QCheckBox("Incluir el dataset transformado")
        self.chk_dataset.setChecked(True)
        self.chk_dataset.setToolTip(
            "Guarda también el dataset ya limpiado y normalizado, en el mismo "
            "archivo que el modelo."
        )
        opciones.addWidget(self.chk_dataset)
        opciones.addStretch()
        v_exportar.addLayout(opciones)

        self.btn_save = QPushButton("Guardar proyecto")
        self.btn_save.setToolTip(
            "Guarda el modelo entrenado, el dataset transformado y los "
            "metadatos en un único archivo .automl."
        )
        marcar(self.btn_save, "primario")
        self.btn_save.clicked.connect(self.save_bundle)
        self.btn_export = QPushButton("Exportar dataset")
        self.btn_export.setToolTip(
            "Exporta solo el dataset ya limpio, en CSV, sin el modelo."
        )
        self.btn_export.clicked.connect(self.export_dataset)
        self.btn_load_in_predict = QPushButton("Ir a Predicción")
        self.btn_load_in_predict.clicked.connect(
            lambda: self.navigate_requested.emit("prediccion")
        )
        v_exportar.addWidget(self.btn_save)
        v_exportar.addWidget(self.btn_export)
        v_exportar.addWidget(self.btn_load_in_predict)
        v_exportar.addStretch()
        self.subpestanas.addTab(pagina_exportar, "Exportar")

        self.gb_comparativa.setVisible(False)

        # La señal se conecta al final: `addTab` emite `currentChanged` y el
        # manejador necesita `pagina_comparativa`, que aún no existe.
        self.subpestanas.currentChanged.connect(self._al_cambiar_subpestana)

        self.lbl_saved = QLabel("")
        self.lbl_saved.setWordWrap(True)
        layout.addWidget(self.lbl_saved)

        # Estado inicial: sin modelo no se puede guardar nada.
        self._update_enabled_state()
        self._alternar_vacio()

    def _build_comparison_group(self):
        """Grupo «Comparativa de modelos (Friedman)», oculto sin datos."""
        gb = QGroupBox("Comparativa de modelos (Friedman)")
        v = QVBoxLayout(gb)

        opciones = QHBoxLayout()
        self.cmb_metrica_comparativa = QComboBox()
        self.spn_repeticiones = QSpinBox()
        self.spn_repeticiones.setRange(1, 10)
        self.spn_repeticiones.setValue(3)
        self.spn_repeticiones.setToolTip(
            "Cada modelo se ajusta una vez por fold y por repetición. Con 3 "
            "repeticiones de 5 folds son 15 ajustes por modelo: puede tardar."
        )
        self.chk_todos_los_modelos = QCheckBox("Todos los modelos compatibles")
        self.chk_todos_los_modelos.setToolTip(
            "Compara todos los modelos que el dataset admite. Si lo "
            "desmarcas, solo se compara la familia del modelo actual."
        )
        self.chk_todos_los_modelos.setChecked(True)
        self.chk_todos_los_modelos.toggled.connect(self._on_alcance_comparativa)
        opciones.addWidget(QLabel("Métrica:"))
        opciones.addWidget(self.cmb_metrica_comparativa)
        opciones.addWidget(QLabel("Repeticiones:"))
        opciones.addWidget(self.spn_repeticiones)
        opciones.addWidget(self.chk_todos_los_modelos)
        opciones.addStretch()
        v.addLayout(opciones)

        self.btn_comparar = QPushButton("Comparar modelos")
        self.btn_comparar.setToolTip(
            "Evalúa todos los modelos compatibles sobre los mismos folds y "
            "calcula el test de Friedman para ver si las diferencias son "
            "reales."
        )
        marcar(self.btn_comparar, "primario")
        self.btn_comparar.clicked.connect(self.comparar_modelos)
        self.btn_comparar.setEnabled(False)
        v.addWidget(self.btn_comparar)

        self.progress_comparativa = QProgressBar()
        self.progress_comparativa.setRange(0, 0)
        self.progress_comparativa.setFormat("Comparando…")
        self.progress_comparativa.hide()
        v.addWidget(self.progress_comparativa)

        self.tbl_comparativa = QTableWidget(0, 5)
        self.tbl_comparativa.setHorizontalHeaderLabels(
            ["Modelo", "Puntuación media", "Desviación", "Rango medio", "Equivalente"]
        )
        self.tbl_comparativa.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )
        self.tbl_comparativa.verticalHeader().setVisible(False)
        acotar_tabla(self.tbl_comparativa, 160)
        v.addWidget(self.tbl_comparativa)

        self.lbl_excluidos = QLabel("")
        self.lbl_excluidos.setWordWrap(True)
        v.addWidget(self.lbl_excluidos)

        self.lbl_comparativa = QLabel("")
        self.lbl_comparativa.setWordWrap(True)
        v.addWidget(self.lbl_comparativa)

        # El diagrama va en su propio canvas: la figura de métricas ya está
        # ocupada y compartir eje lo haría ilegible.
        self.figure_cd = Figure(figsize=(5, 3))
        self.canvas_cd = FigureCanvasQTAgg(self.figure_cd)
        expandir_canvas(self.canvas_cd)
        v.addWidget(self.canvas_cd)

        self.gb_comparativa = gb
        return gb

    # ------------------------------------------------------------------
    # Comparativa de modelos (Friedman)
    # ------------------------------------------------------------------
    def _refresh_comparativa_disponibilidad(self):
        """La comparativa solo aparece con tipo de tarea y dataset transformado.

        Se oculta la **página** entera y no solo el grupo: dentro de una
        sub-pestaña vacía el usuario ve un hueco sin explicación.
        """
        disponible = bool(self.state.task_type) and self.state.export_df is not None
        self.gb_comparativa.setVisible(disponible)
        self._pagina_comparativa.setVisible(disponible)
        if not disponible:
            self.btn_comparar.setEnabled(False)
            # Si se estaba viendo esa página, hay que salir de ella.
            if self.subpestanas.currentWidget() is self._pagina_comparativa:
                self.subpestanas.setCurrentIndex(0)
            return

        metricas = model_specs.available_metrics(self.state.task_type)
        self.cmb_metrica_comparativa.blockSignals(True)
        self.cmb_metrica_comparativa.clear()
        self.cmb_metrica_comparativa.addItems(metricas)
        indice = self.cmb_metrica_comparativa.findText(
            _first_metric(self.state.metrics, metricas)
        )
        if indice >= 0:
            self.cmb_metrica_comparativa.setCurrentIndex(indice)
        self.cmb_metrica_comparativa.blockSignals(False)
        self.btn_comparar.setEnabled(True)
        self._avisar_comparativa_desfasada()

    def _modelos_compatibles(self):
        """Perfiles de las columnas predictoras del dataset transformado."""
        from core import profiling

        df = self.state.export_df
        nombres = [c for c in df.columns if c != self.state.target_column]
        return [profiling.profile_column(df[c]) for c in nombres]

    def _modelos_a_comparar(self):
        """Modelos según el alcance elegido.

        Con la casilla marcada se comparan todos los compatibles. Sin ella, solo
        los de la familia del modelo actual: el test necesita tres algoritmos y
        un único modelo no llega a ese mínimo.
        """
        resultados = model_trainer.compatible_models(
            self.state.task_type,
            self._modelos_compatibles(),
            target=self.state.target_column,
            n_rows=len(self.state.export_df),
        )
        compatibles = [item.name for item in resultados if item.compatible]
        if self.chk_todos_los_modelos.isChecked():
            return compatibles

        familia = self.state.family
        if not familia:
            return [self.state.model_name] if self.state.model_name else compatibles
        return [nombre for nombre in compatibles if _family_of(nombre) == familia]

    def _on_alcance_comparativa(self, _todos):
        """El alcance solo cambia cuando ya hay una comparativa en pantalla."""
        if self.state.comparison is not None:
            self.lbl_comparativa.setText(
                "Has cambiado el alcance: vuelve a comparar para actualizar la tabla."
            )

    def comparar_modelos(self):
        if self.state.export_df is None or not self.state.task_type:
            QMessageBox.warning(self, "Sin datos", "Preprocesa los datos primero.")
            return
        if self._hilo_comparativa is not None and self._hilo_comparativa.isRunning():
            QMessageBox.information(self, "En curso", "Ya hay una comparación activa.")
            return

        modelos = self._modelos_a_comparar()
        if len(modelos) < 3:
            QMessageBox.warning(
                self,
                "Modelos insuficientes",
                "El test de Friedman necesita al menos 3 modelos. Con el alcance "
                f"elegido solo hay {len(modelos)}; vuelve a marcar «Todos los "
                "modelos compatibles».",
            )
            return

        worker = ComparisonWorker(
            df=self.state.export_df,
            target=self.state.target_column,
            model_names=modelos,
            task_type=self.state.task_type,
            metric=self.cmb_metrica_comparativa.currentText(),
            casts=self.state.column_types or None,
            normalizations=self.state.normalizations or None,
            balancing=(
                {"method": self.state.balancing} if self.state.balancing else None
            ),
            n_repeats=self.spn_repeticiones.value(),
        )
        self._hilo_comparativa = WorkerThread(worker, self)
        worker.finished.connect(self._on_comparacion_lista)
        worker.failed.connect(self._on_comparacion_fallida)
        worker.progress.connect(self._on_comparacion_progreso)
        self._set_busy_comparativa(True)
        self._hilo_comparativa.start()

    def _set_busy_comparativa(self, busy):
        self.btn_comparar.setEnabled(not busy)
        self.cmb_metrica_comparativa.setEnabled(not busy)
        self.spn_repeticiones.setEnabled(not busy)
        self.chk_todos_los_modelos.setEnabled(not busy)
        self.progress_comparativa.setVisible(busy)

    def _on_comparacion_progreso(self, etapa, hecho, total):
        if total:
            self.progress_comparativa.setRange(0, total)
            self.progress_comparativa.setValue(hecho)
            self.progress_comparativa.setFormat("%p%: " + etapa)
        else:
            self.progress_comparativa.setRange(0, 0)

    def _al_cambiar_subpestana(self, indice: int):
        """Refresca la disponibilidad al cambiar de sub-pestaña."""
        if indice == self.pagina_comparativa:
            self._refresh_comparativa_disponibilidad()

    def _on_comparacion_lista(self, resultado):
        self.state.comparison = resultado
        self.state.comparison_key = self._comparison_key()
        self._set_busy_comparativa(False)
        self._show_comparison(resultado)
        # La comparativa tardó: el usuario debe acabar viendo para qué la pidió.
        self.subpestanas.setCurrentIndex(self.pagina_comparativa)

    def _on_comparacion_fallida(self, mensaje):
        self._set_busy_comparativa(False)
        QMessageBox.critical(self, "Error al comparar modelos", mensaje)

    def _comparison_key(self):
        """Identifica la configuración con la que se hizo la comparativa."""
        return (
            self.state.target_column,
            self.state.task_type,
            self.cmb_metrica_comparativa.currentText(),
            self.spn_repeticiones.value(),
            self.chk_todos_los_modelos.isChecked(),
        )

    def _avisar_comparativa_desfasada(self):
        """Avisa si la comparativa guardada no corresponde a lo actual."""
        if self.state.comparison is None or not self.state.comparison_key:
            return
        if self._comparison_key() != self.state.comparison_key:
            self.lbl_comparativa.setText(
                "La comparativa guardada se hizo con otra configuración: "
                "vuelve a comparar para actualizarla."
            )

    def _show_comparison(self, resultado):
        self.tbl_comparativa.setRowCount(len(resultado.ranking))
        atenuado = QBrush(QColor(Qt.GlobalColor.gray))
        normal = QBrush(QColor(Qt.GlobalColor.black))
        for fila, datos in enumerate(resultado.ranking.itertuples(index=False)):
            valores = (
                str(datos.modelo),
                f"{datos.puntuacion_media:.4f}",
                f"{datos.desviacion:.4f}",
                f"{datos.rango_medio:.2f}",
                "sí" if datos.equivalente else "no",
            )
            for columna, texto in enumerate(valores):
                self.tbl_comparativa.setItem(fila, columna, QTableWidgetItem(texto))
            # Los que caen fuera de la diferencia crítica se atenúan: no se
            # pueden llamar mejores que el mejor.
            for columna in range(5):
                self.tbl_comparativa.item(fila, columna).setForeground(
                    normal if datos.equivalente else atenuado
                )

        if resultado.excluidos:
            self.lbl_excluidos.setText(
                "Excluidos: "
                + "; ".join(
                    f"{nombre} ({motivo})"
                    for nombre, motivo in resultado.excluidos.items()
                )
            )
        else:
            self.lbl_excluidos.setText("")

        self.lbl_comparativa.setText(resultado.describe())
        self._draw_cd_diagram(resultado)

    def _draw_cd_diagram(self, resultado):
        """Diagrama de diferencias críticas: rangos medios y barra de la CD."""
        self.figure_cd.clear()
        ax = self.figure_cd.add_subplot(111)
        datos = resultado.ranking.sort_values("rango_medio", ascending=False)

        nombres = list(datos["modelo"])
        rangos = list(datos["rango_medio"])
        cd = resultado.critical_difference
        mejor = min(rangos) if rangos else 0.0

        ax.barh(
            range(len(nombres)),
            rangos,
            height=0.6,
            color=["#c7ddf5" if rango <= mejor + cd else "#f0c7c7" for rango in rangos],
        )
        ax.set_yticks(range(len(nombres)))
        ax.set_yticklabels(nombres, fontsize=8)
        ax.set_xlabel("Rango medio (menor es mejor)")
        ax.set_title(
            f"Diferencia crítica = {cd:.2f} "
            f"(Friedman χ²={resultado.friedman_statistic:.2f}, "
            f"p={resultado.friedman_p:.3f})",
            fontsize=9,
        )
        ax.set_xlim(0, max(rangos) if rangos else 1)
        self.figure_cd.tight_layout()
        self.canvas_cd.draw()

    # ------------------------------------------------------------------
    def refresh(self):
        self._update_enabled_state()
        self._refresh_comparativa_disponibilidad()
        self._alternar_vacio()
        if self.state.comparison is not None:
            self._show_comparison(self.state.comparison)
        metrics = self.state.metrics
        if not metrics:
            return
        self._show_metrics(metrics)
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        if not self._plot_cv_results(ax):
            self._plot_test(ax)
        # Sin esto las etiquetas largas (nombres de atributo, clases) se cortan.
        self.figure.tight_layout()
        self.canvas.draw()

    # ------------------------------------------------------------------
    def _alternar_vacio(self):
        """Sin modelo, una tabla de métricas vacía no explica nada."""
        hay_datos = bool(self.state.metrics)
        alternar_estado_vacio(
            self.vacio, (self.lbl, self.tbl_metricas, self.canvas), not hay_datos
        )

    # ------------------------------------------------------------------
    def _update_enabled_state(self):
        """Guardar, exportar e ir a predicción exigen un modelo entrenado.

        Antes los tres botones se podían pulsar sin modelo, y el usuario
        descubría el problema con un `QMessageBox` en vez de con el propio
        aspecto del botón.
        """
        hay_modelo = self.state.pipeline is not None
        sin_modelo = "Entrena un modelo primero para poder guardarlo."
        con_pista(
            self.btn_save,
            hay_modelo,
            (
                "Guarda el modelo, el dataset transformado y los metadatos en un "
                "único archivo .automl."
                if hay_modelo
                else sin_modelo
            ),
        )
        con_pista(
            self.btn_export,
            hay_modelo,
            "Exporta solo el dataset ya limpio, en CSV." if hay_modelo else sin_modelo,
        )
        con_pista(
            self.btn_load_in_predict,
            hay_modelo,
            (
                "Lleva este modelo a la pestaña de Predicción para aplicarlo a "
                "datos nuevos."
                if hay_modelo
                else sin_modelo
            ),
        )
        self.tbl_metricas.setEnabled(hay_modelo)

    def _show_metrics(self, metrics):
        """Tabla Métrica/Valor con `n/d` y motivo cuando algo no existe."""
        claves = model_specs.METRIC_KEYS.get(self.state.task_type, ())
        self.tbl_metricas.setRowCount(len(claves))
        motivo = metrics.get("auc_motivo")

        for fila, clave in enumerate(claves):
            etiqueta = model_specs.metric_label(clave)
            item_clave = QTableWidgetItem(etiqueta)
            item_valor = QTableWidgetItem(_format_metric(metrics.get(clave)))
            if metrics.get(clave) is None:
                item_valor.setForeground(QBrush(QColor(Qt.GlobalColor.gray)))
                if motivo:
                    item_valor.setToolTip(motivo)
            self.tbl_metricas.setItem(fila, 0, item_clave)
            self.tbl_metricas.setItem(fila, 1, item_valor)

        resumen = [
            f"Modelo: {self.state.model_name}",
            (
                f"Evaluado sobre {metrics['n_test']} filas de prueba"
                if "n_test" in metrics
                else ""
            ),
            persistence.balancing_summary(
                self.state.pipeline, {"method": self.state.balancing}
            ),
            persistence.selection_summary(
                self.state.pipeline,
                {"method": self.state.selection_method},
            ),
        ]
        if metrics.get("auc_disponible") is False:
            resumen.append(f"Aviso: {motivo}")
        if isinstance(metrics.get("best_params"), dict):
            resumen.append(f"mejores parámetros: {metrics['best_params']}")
        self.lbl.setText("\n".join(texto for texto in resumen if texto))

    def _plot_test(self, ax):
        if not self.state._test_data:
            return
        X_test, y_test, y_pred = self.state._test_data
        if self.state.task_type == "classification":
            ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
            ax.set_title("Matriz de confusión")
        else:
            ax.scatter(y_test, y_pred, alpha=0.5)
            lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
            ax.plot(lims, lims, "r--")
            ax.set_xlabel("Real")
            ax.set_ylabel("Predicho")
            ax.set_title("Real vs. Predicho")

    def _plot_cv_results(self, ax):
        cv = getattr(self.state, "_cv_results", None)
        if not cv:
            return False
        scores = cv.get("mean_test_score")
        if scores is None:
            return False
        ax.plot(range(1, len(scores) + 1), scores, marker="o")
        ax.set_xlabel("Combinación")
        ax.set_ylabel("Puntuación de la validación cruzada")
        ax.set_title("Resultados de la búsqueda de hiperparámetros")
        return True

    # ------------------------------------------------------------------
    def save_bundle(self):
        if self.state.trained_model is None:
            QMessageBox.warning(self, "Sin modelo", "Entrena un modelo primero.")
            return
        if not self.state.target_column or not self.state.model_name:
            QMessageBox.warning(
                self,
                "Sin modelo",
                "No se sabe qué variable objetivo ni qué modelo se usó.",
            )
            return

        destino, _ = QFileDialog.getSaveFileName(
            self,
            "Guardar modelo y dataset",
            self._default_name(),
            "Modelo AutoML (*.automl)",
        )
        if not destino:
            return

        try:
            metadata = persistence.build_metadata(
                self.state.trained_model,
                target_column=self.state.target_column,
                task_type=self.state.task_type,
                model_name=self.state.model_name,
                metrics=self.state.metrics,
                df=self.state.export_df,
                casts=self.state.column_types or None,
                normalizations=self.state.normalizations or None,
            )
            dataset = self.state.export_df if self.chk_dataset.isChecked() else None
            ruta = persistence.save_bundle(
                destino, self.state.trained_model, metadata, dataset=dataset
            )
        except Exception as e:
            QMessageBox.critical(self, "Error al guardar", str(e))
            return

        self.state.bundle_path = ruta
        self.lbl_saved.setText(
            f"Guardado en {ruta}\n"
            "Puedes abrirlo en la pestaña 5. Predicción para aplicarlo a "
            "nuevos datos."
        )
        self.model_saved.emit(ruta)

    def export_dataset(self):
        if self.state.export_df is None:
            QMessageBox.warning(
                self, "Sin dataset", "Aplica el preprocesamiento primero."
            )
            return
        destino, _ = QFileDialog.getSaveFileName(
            self,
            "Exportar dataset transformado",
            "dataset_preprocesado.csv",
            "CSV (*.csv)",
        )
        if not destino:
            return
        try:
            self.state.export_df.to_csv(destino, index=False)
        except Exception as e:
            QMessageBox.critical(self, "Error al exportar", str(e))
            return
        self.lbl_saved.setText(f"Dataset exportado en {destino}")

    def _default_name(self):
        base = (
            self.state.source_path.rsplit("/", 1)[-1] if self.state.source_path else ""
        )
        nombre = (self.state.model_name or "modelo").replace(" ", "_")
        if base:
            return f"{base.rsplit('.', 1)[0]}_{nombre}{persistence.BUNDLE_SUFFIX}"
        return f"{nombre}{persistence.BUNDLE_SUFFIX}"
