# SDD-005 — Tareas de implementación

Estado: **completada**. Depende de 002, 003 y 004, ya implementadas.

### [x] T-001 · `roc_auc_value` público en `model_trainer`
**Ficheros:** `core/model_trainer.py`
**Diseño §2.3**

1. Extraer la lógica de AUC de `_compute_metrics` a
   `roc_auc_value(pipe, X_test, y_test) -> (float | None, motivo | None)`.
2. `_compute_metrics` la usa; el comportamiento no cambia (tests de 003 verdes).
**Verificación:** `pytest -q tests/test_model_trainer.py`.

### [x] T-002 · Bloques pareados y puntuación por bloque
**Ficheros:** `core/comparison.py` (nuevo)
**Requisitos:** R-002, R-004, R-005, R-006, R-014

1. `X, y = df.drop(columns=[target]), df[target]`.
2. Recorte de `n_splits` al tamaño de la clase minoritaria.
3. `bloques = list(cv.split(X, y))` una sola vez, reutilizados por todos.
4. Por cada modelo: `build_pipeline(...)`, `clone` por bloque, `fit`, `predict`,
   puntuación de la métrica.
5. `NaN` en un bloque → modelo excluido con motivo en español; `< 3` modelos → `ValueError`.
6. `< 3` modelos pedidos → `ValueError` en español (R-003).
**Verificación:** C-001, C-005, C-007, C-008.

### [x] T-003 · Rangos y Friedman
**Ficheros:** `core/comparison.py`
**Requisitos:** R-008, R-011

1. Orientación por signo (`rmse`/`mae` negativos).
2. `rankdata` por bloque, mejor = 1.
3. `stats.friedmanchisquare` sobre las columnas de rangos.
4. `ranking` con media, desviación, rango medio y `equivalente`.
**Verificación:** C-002, C-003, C-006.

### [x] T-004 · Post-hoc Wilcoxon + Holm
**Ficheros:** `core/comparison.py`
**Requisitos:** R-009, R-010

1. `wilcoxon` por par; `ttest_rel` si `b < 6` (columna `test`).
2. `_holm(pvalues)` implementado sin `statsmodels`, con test propio de la
   monotonía y del tope en 1.0.
3. `critical_difference(k, b, alpha)` con `q = sqrt(2) * t.ppf(...)`.
**Verificación:** C-003 + test de `_holm` con valores conocidos.

### [x] T-005 · `ModelComparison` y resumen
**Ficheros:** `core/comparison.py`
**Requisitos:** R-007, R-014

1. Dataclass con todos los campos de la spec y `excluidos`.
2. `progress_callback(etapa, actual, total)` con total = `k * b`.
3. Resumen en español reutilizable por la UI (`describe()` opcional del
   dataclass con el χ², el p-valor y la frase interpretativa de R-018).
**Verificación:** test de que el progreso llega de 0 a `total`.

### [x] T-006 · Tests de `comparison`
**Ficheros:** `tests/test_comparison.py` (nuevo)
**Criterios:** C-001 a C-008

Dataset sintético de 200 filas, 3 modelos rápidos, `n_splits=4`,
`n_repeats=2`. Marcados con `@pytest.mark.slow` si tardan más de 2 s en total.
**Verificación:** `pytest -q tests/test_comparison.py` verde.

### [x] T-007 · `ComparisonWorker`
**Ficheros:** `ui/workers.py`
**Requisitos:** R-019

1. `ComparisonWorker(QObject)` con `finished(object)`, `failed(str)`,
   `progress(str, int, int)` y `run()` que delega en `compare_models`.
2. Reutiliza `WorkerThread`.
**Verificación:** test headless con `QCoreApplication`.

### [x] T-008 · Grupo de comparativa en `ResultsTab`
**Ficheros:** `ui/results_tab.py`
**Requisitos:** R-015…R-018, R-020

1. `gb_comparativa` con los widgets de la tabla del diseño §3.1.
2. `_draw_cd_diagram(ax, comparison)` con el segundo canvas.
3. Tabla con las 5 columnas de `ranking`; `excluidos` en una línea de texto.
4. Resumen con χ², p-valor y frase interpretativa.
5. Botón deshabilitado mientras no haya `task_type` y `export_df`.
6. Bloqueo de controles durante la comparación, liberación en éxito y fallo.
**Verificación:** test headless que inyecta un `ModelComparison` construido a mano y comprueba que la tabla tiene 3 filas y el diagrama se dibuja.

### [x] T-009 · Estado y caché
**Ficheros:** `core/state.py`
**Requisitos:** R-020 (parcial)

1. `comparison` y `comparison_key`, con `reset_data()`/`reset_model()`.
2. Aviso en la UI cuando la clave actual difiere de la guardada.
**Verificación:** test de `reset_*`.

### [x] T-010 · `README.md`
**Ficheros:** `README.md`

1. Sección “Comparar modelos (test de Friedman)” con: qué mide, cómo se leen
   los rangos, qué es la diferencia crítica y el post-hoc con Holm.
2. Aviso de tiempo de ejecución y recomendación de empezar con 2 repeticiones.

## Verificación final de la spec

```bash
python -m pytest -q
grep -R "PySide6\|PyQt\|from ui\." core/   # vacío
```

- [x] C-001…C-011 cubiertos.
- [x] La comparación se ejecuta fuera del hilo principal (AGENTS §5.3).