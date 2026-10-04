# SDD-003 — Tareas de implementación

Estado: **esperando puerta 2**.

### [x] T-001 · `METRIC_LABELS`, `METRIC_KEYS`, `metric_label`, `metrics_for_ranking`
**Ficheros:** `core/model_specs.py`
**Requisitos:** R-009, R-010, R-011

1. Añadir los cuatro símbolos con las etiquetas en español de la spec.
2. Test de que `METRIC_KEYS` y `METRICS` no se contradicen y de que toda clave
   de `METRIC_KEYS` tiene etiqueta (C-008).
**Verificación:** `pytest -q tests/test_model_specs.py` verde.

### [x] T-002 · `available_metrics` sin filtro por modelo
**Ficheros:** `core/model_specs.py`
**Requisitos:** R-009

1. Mantener la firma `(task_type, model_name=None)` y devolver `list(METRICS.get(task_type, ()))`.
2. Añadir comentario explicando que `model_name` se conserva por compatibilidad
   y que `supports_probability` queda como dato informativo.
**Verificación:** C-005 y C-006.

### [x] T-003 · Reescribir `_compute_metrics` (clasificación)
**Ficheros:** `core/model_trainer.py`
**Requisitos:** R-001…R-006

1. Clasificación: `accuracy`, `f1_macro`, `roc_auc`, `auc_disponible`,
   `auc_motivo`, `n_test`.
2. Extraer `_auc_scores(pipe, X_test, y_test, classes) -> (scores, es_matriz, motivo)`.
3. `_compute_roc_auc(pipe, X_test, y_test) -> (float | None, str | None)`.
4. Binario con `proba[:, 1]`; multiclase con `multi_class="ovr"`, `average="macro"`,
   `labels=classes`.
5. Fallback a `decision_function`.
6. Motivos del §2.2 del diseño, comprobados antes de llamar a sklearn.
7. Regresión: añadir `n_test` sin cambiar los cálculos.
**Verificación:** C-001 a C-004.

### [x] T-004 · `summarize_metrics`
**Ficheros:** `core/model_trainer.py`
**Diseño §4**

1. Solo claves de `METRIC_KEYS[task_type]` presentes en el diccionario.
2. `None` → “n/d”; `best_params` fuera; claves desconocidas fuera.
**Verificación:** test con `{"roc_auc": None, "auc_disponible": False, "best_params": {...}}`.

### [x] T-005 · `tune_model` pasa `pipe` y `X_test`
**Ficheros:** `core/model_trainer.py`
**Requisitos:** R-001 (rama sin grid)

1. Sustituir la llamada de la línea ~310 por
   `_compute_metrics(task_type, y_test, y_pred, pipe=pipe, X_test=X_test)`.
**Verificación:** test de un modelo sin grid (todos tienen grid; usar un modelo
ficticio vía `monkeypatch` de `search_grid`) que devuelve las tres métricas.

### [x] T-006 · `TrainTab` formatea con `summarize_metrics`
**Ficheros:** `ui/train_tab.py`
**Diseño §5.2**

1. Sustituir el `isinstance(v, (int, float))` de `_store_result`.
2. Mantener la línea de `best_params` aparte.
**Verificación:** test headless que entrena y comprueba que la etiqueta no
contiene `auc_disponible`.

### [x] T-007 · `ResultsTab` con tabla de métricas
**Ficheros:** `ui/results_tab.py`
**Requisitos:** R-012, R-014

1. `tbl_metricas` de 2 columnas construida desde `METRIC_KEYS`/`METRIC_LABELS`.
2. `None` → `n/d` gris con tooltip = `auc_motivo`.
3. `lbl` para el resumen de la búsqueda y el aviso no bloqueante del AUC.
4. La figura (matriz de confusión o CV) se mantiene igual.
**Verificación:** test headless con métricas de clasificación que incluyen
`roc_auc=None`: la tabla muestra `n/d` y el tooltip tiene el motivo.

## Verificación final de la spec

```bash
python -m pytest -q
grep -R "PySide6\|PyQt\|from ui\." core/   # vacío
```

- [x] C-001…C-010 cubiertos.
- [x] Los tests anteriores de métricas siguen verdes sin reescribirlos.
- [x] `README.md` actualizado con la tabla de métricas y la nota de `n/d`.