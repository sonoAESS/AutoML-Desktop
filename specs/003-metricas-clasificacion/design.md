# SDD-003 — Diseño técnico: métricas de clasificación

Estado: **implementada** (2026-10-04).

## 1. Mapa de cambios

| Fichero | Cambio |
|---|---|
| `core/model_specs.py` | `METRIC_LABELS`, `metric_label`, `metrics_for_ranking`, `available_metrics` sin filtro por modelo |
| `core/model_trainer.py` | `_compute_metrics` reescrito, `summarize_metrics`, `tune_model` pasa `pipe`/`X_test` |
| `ui/train_tab.py` | `_store_result` usa `summarize_metrics` |
| `ui/results_tab.py` | Tabla de métricas con `n/d` y tooltip |
| `tests/test_model_trainer.py` | Tests de AUC binario/multiclase/ausente y contrato de métricas |

## 2. Contrato de `_compute_metrics`

```python
def _compute_metrics(task_type, y_test, y_pred, pipe=None, X_test=None) -> dict
```

Clasificación (siempre estas seis claves):

| Clave | Tipo | Notas |
|---|---|---|
| `accuracy` | `float` | siempre |
| `f1_macro` | `float` | siempre |
| `roc_auc` | `float \| None` | `None` solo si no es calculable |
| `auc_disponible` | `bool` | `False` cuando `roc_auc is None` |
| `auc_motivo` | `str \| None` |Motivo en español cuando no está disponible |
| `n_test` | `int` | filas evaluadas |

Regresión: `rmse`, `mae`, `r2`, `n_test` (sin cambios de cálculo).

### 2.1 Obtención de las puntuaciones

Se reutiliza el modelo a nivel de **pipeline** (`pipe.predict_proba`), que
aplica internamente typer/normalizer/preprocessor/balancer/selector y por tanto
sigue siendo correcto aunque se añadan pasos nuevos:

```python
modelo = pipe.named_steps.get("model") if pipe is not None else None
clases = list(getattr(modelo, "classes_", []))
```

- Binario: `proba[:, 1]`. Es coherente con `roc_auc_score` porque `classes_`
  viene ordenado y `LabelBinarizer` ordena las etiquetas igual, así que la
  columna 1 es siempre la clase positiva que usa sklearn internamente.
- Multiclase: `proba` completo con
  `labels=clases, multi_class="ovr", average="macro"`.
- Sin `predict_proba`: se intenta `pipe.decision_function(X_test)`
  (binario: `scores[:, 1]`; multiclase: una columna por clase en el mismo
  orden que `proba`).
- Sin ninguna de las dos → `auc_motivo` = “el modelo no expone probabilidades
  ni puntuaciones de decisión, que son necesarias para el AUC”.

### 2.2 Motivos de indisponibilidad

| Situación | `auc_motivo` |
|---|---|
| Sin `pipe` / `X_test` | “No se han proporcionado datos de prueba para calcular el AUC.” |
| Sin `proba` ni `decision_function` | “Este modelo no expone probabilidades ni puntuaciones de decisión, necesarias para el AUC.” |
| Una sola clase en `y_test` | “El AUC necesita al menos dos clases distintas en los datos de prueba.” |
| Falla `roc_auc_score` | “No se pudo calcular el AUC: {detalle}” |

Las tres primeras se comprueban **antes** de llamar a sklearn; la cuarta captura
`ValueError`. Nunca se propaga la excepción.

### 2.3 Fallo detectado en `tune_model`

`core/model_trainer.py:310` llama a
`_compute_metrics(task_type, y_test, y_pred)` sin `pipe` ni `X_test` en la rama
“sin espacio de búsqueda”, con lo que hoy no habría AUC en ese camino. Se pasa
`pipe=pipe, X_test=X_test`.

## 3. `available_metrics` y etiquetas

`model_specs.available_metrics(task_type, model_name=None)` conserva su
firma (compatibilidad) pero deja de filtrar: devuelve siempre
`list(METRICS[task_type])`. En el catálogo actual los cinco modelos de
clasificación declaran `supports_probability=True` y ninguno de regresión
incluye `roc_auc`, así que **el valor devuelto no cambia para ningún caso
real**; el cambio solo fija el contrato (R-009).

`supports_probability` se conserva como dato del catálogo: pasa a ser
informativo y sirve a `metrics_for_ranking` y a la documentación del modelo.

```python
METRIC_LABELS = {
    "accuracy": "Exactitud",
    "f1_macro": "F1 (macro)",
    "roc_auc": "AUC (ROC One-vs-Rest)",
    "rmse": "RMSE",
    "mae": "MAE",
    "r2": "R²",
}

METRIC_KEYS = {
    CLASSIFICATION: ("accuracy", "f1_macro", "roc_auc"),
    REGRESSION: ("rmse", "mae", "r2"),
}

def metric_label(key: str) -> str: ...
def metrics_for_ranking(task_type: str) -> str:
    """Métrica usada para ordenar y comparar modelos."""
    return {"classification": "f1_macro", "regression": "r2"}[task_type]
```

## 4. Presentación: `summarize_metrics`

Para que la UI no tenga que filtrar el diccionario (y que un `bool` —que en
Python es `int`— acabe impreso como `auc_disponible: 1.0000`, cosa que hoy
pasaría con `ui/train_tab.py:454`), se centraliza el formato:

```python
def summarize_metrics(metrics: dict) -> list[str]:
    """Líneas legibles en español, solo con claves de métrica conocidas."""
```

Reglas: clave conocida y valor numérico → “Exactitud: 0.9231”; `roc_auc is
None` → “AUC (ROC One-vs-Rest): n/d”; `best_params` se omite (lo muestra la UI
aparte); cualquier clave desconocida se ignora.

## 5. UI

### 5.1 `ui/results_tab.py`

`refresh()` sustituye el `lbl` de texto por:

- `tbl_metricas` (`QTableWidget` 2 columnas: Métrica, Valor) construida con
  `METRIC_KEYS[state.task_type]` y `METRIC_LABELS`.
- Valor `None` → texto `n/d` en gris (`Qt.GlobalColor.gray`) y
  `setToolTip(metrics["auc_motivo"])`.
- `lbl` queda para el mensaje de resumen de la búsqueda (combinaciones, mejor
  puntuación) y para los avisos no bloqueantes, por ejemplo
  “El AUC no está disponible: este modelo no expone probabilidades.”
- Si la tarea es clasificación y `auc_disponible` es `False`, se añade la fila
  informativa al `lbl`, no un `QMessageBox` (R-014: aviso, no error).

### 5.2 `ui/train_tab.py`

- `_store_result` usa `model_trainer.summarize_metrics(metrics)` en lugar de
  `isinstance(v, (int, float))`.
- El combo de métrica objetivo no cambia: sigue alimentando `_scoring_for`.

## 6. Efecto en `tune_model`

Ninguno en la búsqueda: `_scoring_for` y `_resolve_roc_auc` se conservan
(R-007). Con el usuario elegiendo `roc_auc` en un problema multiclase,
`_resolve_roc_auc` ya devuelve `roc_auc_ovr`.

## 7. Tests

- Binario con `LogisticRegression`: tres claves de métrica con `float`.
- Multiclase (3 clases, 200 filas): `roc_auc` no es `None`.
- Doble sin `predict_proba` ni `decision_function`: `roc_auc is None`,
  `auc_disponible is False`, `auc_motivo` en español y no vacío.
- `y_test` con una sola clase: `roc_auc is None` con el motivo del §2.2.
- `available_metrics` para las dos tareas y `metrics_for_ranking`.
- `summarize_metrics` con `roc_auc: None` y con `auc_disponible: True`
  (comprueba que el `bool` no se imprime como número).
- Regresión de `train_model` con las claves de siempre.