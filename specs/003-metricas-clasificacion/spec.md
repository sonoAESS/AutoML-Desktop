# SDD-003 — Métricas de clasificación siempre disponibles (F1, AUC, exactitud)

| Campo | Valor |
|---|---|
| Estado | **Implementada** (SDD-003 cerrada) |
| Orden de ejecución | 3 de 6 |
| Ámbito | `core/model_trainer.py`, `core/model_specs.py`, `ui/train_tab.py`, `ui/results_tab.py`, `tests/test_model_trainer.py` |
| Depende de | — |
| Bloquea a | SDD-005 (usa `available_metrics` para ordenar modelos) |

## 1. Problema

`core/model_trainer.py:82::available_metrics` devuelve para clasificación
exactitud y `f1_macro`, y añade `roc_auc` **solo si el modelo admite
probabilidades y el problema es binario**. `core/model_trainer.py:375::
_compute_metrics` calcula el AUC únicamente en el caso binario.

Consecuencias para el usuario:

- Un problema multiclase (el dataset `Personality.csv` de `example/data/`, con
  7 clases) nunca muestra AUC, aunque la vista One-vs-Rest sea perfectamente
  válida.
- Si un modelo future no expone `predict_proba`, la columna desaparece del
  panel en lugar de explicarse.
- `results_tab.py` y la comparativa de SDD-005 necesitan un contrato estable
  de métricas, no un subconjunto variable según el modelo.

## 2. Objetivo

Para toda tarea de clasificación se ofrecen **siempre** las tres vistas
solicitadas —exactitud, F1 y AUC— con un contrato fijo: la métrica existe en
el diccionario de resultados y, cuando es matemáticamente inaplicable, se
informa con un motivo en español en lugar de omitirse o inventarse.

## 3. Alcance

### 3.1 Incluido

- Exactitud y F1 siempre presentes en clasificación.
- AUC binario y multiclase (OVR, media macro).
- Clave `auc_disponible` + motivo cuando no hay forma de calcularlo.
- `available_metrics("classification", ...)` con lista fija.
- Etiqueta `n/d` con tooltip en la UI.
- Métricas de ranking para SDD-005.

### 3.2 No incluido

- AUC en regresión (no está definido estadísticamente para variables objetivo
  continuas). Las métricas de regresión siguen siendo `rmse`, `mae` y `r2`.
- Curva ROC y AUC por clase (la curva ROC ya existe hoy en `results_tab`).
- F1 ponderado o por clase como métrica seleccionable adicional.
- Métricas de calibración, MCC o Cohen's kappa.

## 4. Requisitos

### `core/model_trainer.py`

- **R-001.** `_compute_metrics` devuelve, para clasificación, siempre las
  claves `accuracy` y `f1_macro`, y la clave `roc_auc`, cuyo valor es `float`
  o `None`.
- **R-002.** Además devuelve `auc_disponible: bool` y `auc_motivo: str | None`
  (motivo en español cuando `auc_disponible` es `False`).
- **R-003.** Binario: si el modelo expone `predict_proba`, se usa la columna
  correspondiente a la clase positiva de `classes_`; si no, se usa
  `decision_function`. El modelo se persiste ya calibrado
  (`SVC(probability=True)`), por lo que en la práctica siempre hay `proba`.
- **R-004.** Multiclase: se calcula
  `roc_auc_score(y, proba, multi_class="ovr", average="macro")`, que es la
  convención usada por la mayoría de herramientas y la única que puede
  devolver un único número comparable entre modelos.
- **R-005.** Si hay menos de 2 clases en `y_test`, o si solo se evalúa una
  clase, `roc_auc` es `None` y `auc_motivo` explica que se necesitan al menos
  dos clases.
- **R-006.** El resto de errores de `roc_auc_score` (valores no normalizados,
 shape incompatible) se capturan y se traducen a `auc_motivo`, nunca a un
  traceback.
- **R-007.** `_resolve_roc_auc` se conserva porque la usa `tune_model` como
  scorer; no se rompe su firma.
- **R-008.** Las métricas de regresión no cambian: `rmse`, `mae`, `r2`, más
  `n_train`, `n_test` y `n_features`.

### `core/model_specs.py` / trainers

- **R-009.** `available_metrics("classification", model_name)` devuelve
  **siempre** `["accuracy", "f1_macro", "roc_auc"]`, sin filtrar por modelo.
  Para regresión devuelve `["rmse", "mae", "r2"]`.
- **R-010.** `metrics_for_ranking(task_type)` devuelve la métrica por defecto
  usada para ordenar y comparar modelos: `f1_macro` en clasificación y `r2`
  en regresión. La usa SDD-005.
- **R-011.** Las etiquetas en español son: “Exactitud”, “F1 (macro)”,
  “AUC (ROC One-vs-Rest)” en clasificación; “RMSE”, “MAE”, “R²” en regresión.
  `METRIC_LABELS` es un dict único que consumen la UI y las specs.

### UI

- **R-012.** `ui/results_tab.py` muestra la tabla de métricas siempre con las
  tres filas de clasificación; cuando el valor es `None` muestra `n/d` en
  gris y el tooltip contiene `auc_motivo`.
- **R-013.** `ui/train_tab.py` sigue ofreciendo la lista de
  `available_metrics` para elegir la métrica objetivo del entrenamiento; los
  valores que ajustan el modelo no cambian.
- **R-014.** Si el AUC no está disponible, un aviso no bloqueante en español
  explica el motivo debajo de la tabla de métricas.

## 5. Invariantes de arquitectura

- El cálculo de métricas sigue siendo función pura de `core/` y testeable sin
  Qt.
- “Siempre ofrecer” significa **siempre existir la vista**, no siempre poder
  inventar un número: un `None` documentado es el comportamiento correcto.
- No se recalculan métricas en la UI.

## 6. Criterios de aceptación

- **C-001.** Dado un problema binario con `LogisticRegression`, cuando se
  entrena, entonces `metrics` contiene `accuracy`, `f1_macro` y `roc_auc` con
  valor `float` entre 0 y 1.
- **C-002.** Dado un problema multiclase con al menos 3 clases, cuando se
  entrena, entonces `roc_auc` es un `float` (no `None`) y se calcula con
  `multi_class="ovr"`, `average="macro"`.
- **C-003.** Dado un estimador sin `predict_proba` ni `decision_function`
  (simulado en test con un doble), cuando se entrena, entonces `roc_auc is
  None`, `auc_disponible is False` y `auc_motivo` no está vacío y está en
  español.
- **C-004.** Dado un conjunto de prueba con una sola clase, cuando se calcula
  el AUC, entonces devuelve `None` con motivo y el resto de métricas se
  calculan con normalidad.
- **C-005.** `available_metrics("classification")` devuelve exactamente las
  tres claves en orden fijo, para los cinco modelos de clasificación.
- **C-006.** `available_metrics("regression")` sigue devolviendo `rmse`, `mae`,
  `r2` y ninguna más.
- **C-007.** `metrics_for_ranking("classification") == "f1_macro"` y
  `metrics_for_ranking("regression") == "r2"`.
- **C-008.** `METRIC_LABELS` contiene las cinco etiquetas en español y ninguna
  clave de `available_metrics` carece de etiqueta.
- **C-009.** Los tests existentes de `tests/test_model_trainer.py` siguen
  pasando sin modificarlos.
- **C-010.** `pytest -q` en verde con al menos 6 tests nuevos o ampliados.

## 7. Dependencias

Ninguna nueva. `scipy`/`scikit-learn` ya aportan `roc_auc_score`,
`accuracy_score` y `f1_score`.

## 8. Riesgos

| Riesgo | Mitigación |
|---|---|
| El AUC multiclase OVR depende del orden de `classes_` | Se pasa la matriz `proba` completa con `labels=modelo.classes_` |
| Modelos sin probabilidad (futuros, p. ej. `SVC` sin calibrar) | Contrato `auc_disponible` + `n/d` con motivo (R-002, R-003) |
| Comparar `f1_macro` entre clases desbalanceadas puede engañar | Se ofrece además la distribución de clases en SDD-004 y el usuario puede ver el balanceo aplicado |

## 9. Decisiones que se cierran en `design.md`

- Si el AUC multiclase se expone también por clase en `results_tab` o solo el
  valor macro.
- Si se añade `f1` binario (clase positiva) a la tabla de métricas.
- Si `available_metrics` mantiene el parámetro `model_name` por compatibilidad
  aunque ya no filtre nada.