# SDD-005 — Comparativa de modelos con test de Friedman

| Campo | Valor |
|---|---|
| Estado | Especificación — **esperando puerta 1** |
| Orden de ejecución | 6 de 6 (última) |
| Ámbito | `core/comparison.py` (nuevo), `core/model_trainer.py`, `ui/results_tab.py`, `ui/workers.py`, `tests/test_comparison.py` |
| Depende de | SDD-002 (selección), SDD-003 (métricas), SDD-004 (balanceo) |
| Bloquea a | — |

## 1. Problema

El usuario pide saber **qué modelos son realmente buenos en estos datos**,
no solo cuál dio el mejor número en un único split. Hoy la pestaña de
Resultados muestra las métricas de un único modelo y la curva ROC; no existe
ninguna comparación sistemática entre los cinco modelos de clasificación
disponibles.

El criterio de referencia en AutoML es el **test de Friedman**: se evalúan `k`
algoritmos sobre los mismos `k` folds, se ordenan por puntuación media en cada
fold y se calcula el estadístico de Friedman sobre esos rangos. Con un
post-hoc (Nemenyi o Holm) se determina qué modelos son
estadísticamente distinguibles entre sí.

## 2. Objetivo

Un panel “Comparativa de modelos” que evalúe todos los modelos compatibles con
la tarea sobre una validación cruzada repetida, calcule el ranking medio, el
estadístico de Friedman, un post-hoc con corrección de Holm y un diagrama de
diferencias críticas, y destaque los modelos que no son significativamente
peores que el mejor.

## 3. Alcance

### 3.1 Incluido

- `core/comparison.py` con evaluación comparada, Friedman y post-hoc.
- Reutilización de `build_pipeline` para que cada modelo se construya con el
  mismo preprocesamiento, balanceo y selección que en entrenamiento.
- Diagrama de diferencias críticas dibujado en la UI con matplotlib.
- Ejecución en worker con barra de progreso.

### 3.2 No incluido

- Comparación entre varios datasets a la vez: el test de Friedman clásico
  trata cada dataset como un bloque y aquí los bloques son folds del mismo
  dataset, que es la variante de validación cruzada emparejada.
- Búsqueda automática del mejor modelo y guardado automático del pipeline
  ganador.
- Pruebas de Nemenyi para diagramas con solo dos modelos (el test necesita al
  menos 3).
- Comparación con algoritmos externos (Weka, AutoML 3, H2O).

## 4. Requisitos

### API de `core`

- **R-001.** `compare_models(df, target, model_names, task_type, metric,
  casts=None, normalizations=None, selection=None, balancing=None,
  n_splits=5, n_repeats=3, random_state=42, progress=None) -> ModelComparison`.
- **R-002.** La validación cruzada es **repetida y estratificada**:
  `RepeatedStratifiedKFold` para clasificación y `RepeatedKFold` para
  regresión. Las mismas particiones se reutilizan para **todos** los modelos
  (condición de validez del test: los bloques deben estar pareados).
- **R-003.** Por defecto `n_splits=5` y `n_repeats=3`, es decir
  `b = n_splits * n_repeats = 15` bloques por modelo. Con menos de 3 modelos,
  `compare_models` lanza `ValueError` en español: el test de Friedman no es
  válido para `k < 3` algoritmos.
- **R-004.** `n_splits` se limita antes de empezar al tamaño de la clase
  minoritaria (clasificación) y al número de filas disponibles; si el valor
  pedido no es válido, se reduce y se avisa en español, de modo que ningún
  fold se quede sin la clase minoritaria.
- **R-005.** Cada modelo se evalúa construyendo su pipeline con
  `build_pipeline` y ajustándolo **solo** con el fold de entrenamiento, igual
  que en `train_model`. Ningún fold de prueba se usa para ajustar.
- **R-006.** La puntuación de cada modelo en cada bloque es la métrica
  indicada en `metric` (de `available_metrics`), no la agregada de un
  `cross_val_score` de sklearn.
- **R-007.** `ModelComparison` es un dataclass con:
  - `metric: str` y `task_type: str`;
  - `scores: pd.DataFrame` con columnas `modelo`, `repeticion`, `fold`,
    `puntuacion`;
  - `ranking: pd.DataFrame` con `modelo`, `puntuacion_media`, `desviacion`,
    `rango_medio` ordenado de mejor a peor;
  - `friedman_statistic: float`, `friedman_p: float`, `n_observaciones: int`;
  - `posthoc: pd.DataFrame` con el detalle de las comparaciones por pares
    emparejando bloques, `p_ajustada` (corrección de Holm) y
    `significativo: bool`;
  - `critical_difference: float` y `alpha: float`;
  - `mejor_modelo: str`.
- **R-008.** El estadístico se calcula con
  `scipy.stats.friedmanchisquare` sobre la matriz de puntuaciones
  bloque × modelo.
- **R-009.** El post-hoc usa el test de Wilcoxon pareado sobre los bloques
  comunes de cada par de modelos, con corrección de Holm para el control de
  la tasa de error familiar. Para modelos con menos de 6 bloques se usa el
  test t pareado y se anota en la tabla.
- **R-010.** La significación se marca con `alpha = 0.05` (parámetro
  modificable) y solo se resaltan los modelos cuyo rango medio está dentro de
  la diferencia crítica del mejor.
- **R-011.** Los rangos se calculan con la convención de que **menor rango es
  mejor**, y el objetivo de cada métrica se invierte cuando es una métrica de
  error (`rmse`, `mae`): se usa el valor negated para ordenar.
- **R-012.** El diagrama de diferencias críticas **no** se genera en `core`:
  `core` devuelve los datos (`ranking`, `critical_difference`) y el dibujo vive
  en `ui/results_tab.py` con matplotlib.
- **R-013.** Si un modelo falla en un fold (por ejemplo, hiperparámetros
  inválidos con una clase ausente), el fallo se registra y el modelo se excluye
  de la comparación con un aviso en español; si sobreviven menos de 3 modelos,
  la comparación entera falla con `ValueError`.
- **R-014.** `progress` es un callback opcional `progress(hecho, total)` para
  que el worker de la UI muestre el avance.

### UI

- **R-015.** `ui/results_tab.py` añade el grupo “Comparativa de modelos
  (Friedman)” visible solo con `state.task_type` definida y dataset
  transformado.
- **R-016.** El grupo contiene: combo de métrica (`available_metrics`), check
  de número de repeticiones, botón “Comparar modelos compatibles”, barra de
  progreso y una tabla con modelo, puntuación media, desviación, rango medio y
  marca de significación.
- **R-017.** Debajo de la tabla se dibuja el diagrama de diferencias críticas
  con `matplotlib` (eje horizontal de rangos medios, barra de error con la
  diferencia crítica, modelos en el eje vertical).
- **R-018.** Se muestra un resumen en español con el estadístico, el p-valor y
  una frase de interpretación (por ejemplo: “Friedman χ²=12.4, p=0.002: las
  diferencias entre modelos son estadísticamente significativas”).
- **R-019.** La comparación se ejecuta en `ui/workers.py` con su propia
  señal `progress`, y los controles se deshabilitan mientras dura.
- **R-020.** El resultado se guarda en `AppState.comparison` para poder
  consultarlo al cambiar de pestaña.

## 5. Invariantes de arquitectura

- `core/comparison.py` no importa Qt ni `ui/`; el diagrama no se dibuja en
  `core`.
- La comparación reutiliza `build_pipeline`; no reimplementa el
  preprocesamiento.
- Los folds son idénticos para todos los modelos; si se cambiaran, el test de
  Friedman dejaría de ser válido.
- El trabajo se hace en el worker, nunca en el hilo principal (`AGENTS.md`
  §5.3).
- La comparación es de solo lectura sobre `export_df`: no modifica
  `AppState.pipeline` ni `AppState.metrics`.

## 6. Criterios de aceptación

- **C-001.** Con tres modelos, `task_type="classification"`, `n_splits=5` y
  `n_repeats=2`, entonces `n_observaciones == 10` (5 folds × 2 repeticiones) y
  `ranking` tiene exactamente 3 filas ordenadas de mejor a peor.
- **C-002.** Dos modelos con exactamente la misma puntuación media empatan en
  rango medio, y el test de Friedman da p-valor alto (no significativo).
- **C-003.** Con un modelo claramente superior a los demás, el p-valor de
  Friedman es menor que `alpha = 0.05` y `posthoc` marca diferencias
  significativas frente al mejor.
- **C-004.** Con 5 modelos (los de clasificación por defecto) la comparación se
  completa y la tabla tiene 5 filas.
- **C-005.** Con menos de 3 modelos solicitados, `compare_models` lanza
  `ValueError` con mensaje en español.
- **C-006.** Con `metric="rmse"` (métrica de error), el `ranking` ordena de
  menor a mayor RMSE (menor rango = mejor).
- **C-007.** Los bloques usados son los mismos para todos los modelos: el test
  comprueba que la partición es reproducible con el mismo `random_state` y que
  `scores` tiene una fila por modelo y bloque.
- **C-008.** Ningún fold de prueba se usa para ajustar: se comprueba con un
  dataset donde un fold concreto tiene una clase ausente y el modelo sigue
  ajustándose sin fuga.
- **C-009.** El diagrama se dibuja sin error con los datos de `ranking` y
  `critical_difference`, y `matplotlib` se invoca desde `ui/`, no desde
  `core/`.
- **C-010.** El progreso informa de `1..total` y llega al total; la UI se
  desbloquea al terminar, tanto en éxito como en fallo.
- **C-011.** `pytest -q` en verde con al menos 6 tests nuevos en
  `tests/test_comparison.py` (los pesados se marcan como `slow`).

## 7. Dependencias

Ninguna nueva: `scipy.stats` (ya instalado) aporta `friedmanchisquare`,
`wilcoxon`, `ttest_rel`; `scikit-learn` aporta `RepeatedStratifiedKFold`;
`matplotlib` se usa para el diagrama.

## 8. Riesgos

| Riesgo | Mitigación |
|---|---|
| Coste computacional alto (5 modelos × 15 folds = 75 ajustes) | Ejecución en worker con progreso y `n_repeats` configurable |
| Fold con una clase ausente en StratifiedKFold | `n_splits` limitado al tamaño de la clase minoritaria y validación previa |
| Interpretación errónea del p-valor | Resumen en español con la frase interpretativa y post-hoc con corrección de Holm |
| Wilcoxon exige pares y n ≥ 6 | Fallback a t pareado con anotación (R-009) |
| Comparar métricas de error como si fueran de ganancia | Inversión de rango documentada (R-011) |

## 9. Decisiones que se cierran en `design.md`

- Valores por defecto de `n_splits` y `n_repeats` (propuesta 5 × 3) y si se
  avisa al usuario del tiempo estimado.
- Si se incluye el test de Nemenyi además del de Wilcoxon-Holm.
- Si el diagrama se dibuja con `matplotlib` en un `FigureCanvas` embebido o
  se genera como imagen temporal.
- Si se cachea el resultado en `AppState` cuando cambia el dataset (y cómo se
  invalida).