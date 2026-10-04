# SDD-002 — Tareas de implementación

Estado: **completada**. Depende de SDD-004, que fija el orden canónico de
pasos del pipeline (R-015).

## Fase A — Núcleo puro

### [x] T-001 · Catálogo `SELECTION_METHODS`
**Ficheros:** `core/feature_selection.py` (nuevo)
**Requisitos:** R-001, R-002, R-003

1. Docstring de módulo: “selección de atributos; solo lógica pura, sin Qt”.
2. `SelectionMethod` frozen dataclass con los campos del diseño §4.1.
3. `SELECTION_METHODS` con `chi2`, `anova`, `mutual_info`, `embedded`.
4. `available_methods(task_type)` en el orden del catálogo, filtrando regresión.
**Verificación:** test de que `available_methods("regression")` no contiene `chi2` y `available_methods("classification")` sí.

### [x] T-002 · Selectores seguros
**Ficheros:** `core/feature_selection.py`
**Criterios:** C-001, C-003

1. `SafeSelectKBest(SelectKBest)` con `fit` que recorta `k` y guarda `k_effective_`.
2. `SafeSelectFromModel(SelectFromModel)` con recorte de `max_features` y `n_features_`.
3. Ambas de primer nivel (pickle-friendly), sin lambdas en los atributos ajustados.
**Verificación:** `pytest` con `X` de 4 columnas y `k=10`: ajusta, deja 4 columnas y `k_effective_ == 4`.

### [x] T-003 · `build_selector`
**Ficheros:** `core/feature_selection.py`
**Requisitos:** R-002, R-004

1. Validar método desconocido, método incompatible con la tarea y porcentaje no soportado (tres `ValueError` en español).
2. `_score_func(method, task_type)` con `partial` para fijar `random_state` en información mutua.
3. `_random_forest(task_type, random_state)` con `n_estimators` de parámetro (100 por defecto, menos en tests).
4. Devolver `SelectPercentile`, `SafeSelectFromModel` o `SafeSelectKBest`.
**Verificación:** 8 tests de la matriz método × tarea.

### [x] T-004 · `feature_origin_map`
**Ficheros:** `core/feature_selection.py`
**Criterios:** C-005

1. Recorrer `preprocessor.transformers_` en orden, `_last_encoder(trans)` para detectar el `OneHotEncoder` final.
2. Un nombre original por columna numérica; uno repetido por cada categoría.
3. Respaldo con `n.split("__", 1)[-1]` si el recuento no cuadra.
**Verificación:** test con 1 numérica + 1 categórica de 3 valores → mapa de 4 entradas con `edad, ciudad, ciudad, ciudad`.

### [x] T-005 · `describe_selection` y `selected_feature_names`
**Ficheros:** `core/feature_selection.py`
**Requisitos:** R-008, R-009, R-010

1. `scores_` → si no existe, `feature_importances_` del estimador interno → si no, `NaN`.
2. Agregar por atributo original: `n_columnas_codificadas`, `puntuacion` (máx), `seleccionado` (any).
3. Ordenar por puntuación descendente; índice reiniciado.
4. Sin paso `selector` → `DataFrame` vacío con las columnas esperadas.
**Verificación:** test sobre un pipeline de toy data que el subconjunto seleccionado son las columnas informativas.

### [x] T-006 · Tests de `feature_selection`
**Ficheros:** `tests/test_feature_selection.py` (nuevo)
**Criterios:** C-001 a C-005

Fixtures: dataset sintético con `y = (x0 + x1 > umbral)` y 20 columnas de ruido
`np.random`, más su versión de regresión.
Tests: catálogo, matriz método × tarea, recorte de `k`, chi2 con negativos,
agrupación one-hot, dataframe vacío.
**Verificación:** `pytest -q tests/test_feature_selection.py` verde.

## Fase B — Integración con el pipeline

### [x] T-007 · `build_pipeline(..., selection=None)` y `analyze_selection`
**Ficheros:** `core/model_trainer.py`
**Requisitos:** R-006, R-007, R-005 (corrigida)

1. `import core.feature_selection as feature_selection` y `MinMaxScaler`.
2. Insertar `scaler_nonneg` (solo chi2) y `selector` en el orden canónico.
3. `train_model` y `tune_model`: parámetro `selection=None` propagado a `build_pipeline`.
4. `analyze_selection` con split de entrenamiento estratificado.
**Verificación:** test de regresión de que `build_pipeline` sin `selection` tiene los mismos pasos que antes; test de que `analyze_selection` devuelve la tabla.

### [x] T-008 · Persistencia y estado
**Ficheros:** `core/persistence.py`, `core/state.py`
**Requisitos:** R-015, R-016, R-017

1. `BundleMetadata.selection: dict = field(default_factory=dict)`.
2. `build_metadata(..., selection=None)` con `selected` = columnas codificadas activas.
3. `describe()` añade la línea de selección si la hay.
4. `_CAMPOS_POR_DEFECTO = {"selection": dict, "balancing": dict}` y `_upgrade_metadata` aplicado en `load_bundle`.
5. `AppState.selection_method`, `selection_criteria` + `reset_data`/`reset_model`.
**Verificación:** test que guarda y carga un bundle con selección y otro sin la clave antigua (construido con un `BundleMetadata` de los campos previos); ambos cargan sin `AttributeError` y `describe()` no revienta.

## Fase C — Interfaz

### [x] T-009 · `SelectionWorker`
**Ficheros:** `ui/workers.py`
**Requisitos:** R-012 (parte del worker)

1. `SelectionWorker(QObject)` con `finished = Signal(object)` y `failed = Signal(str)`.
2. `run()` llama a `model_trainer.analyze_selection` y emite el `DataFrame`.
**Verificación:** test headless con `QCoreApplication` y un dataset pequeño.

### [x] T-010 · Grupo de selección en `TrainTab`
**Ficheros:** `ui/train_tab.py`
**Requisitos:** R-011, R-012, R-014

1. `gb_seleccion` con los widgets de la tabla del diseño §7.1.
2. `_refresh_seleccion()` en `_on_task_changed` y al recargar la interfaz.
3. `_selection_config()` devuelve `None` o el dict esperado.
4. “Analizar selección” → worker; deshabilitar el combo de porcentaje con `embedded`; deshabilitar `chi2` en regresión.
5. `selection` en `train_model`, en `TuneWorker` y en `self._run_context`.
6. Bloquear los controles en `_lock_controls` durante el análisis y el entrenamiento.
**Verificación:** test headless que marca la casilla, elige `anova`, pulsa “Analizar” y comprueba que la tabla tiene filas; y que al entrenar sin selección el resultado es idéntico a antes.

### [x] T-011 · Resumen en `ResultsTab`
**Ficheros:** `ui/results_tab.py`
**Requisitos:** R-013

1. En `refresh()`, `describe_selection(self.state.pipeline)` y línea con método y número de columnas.
2. Si no hay selector, no mostrar nada.
**Verificación:** test headless con pipeline con y sin selector.

## Fase D — Documentación

### [x] T-012 · `README.md` y `AGENTS.md`
**Ficheros:** `README.md`, `AGENTS.md`

1. `AGENTS.md` §2: añadir `core/feature_selection.py`; §3.4 describir su responsabilidad.
2. `README.md`: sección “Selección de atributos” con los cuatro métodos, cuándo usar cada uno y su coste.

## Verificación final de la spec

```bash
grep -R "PySide6\|PyQt\|from ui\." core/   # vacío
python -m pytest -q
```

- [x] C-001…C-010 cubiertos.
- [x] `pytest -q` verde y `black`/`isort` sin diferencias.