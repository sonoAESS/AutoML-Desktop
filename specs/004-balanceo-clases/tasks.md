# SDD-004 — Tareas de implementación

Estado: **completada**. Es la cuarta spec en ejecutarse y fija el orden
canónico de pasos que respeta 002.

### [x] T-001 · `class_distribution` y dataclasses
**Ficheros:** `core/balancing.py` (nuevo)
**Requisitos:** R-001, R-002

1. `ClassCount`, `ClassDistribution` frozen dataclasses.
2. `rows` ordenado por `n` descendente y `str(clase)` ascendente.
3. `imbalance_ratio`, `shannon_entropy` con `scipy.stats.entropy(base=2)`,
   `normalized_entropy` y `level` según los umbrales 1.5 y 3.0.
4. Una sola clase → IR `1.0`, nivel `equilibrado`, entropías `0.0`.
5. `as_dict()` y `describe()` en español.
**Verificación:** C-001, C-002, C-003.

### [x] T-002 · `BALANCING_STRATEGIES` y `available_strategies`
**Ficheros:** `core/balancing.py`
**Requisitos:** R-003

1. Seis estrategias con etiqueta y descripción en español.
2. `available_strategies("regression") == ["none"]`.
3. `available_strategies("classification", model_name)` quita `class_weight`
   cuando `class_weight_path` es `None`.
**Verificación:** C-010, C-013.

### [x] T-003 · `BalancedSampler` esqueleto yestrategias nativas
**Ficheros:** `core/balancing.py`
**Requisitos:** R-004, R-005, R-006, R-007, R-008, R-016

1. `BaseEstimator`; `__init__` con `method`, `k_neighbors`, `random_state`,
   `task_type`.
2. `fit` guarda `n_before` y `classes_before`; `transform` identidad.
3. `fit_transform` remuestrea por índice y devuelve `(X[idx], y[idx])`.
4. `random_under`, `random_over`, `none` con `np.random.default_rng`.
5. `task_type != "classification"` → `ValueError` en español.
6. Atributos `n_after` y `classes_after` tras el remuestreo.
**Verificación:** C-004, C-005, C-006, C-007, C-014.

### [x] T-004 · SMOTE y SMOTEN
**Ficheros:** `core/balancing.py`
**Requisitos:** R-009, R-010, R-011

1. Importación perezosa de `imblearn`; sin él, `ValueError` con el paquete.
2. Entrada dispersa → `.toarray()` para `SMOTENC`.
3. Minoritaria con menos de 6 filas → `ValueError` explicando el límite.
**Verificación:** C-011, C-012 con `pytest.importorskip("imblearn")`.

### [x] T-005 · `class_weight_path` en el catálogo
**Ficheros:** `core/model_specs.py`
**Diseño §2.3**

1. Campo `class_weight_path: Optional[str] = None` en `ModelSpec`.
2. Valor para los cinco modelos de clasificación; `None` en KNN y en todos los
   de regresión.
3. Función `class_weight_path(model_name, task_type=None)`.
**Verificación:** test parametrizado con los seis modelos.

### [x] T-006 · `build_pipeline(..., balancing=None)` y pesos de clase
**Ficheros:** `core/model_trainer.py`
**Requisitos:** R-013, R-014, R-015, R-016

1. Insertar `("balancer", ...)` entre `preprocessor`/`scaler_nonneg` y `selector`.
2. `class_weight` → `pipe.set_params(**{path: "balanced"})`; `None` → `ValueError`.
3. `train_model` y `tune_model`: parámetro `balancing=None`.
4. `tune_model`: fusionar `{path: ["balanced"]}` en `param_grid`.
**Verificación:** C-008, C-009, C-013 (tamaño del test set intacto).

### [x] T-007 · Tests de `balancing`
**Ficheros:** `tests/test_balancing.py` (nuevo)
**Criterios:** C-001 a C-014

1. Tests de `class_distribution` (equilibrado, severo, una clase, vacío).
2. Tests del sampler con `y` sintético 90/10 en ndarray, `DataFrame` y
   `csr_matrix` (comprobar que el esquema y el tipo se conservan).
3. Test de invariante: `train_model` con `random_under` → `len(y_test)` igual
   que sin balancear y `len(export_df)` intacto.
4. Test de `class_weight` en LogisticRegression y aviso con KNN.
**Verificación:** `pytest -q tests/test_balancing.py` verde.

### [x] T-008 · Persistencia y estado
**Ficheros:** `core/persistence.py`, `core/state.py`
**Requisitos:** R-022, R-023, R-024

1. `BundleMetadata.balancing` y `build_metadata(balancing=...)` leyendo las
   estadísticas del paso ajustado.
2. `describe()` con la línea de balanceo.
3. `_CAMPOS_POR_DEFECTO["balancing"] = {}` en la migración de 002.
4. `AppState.balancing`, `AppState.class_distribution` + `reset_*`.
**Verificación:** test de ida y vuelta del bundle con y sin la clave antigua.

### [x] T-009 · Tabla de distribución en `PreprocessTab`
**Ficheros:** `ui/preprocess_tab.py`
**Requisitos:** R-017, R-018

1. `gb_distribucion` con `tbl_distribucion` y `lbl_distribucion`, oculto si no
   es clasificación o no hay objetivo.
2. Cálculo con `balancing.class_distribution(clean_df[target])`.
3. Rellenado de la tabla y `state.class_distribution`.
**Verificación:** test headless con dataset 90/10: la tabla tiene 2 filas y la
etiqueta contiene “IR”.

### [x] T-010 · Selector de estrategia en `TrainTab`
**Ficheros:** `ui/train_tab.py`, `ui/main_window.py`
**Requisitos:** R-019, R-020, R-021

1. `cmb_balancing` + `spn_vecinos` con la visibilidad por método.
2. `_balancing_config()` y propagación a `train_model`, `TuneWorker` y
   `_run_context`.
3. `_build_param_widgets` omite el control de `class_weight` con balanceo activo.
4. Señal `config_changed` en cambio de tarea/objetivo.
5. `MainWindow`: `train_tab.config_changed.connect(preprocess_tab.refresh)`.
6. Aviso en resultados con el efecto `n_before → n_after`.
**Verificación:** test headless que cambia de modelo a KNN y comprueba que
`class_weight` desaparece del combo de balanceo.

### [x] T-011 · Instalar `imbalanced-learn`
**Ficheros:** ninguno (`entorno/`)

1. Añadirlo a `[project.optional-dependencies] smote` de `pyproject.toml`.
2. `pip install imbalanced-learn` con confirmación del usuario.
**Verificación:** `python -c "import imblearn; print(imblearn.__version__)"`.

### [x] T-012 · `README.md` y `AGENTS.md`
**Ficheros:** `README.md`, `AGENTS.md`

1. `AGENTS.md` §2: añadir `core/balancing.py`; §3.4 su responsabilidad y la
   invariante de no remuestrear `clean_df`.
2. `README.md`: tabla estilo Weka, IR, las seis estrategias y la advertencia de
   que el dataset exportado no se balancea.

## Verificación final de la spec

```bash
python -m pytest -q
grep -R "PySide6\|PyQt\|from ui\." core/   # vacío
```

- [x] C-001…C-015 cubiertos.
- [x] Test explícito de que `export_df` no cambia de filas con balanceo.