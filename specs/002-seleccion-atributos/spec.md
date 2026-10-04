# SDD-002 — Selección de atributos (4 algoritmos)

| Campo | Valor |
|---|---|
| Estado | Especificación — **esperando puerta 1** |
| Orden de ejecución | 5 de 6 |
| Ámbito | `core/feature_selection.py` (nuevo), `core/model_trainer.py`, `core/state.py`, `core/persistence.py`, `ui/train_tab.py`, `ui/results_tab.py`, `tests/test_feature_selection.py` |
| Depende de | SDD-004 (ambos tocan `build_pipeline`) |
| Bloquea a | SDD-005 |

## 1. Problema

Hoy `build_pipeline` (`core/model_trainer.py:112`) aplica
`build_preprocessor` y el estimador, sin ningún paso de selección. Con
datasets de ICDAR/fashion/Mall (ver `example/data/`) el modelo se entrena con
todas las columnas, alguna constante o directamente ruido, y la única forma de
saber qué columnas importan es leer coeficientes a mano.

## 2. Objetivo

Ofrecer al usuario **al menos cuatro algoritmos de selección de atributos**
seleccionables desde la pestaña de Entrenamiento, integrada en el pipeline
(por tanto persistidos en el `.automl` y aplicados en la predicción), con una
tabla de resultados queMostrar el atributo original, su puntuación y si
queda seleccionado.

## 3. Alcance

### 3.1 Incluido

- Catálogo declarativo de métodos en `core/feature_selection.py`.
- Cuatro algoritmos: chi-cuadrado, ANOVA F, información mutua y embebido
  (`SelectFromModel` con `RandomForestClassifier/Regressor`).
- Integración como paso del pipeline después del `ColumnTransformer`.
- Tabla de selección (atributo original, score, seleccionado) en la UI.
- Opción “Analizar selección” que permite ver el resultado antes de entrenar.
- Persistencia de la configuración y de las columnas seleccionadas en el
  bundle.

### 3.2 No incluido

- Métodos wrapper (`RFE`, `SequentialFeatureSelector`) o de Azure ML.
- Optimización automática del número de atributosSelected.
- Selección de atributos en la pestaña de Predicción: el bundle ya lleva el
  selector ajustado y no se reconfigura.
- Búsqueda del mejor método de selección dentro de `tune_model`: el método lo
  elige el usuario.

## 4. Requisitos

### API de `core`

- **R-001.** `core/feature_selection.py` expone el catálogo
  `SELECTION_METHODS: dict[str, SelectionMethod]` en el estilo de
  `core/model_specs.py`, con campos: `id`, `label` (español), `supports_regression`,
  `requires_non_negative`, `default_k`, `docstring` breve.
- **R-002.** Los cuatro métodos son:

  | `id` | Implementación | Clasificación | Regresión |
  |---|---|---|---|
  | `chi2` | `SelectKBest(score_func=chi2)` | sí | no |
  | `anova` | `SelectKBest(f_classif / f_regression)` | sí | sí |
  | `mutual_info` | `SelectKBest(mutual_info_classif / mutual_info_regression)` | sí | sí |
  | `embedded` | `SelectFromModel(RandomForest*)` | sí | sí |

- **R-003.** `available_methods(task_type: str) -> list[str]` devuelve solo los
  métodos compatibles con la tarea; la UI usa esa función para poblar el combo
  y no inventar opciones inválidas.
- **R-004.** `build_selector(method, task_type, k=None, percentile=None, random_state=42)`
  devuelve un selector de scikit-learn sin ajustar, con `k` o `percentile` como
  criterio de corte. Si ambos son `None` se usa el valor por defecto del
  catálogo.
- **R-005.** `chi2` exige datos no negativos. `build_preprocessor` normaliza las
  numéricas con `StandardScaler`, que sí puede producir valores negativos, así
  que cuando el método es `chi2` `build_pipeline` inserta **siempre** un paso
  `("scaler_nonneg", MinMaxScaler())` inmediatamente después del
  `ColumnTransformer`, que lleva la matriz ya codificada a `[0, 1]`. El paso no
  depende de si existe ya un escalado.
- **R-006.** `build_pipeline(df, model_name, task_type, target, casts=None,
  normalizations=None, selection=None, balancing=None)` acepta un dict
  `selection` con `{"method": ..., "k": ..., "percentile": ...}` y añade el paso
  `("selector", ...)` respetando el orden canónico de pasos de SDD-004 R-015
  (`preprocessor → [scaler] → [balancer] → selector → modelo`). Con
  `selection=None` el pipeline no cambia respecto al actual.
- **R-007.** El selector se ajusta **dentro** de `train_model` / `tune_model`,
  es decir después del split de entrenamiento. Nunca se ajusta sobre
  `clean_df` completo ni sobre el conjunto de prueba.
- **R-008.** `describe_selection(pipe, columns) -> pd.DataFrame` devuelve la
  tabla de selección con columnas `atributo` (nombre original), `codificado`
  (nombre interno tras el `ColumnTransformer`), `puntuacion` y `seleccionado`
  (`bool`). Para atributos one-hot de la misma columna original, todas las
  columnas codificadas se agregan bajo el nombre original y la puntuación es
  la mayor de ellas.
- **R-009.** El mapeo de nombres codificados a originales se hace por el
  separador `__` de las columnas transformadas: `num__edad → edad`,
  `cat__ciudad_b → ciudad`. Cualquier columna cuyo prefijo no exista en el
  `DataFrame` original se etiqueta con su propio nombre codificado.
- **R-010.** `build_pipeline` y `train_model` devuelven o dejan accesible el
  soporte del selector ajustado para que la UI y el bundle puedan listar las
  columnas seleccionadas sin volver a ajustar nada.

### UI

- **R-011.** `ui/train_tab.py` añade un `QGroupBox` “Selección de atributos” con:
  combo de métodos (datos de `available_methods(state.task_type)`), combo de
  criterio (“Número de atributos (k)” / “Porcentaje”), spin box numérico
  coherente con el criterio y checkbox “Sin selección” activo por defecto.
- **R-012.** Al pulsar “Analizar selección” se ejecuta un ajuste del selector
  sobre una partición de entrenamiento (80 % de `export_df`, estratificada si es
  clasificación) y se muestra la tabla de R-008 ordenada por puntuación
  descendente, con las columnas no seleccionadas atenuadas. La acción se
  ejecuta en `ui/workers.py` si el dataset es grande.
- **R-013.** Después de entrenar, `ui/results_tab.py` muestra la lista final de
  atributos seleccionados y su número.
- **R-014.** Si el método elegido no admite la tarea actual (por ejemplo `chi2`
  en regresión), la UI lo impide deshabilitando la opción y el entrenamiento
  falla con un mensaje en español si aun así llegara.

### Estado y persistencia

- **R-015.** `AppState` incorpora `selection_method` (`str | None`) y
  `selection_criteria` (`dict | None`), documentados en `core/state.py`.
- **R-016.** `BundleMetadata` incorpora `selection` (dict con método, criterio y
  lista de columnas seleccionadas) y `describe()` lo muestra en español.
- **R-017.** Un bundle guardado **sin** clave `selection` se sigue cargando y
  prediciendo sin errores: la clave es opcional y por defecto `None`.

## 5. Invariantes de arquitectura

- `core/feature_selection.py` no importa Qt ni `ui/`.
- El selector es un paso del pipeline: viaja dentro del `.automl`, no se
  recalcula al predecir.
- La selección se aprende solo con datos de entrenamiento.
- El dataset exportado (`export_df`) no se modifica: la selección de atributos
  no altera las filas ni las columnas del CSV que ve el usuario.

## 6. Criterios de aceptación

- **C-001.** Dado un dataset sintético con 3 columnas informativas y 20 de ruido,
  cuando se entrena con `selection={"method": "anova", "k": 3}`, entonces las
  3 columnas informativas aparecen seleccionadas en `describe_selection`.
- **C-002.** Para cada uno de los cuatro métodos y para clasificación y
  regresión (respetando `available_methods`), el pipeline construido se ajusta
  sobre un dataset de prueba y `pipeline.predict(X)` no falla.
- **C-003.** Dado `method="chi2"` y un dataset con valores negativos
  normalizados con z-score, cuando se construye el pipeline, entonces existe
  un paso de escalado no negativo antes del selector y el entrenamiento no
  lanza `ValueError: Input X must be non-negative`.
- **C-004.** Dado `method="chi2"` y `task_type="regression"`, cuando se llama a
  `available_methods`, entonces `"chi2"` no aparece en la lista.
- **C-005.** Dado un dataset con una columna categórica de 3 valores, cuando se
  describe la selección, entonces las columnas codificadas de esa categoría
  aparecen agrupadas bajo un único atributo original.
- **C-006.** Dado un bundle guardado con selección activa, cuando se carga en
  la pestaña Predicción y se aplica a datos nuevos, entonces la predicción usa
  el mismo subconjunto de atributos que en el entrenamiento.
- **C-007.** Dado un `.automl` antiguo sin clave `selection`, cuando se carga,
  entonces no se produce ningún error y la predicción funciona.
- **C-008.** Con `selection=None`, el pipeline resultante tiene exactamente los
  mismos pasos que antes de esta spec (test de regresión sobre
  `tests/test_model_trainer.py`).
- **C-009.** El número de columnas transformadas tras el selector es menor o
  igual al de antes, y `export_df` conserva todas las columnas originales.
- **C-010.** `grep -R "PySide6\|PyQt\|from ui\." core/` vacío y `pytest -q` en
  verde con al menos 6 tests nuevos en `tests/test_feature_selection.py`.

## 7. Dependencias

Ninguna nueva: los cuatro métodos existen en `scikit-learn 1.9.1`
(`SelectKBest`, `SelectFromModel`, `chi2`, `f_classif`, `f_regression`,
`mutual_info_classif`, `mutual_info_regression`). `RandomForest` como estimador
de `SelectFromModel` usa modelos ya presentes en `core/model_specs.py`.

## 8. Riesgos

| Riesgo | Mitigación |
|---|---|
| `chi2` falla con datos negativos | R-005: se fuerza `MinMaxScaler`; test C-003 |
| El prefijo `__` de las columnas transformadas cambia con sklearn | R-009 + test que usa `get_feature_names_out()` en vez de parsear a mano |
| Selección con k mayor que el número de columnas | Se recorta `k = min(k, n_features)` con aviso |
| `SelectFromModel` con `RandomForest` es lento en datasets grandes | Se documenta el coste y se ofrece `percentile` como atajo |
| Información mutua no determinista | `random_state` fijo en todo el catálogo (R-004) |

## 9. Decisiones que se cierran en `design.md`

- Cómo se parametrizan `chi2` y `mutual_info` en regresión (no existen: se
  rechaza el método o se degrada a `f_regression`).
- Si `embedded` expone el umbral del `SelectFromModel` (`median`, `mean`,
  tamaño fijo) en la UI o queda fijo en el catálogo.
- Cómo se agrupan los one-hot de una misma columna en la tabla de R-008
  (score máximo frente a suma).