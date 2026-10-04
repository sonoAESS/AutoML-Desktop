# SDD-004 — Balanceo de clases y tabla de distribución estilo Weka

| Campo | Valor |
|---|---|
| Estado | Especificación — **esperando puerta 1** |
| Orden de ejecución | 4 de 6 |
| Ámbito | `core/balancing.py` (nuevo), `core/model_trainer.py`, `core/state.py`, `core/persistence.py`, `ui/preprocess_tab.py`, `ui/train_tab.py`, `tests/test_balancing.py` |
| Depende de | — |
| Bloquea a | SDD-002 (mismo `build_pipeline`), SDD-005 |

## 1. Problema

El usuario pidió dos cosas distintas:

1. **Ver** el reparto de clases con el formato de Weka y la razón
   `mayoritaria / minoritaria`.
2. **Corregir** el desbalance según el target elegido, no según el dataset
   completo.

Hoy la app no muestra ninguna de las dos cosas, y el riesgo de una
implementación ingenua es grave: si se remuestrea `clean_df` o `export_df`
antes del split, las métricas quedan infladas y el modelo se evalúa sobre datos
que el usuario nunca verá en producción.

## 2. Objetivo

- Tabla de distribución de clases con el ratio de imbalance (IR) al estilo de
  Weka, visible en Preprocesamiento en cuanto hay objetivo de clasificación.
-Cinco estrategias de balanceo seleccionables en Entrenamiento, aplicadas
  **solo al conjunto de entrenamiento**, con SMOTE/SMOTEN disponibles vía
  `imbalanced-learn`.
- La distribución y la estrategia aplicadas quedan registradas en el bundle.

## 3. Alcance

### 3.1 Incluido

- `core/balancing.py` con el cálculo de distribución, IR y entropía.
- Estrategias: `none`, `class_weight`, `random_under`, `random_over`, `smote`,
  `smoten`.
- `BalancedSampler` como paso del pipeline.
- `class_weight="balanced"` inyectado en el estimador.
- Tabla de distribución en `ui/preprocess_tab.py`.
- Selector de estrategia en `ui/train_tab.py`.
- Persistencia de la configuración.

### 3.2 No incluido

- Muestreo cuando la variable objetivo es continua: el balanceo es un
  concepto de clasificación.
- Remuestrear `clean_df` / `export_df` en ningún punto (ver invariante).
- Limpiezas combinadas (Tomek, ENN, ensembles `EasyEnsemble`).
- Ajuste automático de umbral de decisión.
- Balanceo por grupos o por peso de instancia.

## 4. Requisitos

### API de `core`

- **R-001.** `class_distribution(y) -> ClassDistribution` con:
  - `rows: list[dict]` con `clase`, `n`, `pct` (2 decimales), ordenado por
    `n` descendente y luego por nombre de clase para que sea determinista;
  - `n_classes`, `total`, `majority`, `minority`, `majority_n`, `minority_n`;
  - `imbalance_ratio: float` = `majority_n / minority_n` (convención Weka);
  - `shannon_entropy: float` y `normalized_entropy: float` en `[0, 1]`;
  - `level: str` con uno de `equilibrado` (IR < 1.5),
    `moderado` (1.5 ≤ IR < 3), `severo` (IR ≥ 3);
  - `describe() -> str` en español, apta para una etiqueta de la UI.
- **R-002.** Con una sola clase, `imbalance_ratio` es `1.0`, `level` es
  `equilibrado` y no se lanza excepción.
- **R-003.** `BALANCING_STRATEGIES: dict[str, BalancingStrategy]` con `id`,
  `label` (español), `description` (español), `needs_resampling: bool`,
  `requires_smote: bool`.
- **R-004.** `BalancedSampler(method="random_under" | "random_over" | "smote" |
  "smoten", k_neighbors=5, random_state=42)` es un transformer de
  scikit-learn con `fit(X, y)` y `transform(X)`.
- **R-005.** `fit` **remuestrea** `(X, y)` y guarda en el objeto el tamaño
  original y el tamaño resultante (`n_before`, `n_after`,
  `classes_before: dict`, `classes_after: dict`).
- **R-006.** `transform` devuelve `X` **sin modificar**. Esto permite que el
  pipeline tenga siempre el mismo esquema en `fit`, en `GridSearchCV` (que
  vuelve a ajustar en cada fold) y en `predict`.
- **R-007.** `random_under` reduce la clase mayoritaria (o todas las que
  superen el tamaño de la minoritaria, si son varias) hasta el tamaño de la
  minoritaria, que es el comportamiento por defecto de `ClassBalancer` en Weka.
- **R-008.** `random_over` replica con muestreo con reemplazo las clases
  minoritarias hasta igualar la mayoritaria.
- **R-009.** `smote` usa `imblearn.over_sampling.SMOTE` y `smoten` usa
  `SMOTEN` (para variables categéricas), siempre con
  `random_state=random_state`.
- **R-010.** Si `imbalanced-learn` no está instalado y el usuario elige
  `smote` o `smoten`, `build_pipeline` lanza un `ValueError` en español
  indicando el paquete que falta; no se degrada silenciosamente a otro
  método.
- **R-011.** Con menos de 6 muestras de la clase minoritaria, `smote` y
  `smoten` lanzan un `ValueError` explicativo (es el límite de `k_neighbors`).
- **R-012.** `apply_class_weight(model, estimator)` devuelve el nombre del
  estimador con `class_weight="balanced"` si el modelo lo admite, o `None`
  si el estimador no acepta ese parámetro (KNN y DecisionTree lo ignoran), de
  forma que la UI pueda avisar.

### `core/model_trainer.py`

- **R-013.** `build_pipeline(..., balancing=None)` acepta
  `{"method": ..., "k_neighbors": ..., "random_state": ...}` y, si el método
  requiere remuestreo, inserta el paso `("balancer", BalancedSampler(...))`
  inmediatamente después de `("preprocessor", ...)`.
- **R-014.** Si el método es `class_weight`, `build_pipeline` no añade ningún
  paso y en su lugar Train/Test anulará el parámetro `class_weight="balanced"`
  al estimador; el paso es por tanto **preprocesador → modelo**, sin remuestreo.
- **R-015.** El orden de los pasos es siempre
  `preprocessor → [scaler de chi2] → [balancer] → [selector] → modelo`
  (coherente con SDD-002, R-005).
- **R-016.** El balanceo solo aplica a clasificación. Con
  `task_type="regression"` la UI no muestra el selector y `build_pipeline`
  lanza `ValueError` si recibe una estrategia de remuestreo.

### UI

- **R-017.** `ui/preprocess_tab.py` añade el grupo “Distribución de clases”
  (visible solo si `state.task_type == "classification"` y hay objetivo) con
  una tabla de tres columnas (`Clase`, `Instancias`, `%`) y una etiqueta
  resumen con el IR, el nivel y la entropía, usando `describe()` de R-001.
- **R-018.** La tabla de distribución se recalcula al cambiar el objetivo o al
  reprocesar los datos, y **no** al elegir la estrategia (esa va en
  Entrenamiento).
- **R-019.** `ui/train_tab.py` añade el combo “Balanceo de clases” con las
  etiquetas de `BALANCING_STRATEGIES`, con `none` por defecto, y el spin de
  vecinos `k` visible solo para `smote`/`smoten`.
- **R-020.** Si el modelo no admite `class_weight` (R-012), la UI muestra un
  aviso en español y no ofrece esa opción para ese modelo.
- **R-021.** Cuando el balanceo está activo, la etiqueta de resultados indica
  cuántas instancias se añadieron o eliminaron en entrenamiento, con los datos
  de `n_before` / `n_after` del sampler ajustado.

### Estado y persistencia

- **R-022.** `AppState` incorpora `balancing` (`str | None`) y
  `class_distribution` (`dict | None`), documentados en `core/state.py`.
- **R-023.** `BundleMetadata` incorpora `balancing` (dict con método,
  distribución original y distribución resultante) y `describe()` lo muestra.
- **R-024.** Un bundle sin clave `balancing` se carga sin error.

## 5. Invariantes de arquitectura

- **`clean_df` y `export_df` nunca se remuestrean.** El remuestreo ocurre
  dentro de `pipeline.fit`, es decir solo sobre las filas de entrenamiento.
  Esta es la invariante más importante de la spec.
- `core/balancing.py` no importa Qt ni `ui/` y no lee ficheros.
- La tabla de distribución se calcula en `core`, la UI solo la pinta.
- El dataset que el usuario exporta sigue siendo el dataset real, no el
  balanceado.

## 6. Criterios de aceptación

- **C-001.** Dado `y = [0]*90 + [1]*10`, cuando se calcula
  `class_distribution(y)`, entonces `imbalance_ratio == 9.0`,
  `level == "severo"`, `pct` de la clase mayoritaria es `90.0` y el total es
  `100`.
- **C-002.** Dado `y = [0]*5 + [1]*5`, entonces `imbalance_ratio == 1.0` y
  `level == "equilibrado"`.
- **C-003.** Dado un target de una sola clase, entonces
  `imbalance_ratio == 1.0` y no hay excepción.
- **C-004.** Dado `random_under` con el mismo `y` de C-001, cuando se ajusta
  `BalancedSampler`, entonces `n_after == 20` (10 + 10) y
  `classes_after` muestra `10` en cada clase.
- **C-005.** Dado `random_over` con `y` de C-001, cuando se ajusta el sampler,
  entonces `n_after == 180` y cada clase tiene `90`.
- **C-006.** Dado un dataset con columnas categóricas codificadas, cuando se
  ajusta `BalancedSampler`, entonces el sampler funciona sobre la matriz
  transformada y devuelve el mismo número de columnas.
- **C-007.** Dado un `BalancedSampler` ajustado, cuando se llama a
  `transform` con datos nuevos, entonces devuelve el `DataFrame` sin cambios y
  sin excepción.
- **C-008.** Dado `train_model` con `balancing={"method": "random_under"}`,
  entonces el tamaño del conjunto de prueba es el mismo que sin balanceo, lo
  que demuestra que no se remuestreó el dataset completo (invariante crítica).
- **C-009.** Dado `method="class_weight"` y `LogisticRegression`, entonces el
  estimador ajustado tiene `class_weight == "balanced"` y el pipeline no tiene
  paso `balancer`.
- **C-010.** Dado `method="class_weight"` y `KNeighborsClassifier`, entonces la
  UI/constructor avisa de que el modelo no admite pesos de clase y el
  entrenamiento continúa sin ese ajuste.
- **C-011.** Dado `method="smote"` con `imbalanced-learn` instalado, cuando se
  ajusta el sampler, entonces las clases minoritarias quedan igualadas y el
  modelo entrena con los datos sintéticos.
- **C-012.** Dado `method="smote"` sin `imbalanced-learn`, entonces el error
  menciona el paquete que falta; con la minoritaria por debajo de 6 muestras,
  el error explica el límite de vecinos.
- **C-013.** Dado `task_type="regression"` y una estrategia de remuestreo,
  entonces `build_pipeline` lanza `ValueError` en español.
- **C-014.** `export_df` tiene el mismo número de filas antes y después de
  entrenar con cualquier estrategia de balanceo.
- **C-015.** `pytest -q` en verde con al menos 10 tests nuevos en
  `tests/test_balancing.py`.

## 7. Dependencias

- `imbalanced-learn` — no instalado. Necesario únicamente para `smote` y
  `smoten`. Alternativa sin dependencia: implementar SMOTE a mano, descartada
  por riesgo de bugs numéricos.
- `scipy` para `scipy.stats.entropy`, ya instalado.

Los tests de `smote` se marcan con `pytest.importorskip("imblearn")`.

## 8. Riesgos

| Riesgo | Mitigación |
|---|---|
| Remuestrar por error el dataset completo y falsear métricas | Invariante de §5 + test explícito C-008 y C-014 |
| `BalancedSampler` rompe el esquema en `predict` | `transform` es identidad (R-006) + test C-007 |
| `class_weight` ignorado en KNN / DecisionTree | `apply_class_weight` devuelve `None` y la UI avisa (R-012, R-020) |
| SMOTE genera ruido en columnas one-hot | `smoten` específico para categóricas (R-009) y nota en la documentación |
| Coste de memoria al balancear | Se muestra el tamaño antes/después (R-021) y el usuario puede revertir a `none` |

## 9. Decisiones que se cierran en `design.md`

- Si el balanceo se aplica también en `tune_model` (recomendado: sí, porque
  `GridSearchCV` vuelve a ajustar el pipeline en cada fold) o solo en
  `train_model`.
- Si `SMOTENC` se añade para datos mixtos o se documenta `smote` para
  numéricas y `smoten` para categóricas puras.
- Dónde se muestra la tabla de distribución: `preprocess_tab` o también en
  `results_tab`.