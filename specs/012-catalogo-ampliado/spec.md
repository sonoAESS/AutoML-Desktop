# 012 — Catálogo ampliado

**Titular:** Añadir modelos nuevos a cada familia establecida y crear tres
familias nuevas, todo con scikit-learn 1.9 ya instalado.

**Orden:** 2 · **Depende de:** 011 · **Rama:** `design/redesign-ui`

---

## Contexto

El catálogo actual tiene 5 clasificadores y 6 regresores en 5 familias. El
usuario pidió «adicionar mas modelos a cada tipo de modelo establecido». La
elección es **solo scikit-learn**: xgboost y lightgbm no están instalados y
serían dependencias nuevas (AGENTS §7.3).

Todos los modelos candidatos se verificaron contra sklearn 1.9.1 instalado en
`entorno/`: qué parámetros aceptan, si soportan `class_weight`, si exponen
`random_state` y si tienen `predict_proba`.

## Requisitos

### R-001 · Modelos nuevos por familia

**Familias existentes:**

| Familia | Clasificación nueva | Regresión nueva |
|---|---|---|
| `lineal` | `SGD` | `Lasso`, `ElasticNet`, `Huber`, `TheilSen` |
| `ensemble` | `ExtraTrees`, `Bagging`, `AdaBoost`, `GradientBoosting`, `HistGradientBoosting` | los mismos cinco |
| `vecinos` | `RadiusNeighbors` | `RadiusNeighbors` |
| `svm` | `LinearSVC` | `KernelRidge` |

**Familias nuevas:**

| Familia | Etiqueta | Clasificación | Regresión |
|---|---|---|---|
| `naive_bayes` | Naive Bayes | `GaussianNB`, `BernoulliNB` | — |
| `lda` | Análisis discriminante | `LinearDiscriminant`, `QuadraticDiscriminant` | — |

### R-002 · `class_weight_path` solo donde funciona

El campo `class_weight_path` se informa **solo** en modelos que aceptan
`class_weight`. Verificado en sklearn 1.9.1:

| Modelo | `class_weight` | `class_weight_path` |
|---|---|---|
| `HistGradientBoosting` (clasif.) | sí | `model__class_weight` |
| `ExtraTrees` | sí | `model__class_weight` |
| `SGD` (clasif.) | sí | `model__class_weight` |
| `LinearSVC` | sí | `model__class_weight` |
| `Bagging`, `AdaBoost`, `GradientBoosting` | no | — |
| `GaussianNB`, `BernoulliNB`, `LDA`, `QDA` | no | — |
| Todos los regresores | no aplica | — |

### R-003 · `MIN_ROWS` por familia nueva

| Familia | Mínimo de filas | Motivo |
|---|---|---|
| `naive_bayes` | 10 | Estimación de probabilidades con muy pocas filas es inestable. |
| `lda` | 20 | La matriz de covarianza necesita suficientes muestras por clase. |

Las familias existentes conservan sus umbrales (`svm`: 50, `vecinos`: 30).

### R-004 · Grids de hiperparámetros acotados

Cada modelo nuevo define un `ParamSpec` por parámetro relevante. El grid
completo de cada modelo no supera las **500 combinaciones** para que
`search_size` siga siendo usable y `RandomizedSearchCV` tenga sentido.

Parámetros por modelo (resumen):

- **HistGradientBoosting**: `learning_rate`, `max_iter`, `max_depth`,
  `min_samples_leaf`, `l2_regularization`, `class_weight` (solo clasif.).
- **ExtraTrees**: `n_estimators`, `max_depth`, `min_samples_split`,
  `max_features`, `class_weight` (solo clasif.).
- **Bagging**: `n_estimators`, `max_samples`, `max_features`.
- **AdaBoost**: `n_estimators`, `learning_rate`, `loss` (solo regres.).
- **GradientBoosting**: `n_estimators`, `learning_rate`, `max_depth`,
  `subsample`.
- **LinearDiscriminant**: `solver`, `shrinkage`.
- **QuadraticDiscriminant**: `reg_param`.
- **GaussianNB**: `var_smoothing`.
- **BernoulliNB**: `alpha`, `binarize`.
- **SGD**: `loss`, `penalty`, `alpha`, `class_weight` (solo clasif.).
- **LinearSVC**: `C`, `class_weight`.
- **RadiusNeighbors**: `radius`, `weights`.
- **SGD** (regresión): `loss`, `penalty`, `alpha`.
- **Lasso**: `alpha`, `fit_intercept`.
- **ElasticNet**: `alpha`, `l1_ratio`, `fit_intercept`.
- **Huber**: `epsilon`, `alpha`, `max_iter`.
- **TheilSen**: `fit_intercept`, `max_iter`.
- **KernelRidge**: `alpha`, `kernel`, `gamma`.

### R-005 · `random_state=42` en todos los modelos que lo aceptan

Todos los modelos con parámetro `random_state` lo fijan a `42` en la factory,
igual que los modelos existentes. Los que no lo aceptan (`GaussianNB`,
`BernoulliNB`, `LDA`, `QDA`, `RadiusNeighbors`, `Huber`, `KernelRidge`) son
deterministas por naturaleza.

### R-006 · Limpieza de metadatos muertos

- `supports_probability` está declarado y asignado 5 veces, pero **no se lee
  en ningún sitio** de `core/`, `ui/` ni `tests/`. La disponibilidad de AUC se
  decide en runtime con `hasattr(pipe, "predict_proba")`. Se elimina el campo.
- `RANKING_METRIC` y `metrics_for_ranking()` solo se usan en un test. Se
  eliminan del catálogo.

## Invariantes que no se rompen

| Invariante | Origen |
|---|---|
| `core/` no importa PySide6, PyQt ni `ui/` | `AGENTS.md` §3.1 |
| La UI no contiene lógica científica | `AGENTS.md` §3.5 |
| Toda función nueva de `core/` tiene test en `tests/` | `AGENTS.md` §7.5 |
| No se añaden dependencias sin justificar en la spec | `AGENTS.md` §7.3 |
| Los mensajes al usuario están en español | `AGENTS.md` §4 |

## Criterios de aceptación

| ID | Criterio |
|---|---|
| C-001 | `model_specs.list_models("classification")` devuelve 17 modelos y `list_models("regression")` devuelve 18. |
| C-002 | `model_specs.list_families()` devuelve 7 familias con sus etiquetas en español. |
| C-003 | `class_weight_path(nombre, "classification")` devuelve una ruta válida para `HistGradientBoosting`, `ExtraTrees`, `SGD` y `LinearSVC`, y `None` para el resto. |
| C-004 | `MIN_ROWS` contiene `{"svm": 50, "vecinos": 30, "naive_bayes": 10, "lda": 20}`. |
| C-005 | Para cada modelo nuevo, `spec.grid()` produce un diccionario con claves `model__*` y el producto de los valores no supera 500 combinaciones. |
| C-006 | `ModelSpec` ya no tiene el campo `supports_probability` y `model_specs` no exporta `RANKING_METRIC` ni `metrics_for_ranking`. |
| C-007 | `build_estimator(nombre, task)` devuelve una instancia válida para cada modelo nuevo, con `random_state=42` cuando aplica. |
| C-008 | `evaluate_compatibility` clasifica correctamente un dataset con 15 filas: `lda` aparece como no compatible con su motivo. |
| C-009 | Los tests existentes de `test_model_specs.py` y `test_model_trainer.py` siguen pasar sin cambios de API. |

## No incluido

- xgboost, lightgbm ni catboost: dependencias nuevas no instaladas.
- MLP (redes neuronales): descartado a petición del usuario.
- Cambiar los modelos existentes ni sus grids.
- Añadir `CalibratedClassifierCV` a `LinearSVC` (ya tiene `class_weight` y
  `decision_function`; el AUC se calcula igual que en SVM).
- Tocar `ui/train_tab.py` ni `ui/results_tab.py` (la UI consume el catálogo
  dinámicamente).
