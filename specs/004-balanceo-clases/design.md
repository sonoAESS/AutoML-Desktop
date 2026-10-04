# SDD-004 — Diseño técnico: balanceo de clases

Estado: **esperando puerta 2**.

## 1. Mapa de cambios

| Fichero | Cambio |
|---|---|
| `core/balancing.py` (nuevo) | `ClassDistribution`, `class_distribution`, `BALANCING_STRATEGIES`, `BalancedSampler`, `available_strategies` |
| `core/model_specs.py` | Campo `class_weight_path` en `ModelSpec` + `class_weight_path(model_name, task_type)` |
| `core/model_trainer.py` | `build_pipeline(..., balancing=None)`, `train_model`/`tune_model` con `balancing`, poids de clase en el grid |
| `core/persistence.py` | `BundleMetadata.balancing`, `build_metadata(balancing=...)`, `_CAMPOS_POR_DEFECTO` |
| `core/state.py` | `balancing`, `class_distribution` |
| `ui/preprocess_tab.py` | Grupo “Distribución de clases” |
| `ui/train_tab.py` | Combo de estrategia + `config_changed` |
| `ui/main_window.py` | Conecta `train_tab.config_changed` con `preprocess_tab.refresh` |
| `tests/test_balancing.py` (nuevo) | Tests del núcleo |

## 2. `core/balancing.py`

```python
IR_NIVELES = ((1.5, "equilibrado"), (3.0, "moderado"))   # (umbral, etiqueta)

@dataclass(frozen=True)
class ClassCount:
    clase: Any
    n: int
    pct: float

@dataclass(frozen=True)
class ClassDistribution:
    rows: tuple[ClassCount, ...]
    total: int
    n_classes: int
    majority: Any
    minority: Any
    majority_n: int
    minority_n: int
    imbalance_ratio: float
    shannon_entropy: float
    normalized_entropy: float
    level: str

    def as_dict(self) -> dict          # para AppState y el bundle
    def describe(self) -> str          # una línea en español

@dataclass(frozen=True)
class BalancingStrategy:
    id: str
    label: str
    description: str
    needs_resampling: bool
    requires_smote: bool

BALANCING_STRATEGIES: dict[str, BalancingStrategy]

def class_distribution(y) -> ClassDistribution
def available_strategies(task_type: str, model_name: str | None = None) -> list[str]
class BalancedSampler(BaseEstimator)
```

### 2.1 `class_distribution`

- `rows`: ordered por `n` descendente y, a igualdad, por `str(clase)` — así el
  resultado es determinista y la tabla no “salta” entre refrescos.
- `imbalance_ratio = majority_n / minority_n`; con una sola clase, `1.0`.
- Entropía: `scipy.stats.entropy(counts, base=2)` y
  `normalized_entropy = shannon / log2(n_classes)`, con `0.0` si
  `n_classes < 2`.
- `level`: `equilibrado` si IR < 1.5, `moderado` si < 3.0, `severo` si no.
- `describe()`: “3 clases · 1 000 instancias · IR = 9,00 (n_mayor/n_minor) ·
  desbalance severo · entropía normalizada 0,53”.

### 2.2 `BalancedSampler`

```python
class BalancedSampler(BaseEstimator):
    """Remuestrea el conjunto de ENTRENAMIENTO dentro del pipeline."""

    def __init__(self, method="random_under", k_neighbors=5, random_state=42):
        self.method = method
        self.k_neighbors = k_neighbors
        self.random_state = random_state

    # Persistencia para la UI y el bundle:
    n_before = None; n_after = None
    classes_before = None; classes_after = None

    def fit(self, X, y=None): ...          # guarda tamaños y clases; no remuestrea
    def fit_transform(self, X, y=None, **fit_params): ...   # remuestrea y devuelve (X', y')
    def transform(self, X): ...            # identidad: el esquema no cambia
```

Decisiones de diseño:

1. **`fit` no remuestrea; `fit_transform` sí.** Es más explícito que apoyarse en
   el `fit_transform` por defecto de `TransformerMixin` y evita sorpresas con
   `_SetOutputMixin`. En el pipeline, `fit` + `transform` solo ocurren en
   `predict`, donde debe ser identidad.
2. **El remuestreo se hace con índices**, no con concatenación: se calcula un
   array `idx` y se devuelve `X[idx], y[idx]`. Funciona igual con `ndarray`,
   `DataFrame` y matriz dispersa de scipy, que es lo que produce
   `ColumnTransformer` con `OneHotEncoder` (sparse por defecto). El tipo de
   salida es el mismo que el de entrada, así que los pasos siguientes (selector,
   SVM, KNN) siguen funcionando.
3. **Regresión prohibida**: `fit`/`fit_transform` lanzan `ValueError` en
   español si `task_type != "classification"`; el task_type se pasa como
   parámetro del sampler en `build_pipeline` para que la invariante se cumpla
   también dentro del núcleo.
4. `method="none"` devuelve `(X, y)` sin tocar nada y sirve como no-op explícito.
5. `random_under` baja **todas** las clases por encima del tamaño de la minoritaria
   hasta ese tamaño (R-007), que es lo que hace `ClassBalancer` de Weka.
6. `random_over` sube las minoritarias hasta el tamaño de la mayoritaria con
   reemplazo, usando `np.random.default_rng(self.random_state)`.
7. `smote` / `smoten` delegan en `imblearn.over_sampling.SMOTE` /
   `SMOTENC`. Con entrada dispersa, `SMOTEN` no acepta disperso, así que se
   convierte con `.toarray()`. Si `imblearn` no está instalado → `ValueError`
   en español indicando el paquete (R-010). Si la minoritaria tiene menos de 6
   filas → `ValueError` explicando el límite de vecinos (R-011).
8. `get_params`/`set_params` vienen de `BaseEstimator` a partir de la firma de
   `__init__`; `clone()` funciona y `joblib` puede pickled la clase (de primer
   nivel, sin lambdas).
9. `available_strategies(task_type, model_name)`: en regresión devuelve solo
   `["none"]`; en clasificación devuelve las seis, pero quita `class_weight` si
   `model_specs.class_weight_path(model_name, task_type)` es `None` (R-020).

### 2.3 Pesos de clase en el catálogo

`ModelSpec` gana un campo declarativo `class_weight_path: Optional[str] = None`:

| Modelo | `class_weight_path` |
|---|---|
| Regresión Logística | `model__class_weight` |
| Árbol de Decisión | `model__class_weight` |
| Random Forest | `model__class_weight` |
| SVM (`CalibratedClassifierCV(SVC)`) | `model__estimator__class_weight` |
| KNN | `None` (no lo admite) |

`model_specs.class_weight_path(model_name, task_type)` devuelve el valor o
`None`. Se declara en el catálogo en vez de en un `if` de `balancing.py` para
que el dato siga el patrón declarativo de `model_specs.py` (AGENTS §3.4).

## 3. Integración en `core/model_trainer.py`

```python
def build_pipeline(df, model_name, task_type, target, casts=None,
                   normalizations=None, selection=None, balancing=None):
    ...
    steps.append(("preprocessor", build_preprocessor(X)))
    if _needs_scaler_nonneg(selection):
        steps.append(("scaler_nonneg", MinMaxScaler()))
    if balancing and balancing.get("method") not in (None, "none", "class_weight"):
        steps.append(("balancer", BalancedSampler(
            method=balancing["method"],
            k_neighbors=balancing.get("k_neighbors", 5),
            random_state=balancing.get("random_state", 42),
            task_type=task_type,
        )))
    if selection and selection.get("method"):
        steps.append(("selector", feature_selection.build_selector(**selection)))
    pipe = Pipeline(steps + [("model", model_specs.build_estimator(model_name, task_type))])

    if balancing and balancing.get("method") == "class_weight":
        path = model_specs.class_weight_path(model_name, task_type)
        if path is None:
            raise ValueError(
                f"El modelo {model_name} no admite pesos de clase. "
                "Usa submuestreo, sobremuestreo o SMOTE."
            )
        pipe.set_params(**{path: "balanced"})
    return pipe
```

`task_type` entra en el sampler para que R-016 se cumpla también dentro de
`core` (lanzar el error desde `BalancedSampler` y no solo desde la UI).

`tune_model`: si el método es `class_weight`, se fusiona en el grid

```python
param_grid = {**param_grid, path: ["balanced"]}
```

antes de construir `GridSearchCV`/`RandomizedSearchCV`. Es preferible a
`set_params` post-hoc sobre `best_estimator_` porque deja la decisión dentro de
la búsqueda y aparece en `best_params`. El control de hiperparámetro
correspondiente se oculta en la UI mientras el balanceo esté activo
(`_build_param_widgets` lo salta) para que no haya dos fuentes de verdad.

`train_model` y `tune_model` aceptan `balancing=None` y lo propagan.

### 3.1 Invariante de no fuga: por qué es segura

`train_model` hace `train_test_split` **antes** de `pipe.fit`, y el paso
`balancer` vive dentro de `pipe`. Por tanto el remuestreo solo ve filas de
entrenamiento, y el conjunto de prueba mantiene su tamaño y su distribución
original. `export_df` se construye con `transform_with_pipeline`, que solo
aplica los pasos `DATASET_STEPS = ("typer", "normalizer")` (`core/model_trainer.py:140`),
por lo que el dataset exportado nunca está balanceado. Los tests C-008 y C-014
comprueban las dos mitades de la invariante.

## 4. Persistencia y estado

```python
# BundleMetadata
balancing: dict = field(default_factory=dict)
```

`build_metadata(..., balancing=None)` guarda
`{"method", "k_neighbors", "n_before", "n_after", "classes_before", "classes_after"}`,
tomando los tres últimos del paso `balancer` ya ajustado si existe (o de los
estadísticas guardados por `class_weight`, en cuyo caso no hay remuestreo y se
guarda `"method": "class_weight"` con la distribución original). `describe()`
añade “Balanceo: submuestreo (1 000 → 200 instancias)”.

La migración de bundles antiguos es la misma regla P1 de SDD-002:
`_CAMPOS_POR_DEFECTO` incluye `balancing` y `load_bundle` aplica
`_upgrade_metadata`.

```python
# core/state.py
balancing: Optional[str] = None          # id de la estrategia elegida
class_distribution: Optional[dict] = None
```
Ambos se limpian en `reset_data()` y `reset_model()`.

## 5. UI

### 5.1 `ui/preprocess_tab.py` — “Distribución de clases”

Nuevo `gb_distribucion`, oculto salvo que
`state.task_type == "classification"` y exista `state.target_column`:

- `tbl_distribucion` (`QTableWidget`): columnas `Clase`, `Instancias`, `%`.
- `lbl_distribucion`: `ClassDistribution.describe()`.

`refresh()` recalcula con `balancing.class_distribution(
self.state.clean_df[state.target_column])`, guarda el dict en
`state.class_distribution` y rellena la tabla. Si el dataset está desbalanceado
se aplica una fuente negrita a la etiqueta (sin `QMessageBox`: es información,
no un error).

El cálculo se hace sobre `clean_df` (no sobre `raw_df`) porque es el dataset
que se va a entrenar.

### 5.2 `ui/train_tab.py` — estrategia

- `cmb_balancing` con `balancing.available_strategies(task, model_name)`.
- `spn_vecinos` (`QSpinBox` 2…10) visible solo para `smote`/`smoten`.
- `_balancing_config() -> dict | None` devuelve `None` para `none`, si no el dict
  que espera `build_pipeline`.
- Se propaga en `train_model`, en `TuneWorker` y en `self._run_context`.
- Nueva señal `config_changed = Signal()`, emitida en `_on_task_changed` y al
  cambiar el objetivo, para que `MainWindow` refresque la tabla de
  distribución.
- `_build_param_widgets` omite el `ParamSpec` de `class_weight` mientras el
  balanceo sea `class_weight`.

### 5.3 `ui/main_window.py`

```python
self.train_tab.config_changed.connect(self.preprocess_tab.refresh)
```

### 5.4 `ui/results_tab.py`

`save_bundle` pasa `balancing=...` a `build_metadata`, y `refresh()` añade una
línea con el efecto del balanceo (`n_before → n_after`) leyéndolo del paso
`balancer` de `state.pipeline`.

## 6. Dependencias

`imbalanced-learn`. Se declara como extra opcional `smote` en `pyproject.toml`
y se instala en `entorno/` con confirmación del usuario (T-001 de 001 ya pide
instalar dependencias; esta se añade a esa tanda). Los tests de SMOTE usan
`pytest.importorskip("imblearn")`, de forma que la suite es verde sin él.

## 7. Verificación

```bash
python -m pytest -q tests/test_balancing.py
grep -R "PySide6\|PyQt\|from ui\." core/
```