# SDD-002 — Diseño técnico: selección de atributos

Estado: **esperando puerta 2**.

## 1. Mapa de cambios

| Fichero | Cambio |
|---|---|
| `core/feature_selection.py` (nuevo) | Catálogo, `available_methods`, `build_selector`, `describe_selection`, selectores seguros |
| `core/model_trainer.py` | `build_pipeline(..., selection=None)`, `train_model`/`tune_model` aceptan `selection`, nueva `analyze_selection` |
| `core/persistence.py` | Campo `BundleMetadata.selection`, `build_metadata(selection=...)`, `_upgrade_metadata` |
| `core/state.py` | `selection_method`, `selection_criteria` |
| `ui/train_tab.py` | Grupo “Selección de atributos” + botón “Analizar” |
| `ui/results_tab.py` | Lista de atributos seleccionados |
| `ui/workers.py` | `SelectionWorker` |
| `tests/test_feature_selection.py` (nuevo) | Tests del núcleo |

## 2. Corrección de la spec (detectada al leer el código)

`core/model_trainer.py:99` ya aplica `StandardScaler` a las columnas numéricas,
de modo que la redacción de **R-005** (“si no existe ya un paso de escalado”) es
inexacta: el paso existe, pero **no** garantiza valores no negativos. La regla
definitiva, que es la que implementan las tareas, es:

> Cuando el método es `chi2`, `build_pipeline` inserta **siempre** un paso
> `("scaler_nonneg", MinMaxScaler())` inmediatamente después de
> `("preprocessor", ...)`.

`MinMaxScaler` actúa sobre la matriz ya codificada (numéricas + one-hot) y la
lleva a `[0, 1]`, con lo que `chi2` recibe siempre datos no negativos sin tocar
`build_preprocessor`. Se actualiza el texto de R-005 en `spec.md` para que
spec y diseño digan lo mismo.

## 3. Orden canónico de pasos del pipeline

```
typer → normalizer → preprocessor → [scaler_nonneg (solo chi2)]
      → [balancer (SDD-004)] → [selector] → model
```

## 4. API resultante

```python
# core/feature_selection.py

@dataclass(frozen=True)
class SelectionMethod:
    id: str
    label: str                    # español
    supports_regression: bool
    requires_non_negative: bool
    supports_percentile: bool
    default_k: int
    notes: str

SELECTION_METHODS: dict[str, SelectionMethod]

def available_methods(task_type: str) -> list[str]
def build_selector(method, task_type, k=None, percentile=None, random_state=42)
def describe_selection(pipe, columns=None) -> pd.DataFrame
def selected_feature_names(pipe) -> list[str]      # nombres codificados
```

### 4.1 Catálogo

| `id` | `label` | regresión | no negativos | porcentaje | `default_k` |
|---|---|---|---|---|---|
| `chi2` | “Chi-cuadrado” | no | sí | sí | 10 |
| `anova` | “ANOVA F” | sí | no | sí | 10 |
| `mutual_info` | “Información mutua” | sí | no | sí | 10 |
| `embedded` | “Embebido (bosque aleatorio)” | sí | no | **no** | — |

`embedded` no admite porcentaje: `SelectFromModel` necesita conocer `n_features`
en el momento de construir el estimador interno, así que la UI deshabilita el
criterio de porcentaje cuando el método es `embedded`.

### 4.2 `build_selector`

```python
def build_selector(method, task_type, k=None, percentile=None, random_state=42):
    spec = SELECTION_METHODS.get(method)
    if spec is None:
        raise ValueError(f"Método de selección desconocido: {method!r}.")
    if task_type == REGRESSION and not spec.supports_regression:
        raise ValueError(
            f"El método {spec.label} no se puede usar en regresión."
        )
    if percentile is not None:
        if not spec.supports_percentile:
            raise ValueError(f"{spec.label} no admite el criterio por porcentaje.")
        return SelectPercentile(
            score_func=_score_func(method, task_type), percentile=percentile
        )
    k_efectivo = int(k) if k else spec.default_k
    if method == "embedded":
        return SafeSelectFromModel(
            estimator=_random_forest(task_type, random_state),
            max_features=k_efectivo, threshold=None,
        )
    return SafeSelectKBest(score_func=_score_func(method, task_type), k=k_efectivo)
```

`_score_func` devuelve `chi2`, `f_classif`, `f_regression`,
`mutual_info_classif` o `mutual_info_regression` según método y tarea; para
información mutua se fija `random_state` con `functools.partial`.

### 4.3 Selectores seguros (recorte de `k`)

`SelectKBest` y `SelectFromModel` avisan por `warnings` si `k > n_features` y
dejan el recorte a sklearn, lo que produce mensajes en inglés y un
comportamiento distinto según la versión. Se envuelven en dos clases de
primer nivel (necesario: el pipeline se serializa con `joblib` dentro del
bundle, así que las clases deben ser importables por nombre):

```python
class SafeSelectKBest(SelectKBest):
    """SelectKBest que recorta k al número de columnas disponibles."""
    def fit(self, X, y):
        self.k_effective_ = int(min(self.k, np.asarray(X).shape[1]))
        original = self.k
        self.k = self.k_effective_
        try:
            super().fit(X, y)
        finally:
            self.k = original
        return self

class SafeSelectFromModel(SelectFromModel):
    """SelectFromModel que recorta max_features y expone el umbral usado."""
    # mismo patrón, guardando `max_features_effective_` y `n_features_`
```

`SafeSelectFromModel` usa `threshold=None` y `max_features=k`, que es la forma
explícita e independiente de la versión de “quedarse con las k mejores”.

### 4.4 `describe_selection`

```python
def describe_selection(pipe, columns=None) -> pd.DataFrame
```

1. `names = list(pipe.named_steps["preprocessor"].get_feature_names_out())`.
2. `selector = pipe.named_steps["selector"]`; `soporte = selector.get_support()`.
3. `scores = getattr(selector, "scores_", None)`; si no existe,
   `getattr(estimator, "feature_importances_", None)`; si tampoco, `NaN`.
4. Mapa de origen: `feature_origin_map(preprocessor, names)` (§4.5).
5. Una fila por **atributo original**: `atributo`, `n_columnas_codificadas`,
   `puntuacion` (máximo de sus columnas), `seleccionado` (`any`).
6. Devuelve el `DataFrame` ordenado por `puntuacion` descendente, con el índice
   reiniciado.

Devuelve un `DataFrame` vacío con las mismas columnas si el pipeline no tiene
paso `selector`, para que la UI pueda llamarlo siempre.

### 4.5 Mapa de atributo original

`get_feature_names_out()` produce nombres como `num__edad` y
`cat__ciudad_Madrid`. El prefijo solo dice “rama”, no el atributo original: para
`cat__ciudad_Madrid` hay que quitar el valor codificado. La información está en
el `ColumnTransformer` ya ajustado, así que se reconstruye desde ahí:

```python
def feature_origin_map(preprocessor, names) -> list[str]:
    """Nombre del atributo original de cada columna transformada."""
    origins, cursor = [], 0
    for _name, trans, cols in preprocessor.transformers_:
        cols = [cols] if isinstance(cols, str) else list(cols)
        encoder = _last_encoder(trans)          # OneHotEncoder | None
        if encoder is None:                     # 1 salida por columna
            for col in cols:
                origins.append(col); cursor += 1
        else:
            for col, categoria in zip(cols, encoder.categories_):
                origins.extend([col] * len(categoria)); cursor += 1
    # Si la cuenta no cuadra (drop, handle_unknown raro), se cae al prefijo.
    if len(origins) != len(names):
        origins = [n.split("__", 1)[-1] for n in names]
    return origins
```

El recorrido es posicional y coincide con el orden de `get_feature_names_out()`.
Si el recuento no cuadra, el recurso alternativo (quitar el prefijo `rama__`)
degrada la etiqueta pero nunca falla.

## 5. Integración en `core/model_trainer.py`

```python
def build_pipeline(df, model_name, task_type, target, casts=None,
                   normalizations=None, selection=None, balancing=None):
    ...
    steps.append(("preprocessor", build_preprocessor(X)))
    if selection and selection.get("method") == "chi2":
        steps.append(("scaler_nonneg", MinMaxScaler()))
    if balancing:  ...   # SDD-004
    if selection and selection.get("method"):
        steps.append(("selector", feature_selection.build_selector(**selection)))
    steps.append(("model", model_specs.build_estimator(model_name, task_type)))
```

`train_model` y `tune_model` aceptan `selection=None` y lo pasan a
`build_pipeline`. Con `selection=None` la lista de pasos es idéntica a la
actual (test de regresión C-008).

### 5.1 `analyze_selection` (evita importación circular)

`core/feature_selection.py` importa solo scikit-learn. La función que necesita
`build_pipeline` vive en `model_trainer.py`:

```python
def analyze_selection(df, target, task_type, selection, casts=None,
                      normalizations=None, random_state=42, test_size=0.2):
    """Ajusta el selector sobre el split de entrenamiento y describe el
    resultado. No toca `AppState` ni entrena el modelo final."""
    X, y = df.drop(columns=[target]), df[target]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y if task_type == CLASSIFICATION else None,
    )
    pipe = build_pipeline(df, "Regresión Logística" if task_type == CLASSIFICATION
                          else "Regresión Lineal", task_type, target,
                          casts=casts, normalizations=normalizations,
                          selection=selection)
    pipe.fit(X_train, y_train)
    return feature_selection.describe_selection(pipe)
```

El estimador concreto es irrelevante porque el selector es el penúltimo paso:
se usa el más barato de cada tarea (el propio diseño lo documenta).

## 6. Persistencia

```python
# BundleMetadata
selection: dict = field(default_factory=dict)
```

`build_metadata(..., selection=None)` guarda
`{"method": ..., "k": ..., "percentile": ..., "selected": [...]}`, donde
`selected` es la lista de columnas codificadas que quedaron activas (útil para
auditar sin volver a mirar el pipeline).

`describe()` añade una línea “Selección: Chi-cuadrado (12 de 87 columnas)” si
hay método.

### 6.1 Compatibilidad con bundles antiguos (regla P1)

Los `.automl` anteriores guardan un `BundleMetadata` **sin** los campos nuevos:
al cargarlos, el objeto restaurado por `joblib` no tiene `.selection`, y
`describe()` lanzaría `AttributeError`. Para evitarlo en cada sitio, se
implementa la migración en un punto único:

```python
def _upgrade_metadata(metadata: BundleMetadata) -> BundleMetadata:
    """Rellena los campos añadidos después de crear el bundle."""
    datos = asdict(metadata)
    for campo, valor in _CAMPOS_POR_DEFECTO.items():
        datos.setdefault(campo, valor)
    return BundleMetadata(**datos)
```

`_CAMPOS_POR_DEFECTO` es el único sitio donde se_declaran los valores por
defecto de los campos nuevos (`selection={}`, `balancing={}`), de modo que
añadir un campo en el futuro es una línea. `load_bundle` aplica la migración
justo después de `joblib.load` y `BUNDLE_VERSION` sigue siendo `1`.

## 7. UI

### 7.1 `ui/train_tab.py`

Nuevo `gb_seleccion` (dentro de `gb_tune`? no: grupo hermano, porque la
selección aplica también al entrenamiento simple) con:

| Widget | Tipo | Comportamiento |
|---|---|---|
| `chk_seleccion` | `QCheckBox` | “Seleccionar atributos automáticamente” (desmarcado por defecto) |
| `cmb_metodo` | `QComboBox` | `feature_selection.available_methods(task_type)` |
| `cmb_criterio` | `QComboBox` | “Número de atributos (k)” / “Porcentaje” |
| `spn_valor` | `QSpinBox` | Rango 1…2000; se ajusta al cambiar de criterio |
| `btn_analizar` | `QPushButton` | “Analizar selección” (worker) |
| `tbl_seleccion` | `QTableWidget` | 4 columnas de `describe_selection` |
| `lbl_seleccion` | `QLabel` | “Sin analizar todavía” / resumen |

- `_refresh_seleccion()` repuebla el combo cuando cambia la tarea; con
  regresión desaparece `chi2`; con `embedded` se deshabilita el criterio de
  porcentaje.
- `_selection_config() -> dict | None` devuelve `None` si el checkbox está
  desmarcado, y el dict que espera `build_selector` si está marcado.
- `btn_analizar` lanza `SelectionWorker` + `WorkerThread`; los controles se
  bloquean con `_set_busy(True)` y se liberan en `_on_thread_done`.
- Al entrenar, `selection` viaja en `train_model`/`TuneWorker`; se guarda en
  `self._run_context` para poder describirlo al terminar, igual que ya se hace
  con target/tarea/modelo.

### 7.2 `ui/workers.py`

```python
class SelectionWorker(QObject):
    finished = Signal(object)   # DataFrame de describe_selection
    failed = Signal(str)
    def __init__(self, df, target, task_type, selection, casts, normalizations): ...
    def run(self): ...
```

Reutiliza `WorkerThread` sin cambios: la interfaz es `run()` + `finished` +
`failed`.

### 7.3 `ui/results_tab.py`

`refresh()` añade una línea al resumen:
`Selección: Chi-cuadrado → 12 de 87 columnas (num__edad, cat__ciudad_Madrid…)`
usando `feature_selection.describe_selection(pipe)` sobre
`state.pipeline`. Si no hay selector, no se muestra nada.

## 8. Estado

```python
# core/state.py
selection_method: Optional[str] = None
selection_criteria: Optional[dict] = None
```

Se incluyen en `reset_data()` y `reset_model()`.

## 9. Dependencias

Ninguna. Test de `embedded` marcado como lento por usar un bosque de 100
árboles; en los tests se puede bajar `n_estimators` mediante un parámetro
`n_estimators` del catálogo (por defecto 100).

## 10. Verificación

```bash
python -m pytest -q tests/test_feature_selection.py
grep -R "PySide6\|PyQt\|from ui\." core/
```