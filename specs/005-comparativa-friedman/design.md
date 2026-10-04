# SDD-005 — Diseño técnico: comparativa con test de Friedman

Estado: **esperando puerta 2**. Última spec en ejecutarse.

## 1. Mapa de cambios

| Fichero | Cambio |
|---|---|
| `core/comparison.py` (nuevo) | `ModelComparison`, `compare_models`, `critical_difference`, `_holm` |
| `core/model_trainer.py` | Se extrae `roc_auc_value(pipe, X_test, y_test)` reutilizable y `block_score(...)` |
| `ui/workers.py` | `ComparisonWorker` |
| `ui/results_tab.py` | Grupo “Comparativa (Friedman)”, tabla y diagrama de diferencias críticas |
| `ui/train_tab.py` | La comparativa toma los modelos compatibles de `state` |
| `core/state.py` | `comparison`, `comparison_key` |
| `tests/test_comparison.py` (nuevo) | Tests del núcleo |

## 2. API resultante

```python
# core/comparison.py

@dataclass(frozen=True)
class ModelComparison:
    metric: str
    task_type: str
    scores: pd.DataFrame          # modelo, repeticion, fold, puntuacion
    ranking: pd.DataFrame         # modelo, puntuacion_media, desviacion, rango_medio, equivalente
    friedman_statistic: float
    friedman_p: float
    n_observaciones: int
    posthoc: pd.DataFrame         # modelo_a, modelo_b, p, p_ajustada, significativo
    critical_difference: float
    alpha: float
    mejor_modelo: str
    excluidos: dict               # modelo -> motivo en español

def compare_models(df, target, model_names, task_type, metric,
                   casts=None, normalizations=None, selection=None,
                   balancing=None, n_splits=5, n_repeats=3, random_state=42,
                   progress_callback=None) -> ModelComparison

def critical_difference(k: int, n: int, alpha: float = 0.05) -> float
```

### 2.1 Firma del progreso

La spec (R-014) propone `progress(hecho, total)`; el código existente usa
`progress_callback(etapa, actual, total)` (`core/model_trainer.py:283`) y la
señal Qt `progress = Signal(str, int, int)`. **Se adopta la convención de tres
argumentos** para reutilizar `TuneWorker`/`WorkerThread` sin inventar un segundo
protocolo. Queda anotado como refinamiento de R-014.

### 2.2 Bloques pareados

```python
cv = RepeatedStratifiedKFold(n_splits, n_repeats, random_state=random_state) \
     if classification else RepeatedKFold(...)
bloques = list(cv.split(X, y))          # b = n_splits * n_repeats bloques
```

Los mismos `bloques` se recorren para **todos** los modelos en el mismo orden:
es la condición de validez del test (fold *i* del modelo A y fold *i* del
modelo B son el mismo conjunto de filas).

`n_splits` se recorta antes de empezar a
`min(n_splits, min_class_count)` en clasificación (R-004); si el recorte
impediría tener 3 modelos comparables, se propaga el `ValueError`.

### 2.3 Puntuación por bloque

Se extrae de `model_trainer` un helper público para que la comparación y el
entrenamiento no dupliquen el cálculo del AUC:

```python
def roc_auc_value(pipe, X_test, y_test) -> tuple[float | None, str | None]
```

`_compute_metrics` pasa a usarla, y `compare_models` puntúa cada bloque con:

| Tarea | Métrica | Cálculo |
|---|---|---|
| clasificación | `accuracy` | `accuracy_score` |
| clasificación | `f1_macro` | `f1_score(average="macro")` |
| clasificación | `roc_auc` | `roc_auc_value(...)` |
| regresión | `rmse` / `mae` / `r2` | igual que `_compute_metrics` |

Un `roc_auc` no calculable en un bloque produce `NaN`. Si un modelo tiene algún
`NaN`, se **excluye** de la comparación y se anota en `excluidos` (no se puede
construir una matriz de rangos rectangular con `NaN`). Si sobreviven menos de 3
modelos → `ValueError` en español.

### 2.4 Rangos, Friedman y post-hoc

1. **Orientación**: `_ORIENT = {"rmse": -1, "mae": -1}`; se multiplica la
   puntuación por el signo para que “mayor es mejor” sea universal.
2. **Rangos por bloque**: `stats.rankdata(-orientadas, axis=None)` sobre la
   matriz `b × k` → el mejor de cada bloque recibe rango 1.
3. `friedman_statistic, friedman_p = stats.friedmanchisquare(*rangos_por_columna)`.
4. **Post-hoc**: para cada par, `diff = orientada_A − orientada_B` sobre los
   `b` bloques y `stats.wilcoxon(diff)` (si `b < 6`, `stats.ttest_rel` y la
   columna `test` indica cuál).
5. **Holm implementado a mano** (`_holm`), porque `statsmodels` no es una
   dependencia permitida:
   ```
   p_ordenadas = sorted(p)
   p_adj[i] = max(p_adj[i-1], (m - i) * p_ordenadas[i])   # step-down monotónico
   p_adj = min(p_adj, 1.0)
   ```
6. `significativo = p_ajustada < alpha`.
7. `critical_difference(k, b, alpha)` con el estadístico *q* de Nemenyi:
   `q = sqrt(2) * t.ppf(1 - alpha/2, df=b - 1)` y `CD = q / sqrt(k)`
   (aproximación estándar de Demšar, la misma que usa Orange).
8. `equivalente` en la tabla de ranking: `True` si
   `rango_medio - rango_del_mejor <= critical_difference` (R-010).

`mejor_modelo` = el de menor rango medio (desempate por nombre, para que sea
determinista).

### 2.5 Ejemplo de salida

`ranking`:

| modelo | puntuacion_media | desviacion | rango_medio | equivalente |
|---|---|---|---|---|
| Random Forest | 0.8421 | 0.0310 | 1.5 | sí |
| KNN | 0.8210 | 0.0344 | 2.0 | sí |
| SVM | 0.7990 | 0.0401 | 3.0 | no |
| Regresión Logística | 0.7610 | 0.0455 | 4.5 | no |

## 3. UI

### 3.1 `ui/results_tab.py`

Nuevo `gb_comparativa` debajo de la figura de métricas:

| Widget | Comportamiento |
|---|---|
| `cmb_metrica_comparativa` | `model_specs.available_metrics(task_type)` |
| `spn_repeticiones` | 1…10, valor por defecto 3 |
| `chk_modelos` | “Todos los modelos compatibles” (marcado) o usar solo `state.model_name` |
| `btn_comparar` | Lanza el worker |
| `progress` | `QProgressBar` reutilizado con su propio rango |
| `tbl_comparativa` | 5 columnas de `ranking` + `excluidos` en una línea aparte |
| `canvas_cd` | Segundo `FigureCanvasQTAgg` con el diagrama |

Diagrama de diferencias críticas (`_draw_cd_diagram(ax, comparison)`):

- eje x = rango medio, límites `[0.5, k + 0.5]` para que la barra quepa;
- una línea horizontal por modelo, de `rango_medio` a `rango_medio + CD`;
- una línea vertical punteada en el rango del mejor modelo, uniendo los modelos
  dentro de `CD` con ella;
- etiquetas con el nombre del modelo y su rango con dos decimales.

### 3.2 `ui/workers.py`

```python
class ComparisonWorker(QObject):
    finished = Signal(object)                 # ModelComparison
    failed = Signal(str)
    progress = Signal(str, int, int)

    def run(self):
        try:
            self.finished.emit(comparison.compare_models(
                progress_callback=self._reportar, **self._kwargs))
        except Exception as e:
            self.failed.emit(str(e))
```

Se reutiliza `WorkerThread` sin cambios (misma interfaz `run()`).

### 3.3 Progreso

`compare_models` emite tres fases: preparación, “Modelo X (i/n)” por cada ajuste
de bloque y cálculo de estadísticos. Total = `k * b` ajustes.

### 3.4 Caché e invalidación

`AppState.comparison` guarda el resultado y `AppState.comparison_key` la tupla
`(target, task_type, metric, n_splits, n_repeats, tuple(model_names),
selection, balancing)`. Si al volver a la pestaña la clave actual difiere de la
guardada, el grupo muestra un aviso: “Estos resultados son de otra configuración
(vuelve a comparar para actualizarlos)”, sin borrar el diagrama. Ambas se
limpian en `reset_data()` y `reset_model()`.

## 4. Qué modelos se comparan

Por defecto, `compatible_models(task_type, profiles, target, n_rows)` filtrado
por los que están `compatible` — los mismos que muestra la pestaña 3. Se
calcula en `core` para que la UI no reimplemente el filtro: la comparación toma
`model_names` como argumento, y el worker lo recibe ya listo desde la UI. Si el
usuario ha restringido la comparación a un modelo o a una familia, se respetan
los modelos visibles en el combo de modelo de la pestaña 3.

## 5. Rendimiento

`k = 5` modelos × `b = 15` bloques = 75 ajustes. Con `n_jobs` de los
estimadores (bosques con 200 árboles) puede tardar minutos; por eso es
obligatorio ir en worker (AGENTS §5.3) y el spin de repeticiones limita el
coste. En los tests se usan `n_repeats=2`, modelos pequeños y datasets de 200
filas para que la suite siga siendo rápida.

## 6. Verificación

```bash
python -m pytest -q tests/test_comparison.py
grep -R "PySide6\|PyQt\|from ui\." core/
```