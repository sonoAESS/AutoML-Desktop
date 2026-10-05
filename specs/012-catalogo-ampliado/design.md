# 012 — Catálogo ampliado · Diseño

**Spec:** [spec.md](spec.md) · **Orden:** 2 · **Depende de:** 011

---

## D1 · FAMILIES y MIN_ROWS

`FAMILIES` pasa de 5 a 7 entradas. El orden de inserción es el que ve el
desplegable de la barra lateral, así que las nuevas van al final:

```python
FAMILIES = {
    "lineal": "Modelos lineales",
    "arbol": "Árboles de decisión",
    "ensemble": "Conjuntos de modelos",
    "svm": "Máquinas de soporte vectorial",
    "vecinos": "Vecinos más cercanos",
    "naive_bayes": "Naive Bayes",
    "lda": "Análisis discriminante",
}

MIN_ROWS = {"svm": 50, "vecinos": 30, "naive_bayes": 10, "lda": 20}
```

`list_families()` (L335) ya itera `FAMILIES` y filtra por los specs presentes,
así que no hay que tocar nada: añadir entradas basta.

## D2 · Param groups compartidos

`_TREE_PARAMS`, `_FOREST_PARAMS` y `_KNN_PARAMS` ya existen. Los reutilizo donde
el parámetro y los valores sirven, para no duplicar tuplas:

| Grupo | Se reutiliza en |
|---|---|
| `_TREE_PARAMS` | `ExtraTrees` (además de `n_estimators`, `max_features`, `class_weight`) |
| `_FOREST_PARAMS` | `ExtraTrees` comparte `n_estimators`, `max_depth`, `min_samples_split`, `max_features` |

`Bagging` no usa `n_estimators` igual: su grid propio es `n_estimators`,
`max_samples`, `max_features` con valores de fracción.

## D3 · Grids acotados a ≤500 combinaciones

Criterio: el producto de las longitudes de los valores no pasa de 500. Ejemplos
del límite:

| Modelo | Producto | Justificación |
|---|---|---|
| `GradientBoosting` | 4×4×3×2 = 96 | holgado |
| `SGD` | 3×3×4×2 = 72 | holgado |
| `HistGradientBoosting` (clasificación) | 3×3×4×2×3×2 = 432 | es el más caro del catálogo |
| `HistGradientBoosting` (regresión) | 3×3×4×2×3 = 216 | sin `class_weight` |
| `ExtraTrees` | 3×3×3×2×2 = 108 | holgado |
| `Huber` | 4×4×3 = 48 | holgado |
| `TheilSen` | 2×3 = 6 | `max_iter` es solo 3 valores porque es lento |
| `KernelRidge` | 3×3×3 = 27 | holgado |

El total del catálogo son 1620 combinaciones. El límite de 500 no aprieta a
ningún modelo, pero el test lo fija para que nadie añada un grid de 1000 por
descuido sin darse cuenta.

### D3.1 · Un grid que no falla es peor que un grid que falla

Un modelo con un grid inválido no lanza ningún error: `GridSearchCV` captura el
fallo por combinación, la puntuación acaba siendo `NaN` y la búsqueda elige
sobre resultados incompletos, en silencio. Por eso `tests/test_model_specs.py`
recorre **las 1620 combinaciones del catálogo** y ajusta cada una
(`test_todas_las_combinaciones_del_grid_son_validas`, marcado `slow`).

Ese test encontró tres grids rotos que esta spec había dado por buenos:

1. **`LinearDiscriminant`**: `solver` × `shrinkage` como producto cartesiano.
   `svd` no acepta `shrinkage` y lanza `NotImplementedError`, así que 2 de las
   6 combinaciones fallaban siempre. Solución: `solver="lsqr"` fijo en el
   factory y `shrinkage` como único parámetro del grid.
2. **`KernelRidge`**: `gamma` con `("scale", "auto")`, que son valores de
   `SVC`. En `KernelRidge` `gamma` es un `float` o `None`; `"scale"` produce
   `InvalidParameterError` en las 6 combinaciones del modelo.
3. **`LinearSVC`**: `hinge` con `C=100` no converge ni con 10000 iteraciones
   sobre datos estándar, así que el modelo se entregaba a medio entrenar. Se
   fija `squared_hinge` y `loss` sale del grid.

## D4 · `hidden_layer_sizes` fuera (MLP descartado)

Sin MLP no hay tuplas anidadas en los grids, que es lo que hacía el `as_grid`
más frágil. Todos los valores son escalares o cadenas.

## D5 · Limpieza de metadatos muertos

Dos cosas se eliminan, y ambas están verificadas como no leídas:

- `ModelSpec.supports_probability`: el campo está en L119 y asignado 5 veces;
  `grep` no encuentra ninguna lectura. La disponibilidad del AUC la decide
  `model_trainer._auc_scores` con `hasattr(pipe, "predict_proba")`.
- `RANKING_METRIC` (L60) y `metrics_for_ranking()` (L371): solo los referencia
  `tests/test_model_trainer.py:538-540`. Ese test se borra.

**Por qué ahora y no antes:** al añadir 24 modelos, un campo que dice «este
modelo da probabilidades» y que nadie consulta pasa de ser ruido a ser un
riesgo: alguien lo leería creyendo que es la fuente de verdad del AUC.

## D6 · Modelos con `random_state=42`

Factories deterministas, igual que las existentes:

```python
"ExtraTrees": ModelSpec(..., factory=lambda: ExtraTreesClassifier(n_estimators=200, random_state=42)),
"Bagging":   ModelSpec(..., factory=lambda: BaggingClassifier(n_estimators=20, random_state=42)),
"AdaBoost":  ModelSpec(..., factory=lambda: AdaBoostClassifier(n_estimators=100, random_state=42)),
...
```

`GaussianNB`, `BernoulliNB`, `LDA`, `QDA`, `RadiusNeighbors`, `Huber` y
`KernelRidge` no aceptan el parámetro: no se les pasa.

`LinearSVC` sí lo acepta y necesita dos ajustes, ambos verificados
ajustando el modelo: `max_iter=3000` y `loss="squared_hinge"`. Con el valor por
defecto de 1000 iteraciones avisa de convergencia en cuanto las variables no
están escaladas, y el preprocesador no escala cuando el usuario elige
«ninguna»; con `hinge`, que es la `loss` por defecto de scikit-learn, tampoco
converge a `C=100`.

## D7 · `RadiusNeighbors` necesita un radio por defecto mayor

El `radius` por defecto de scikit-learn (1.0) está pensado para datos ya
escalados. Aquí el preprocesador no escala salvo que se elija `chi2`, así que
hay muestras que se quedan sin ningún vecino dentro del radio y sus
predicciones salen como `NaN`. Con `radius=5.0` el modelo responde bien sobre
iris y diabetes, que es lo que verifica `test_cada_regresor_nuevo_se_entrena`.

El grid usa `(2.0, 5.0, 10.0)` por el mismo motivo.

## D8 · Lo que queda fuera: escalado automático

`build_pipeline` solo añade un escalador cuando el método de selección es
`chi2`. Los modelos lineales, los SVM y los de vecinos se ajustan, por tanto,
sobre variables en su escala original, y eso penaliza a `SGD`, `LinearSVC`,
`KernelRidge`, `RadiusNeighbors`, `Lasso`, `ElasticNet`, `Huber` y `TheilSen`.

No se toca aquí: cambiar el preprocesamiento altera las métricas de los
modelos que ya existen y pertenece a su propia spec
(`013-...`, pendiente de numerar). Queda anotado aquí para que no se pierda.

## Ficheros

| Fichero | Cambio |
|---|---|
| `core/model_specs.py` | 7 familias, `MIN_ROWS` nuevos, 12 clasificadores y 12 regresores nuevos, quitar `supports_probability`, `RANKING_METRIC` y `metrics_for_ranking` |
| `tests/test_model_specs.py` | recuentos, `class_weight_path`, tamaño de grid, `build_estimator`, compatibilidad y barrido de las 1620 combinaciones del grid |
| `tests/test_model_trainer.py` | borrar el test de `metrics_for_ranking`; entrenar de verdad los 24 modelos nuevos |
| `README.md` | tabla de modelos y de familias, con las decisiones no obvias del catálogo |

`ui/` no se toca: el desplegable se llena desde `list_models()` y el panel de
hiperparámetros desde `spec.grid()`.