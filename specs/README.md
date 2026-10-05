# Especificaciones SDD — AutoML Desktop

Este directorio contiene el ciclo de vida **Spec-Driven Development (SDD)**
de las nuevas funcionalidades. Ninguna línea de código se escribe hasta que
la especificación y el diseño están aprobados.

## Protocolo

```
1. spec.md          redactado por el agente
2. APROBACIÓN       revisado por el usuario   <- puerta 1
3. design.md        + tasks.md
4. APROBACIÓN       revisado por el usuario   <- puerta 2
5. código           + tests, siguiendo design.md y tasks.md
6. verificación     pytest -q y grep de arquitectura
```

`design.md` y `tasks.md` **no se crean** hasta pasar la puerta 1. El
contenido de cada spec.md es la lista de requisitos (`R-xxx`), las
invariantes que no se pueden romper y los criterios de aceptación
(`C-xxx`) que verifican el comportamiento.

## Reglas comunes a todas las specs

| Regla | Origen |
|---|---|
| `core/` no importa PySide6, PyQt ni `ui/` | `AGENTS.md` §3.1 |
| La UI no contiene lógica científica | `AGENTS.md` §3.5 |
| Trabajo pesado siempre en `ui/workers.py` | `AGENTS.md` §5.3 |
| Toda función nueva de `core/` tiene test en `tests/` | `AGENTS.md` §7.5 |
| No se cambia la API pública sin actualizar README y tests | `AGENTS.md` §7.4 |
| Mensajes al usuario en español | `AGENTS.md` §4 |
| No se añaden dependencias sin justificar en la spec | `AGENTS.md` §7.3 |

## Roadmap y orden de ejecución

El **número** de la spec es su identificador estable. El **orden de
ejecución** es el que minimiza el trabajo: se van encadenando las specs en el
orden indicado porque todas tocan `core/model_trainer.py::build_pipeline`.

| Spec | Titular | Orden | Depende de |
|---|---|---|---|
| [001](001-carga-multiformato/spec.md) | Carga multiformato CSV + Excel | 1 | — |
| [002](002-seleccion-atributos/spec.md) | Selección de atributos (4 algoritmos) | 5 | 004 |
| [003](003-metricas-clasificacion/spec.md) | Métricas F1 / AUC / exactitud | 3 | — |
| [004](004-balanceo-clases/spec.md) | Balanceo de clases + tabla estilo Weka | 4 | — |
| [005](005-comparativa-friedman/spec.md) | Comparativa de modelos con Friedman | 6 | 002, 004 |
| [006](006-export-predicciones/spec.md) | Exportar datos de entrada + predicción | 2 | 001 |

### Rediseño de la interfaz — rama `design/redesign-ui`

No son funcionalidades nuevas: es la reescritura de la capa visual. Las tres
se ejecutan en ese orden porque cada una usa lo que la anterior dejó.

| Spec | Titular | Orden | Depende de |
|---|---|---|---|
| [007](007-sistema-visual/spec.md) | Sistema visual y marca institucional | 1 | — |
| [008](008-estructura-navegacion/spec.md) | Estructura, navegación y densidad | 2 | 007 |
| [009](009-estados-y-densidad/spec.md) | Estados de los controles y densidad | 3 | 007, 008 |
| [010](010-estados-vacios/spec.md) | Estados vacíos explicados | 4 | 007, 008, 009 |
| [011](011-marca-visual-clara/spec.md) | Marca visual clara | 5 | — |

Orden de ejecución: **001 → 006 → 003 → 004 → 002 → 005 → 011**.

001 va primero porque 006 reutiliza su `detect_separator` para elegir el
separador del CSV exportado.

## Documentos por spec

| Spec | `spec.md` | `design.md` | `tasks.md` |
|---|---|---|---|
| 001 | puerta 1 superada | implementado | implementado |
| 002 | puerta 1 superada (R-005 corregido) | escrito | escrito |
| 003 | puerta 1 superada | implementado | implementado |
| 004 | puerta 1 superada | escrito | escrito |
| 005 | puerta 1 superada | escrito | escrito |
| 006 | puerta 1 superada | implementado | implementado |
| 007 | puerta 1 superada | implementado | implementado |
| 008 | puerta 1 superada | implementado | implementado |
| 009 | puerta 1 superada | implementado | implementado |
| 010 | puerta 1 superada | implementado | implementado |
| 011 | puerta 1 superada | implementado | implementado |
| 012 | puerta 1 superada (MLP excluido por el usuario) | implementado | implementado |

### Desviaciones registradas en 003

| Tema | Decisión tomada al implementar |
|---|---|
| `roc_auc_value` | El diseño lo enunciaba pero lo dejaba en el diseño; se implementa como función **pública** en `model_trainer` porque 005 lo reutiliza al comparar modelos. La comprobación de «una sola clase» vive en ella, no en `_compute_metrics`. |
| `_auc_scores` | La rama binaria usa la **columna 1** de `predict_proba`, igual que antes, pero ahora el binaryario se detecta por el número de clases (`len(clases) == 2`) en vez de por `proba.shape[1] == 2`: así multiclase y binario comparten camino y el AUC multiclase deja de perderse en silencio. |
| `decision_function` | Se admite: si existe y no hay `predict_proba`, se usa (con `scores.ndim == 1` convertido a dos columnas). El diseño lo pedía para modelos sin probabilidad. |
| `summarize_metrics` | Devuelve `[]` para un diccionario con solo claves de control (`auc_disponible`, `n_test`, `best_params`): son para la interfaz, no texto de métrica. Es justo el test que garantiza que un `bool` no acabe impreso como `1.0000`. |
| `n_test` | Se añadió a los dos tipos de tarea; el diseño solo lo pedía en clasificación. Es información gratuita y la tabla de resultados la muestra. |

### Desviaciones registradas en 012

| Tema | Decisión tomada al implementar |
|---|---|
| `_HIST_PARAMS` | El diseño anunciaba 216 combinaciones, pero `max_depth` tenía 4 valores y no 3: con `class_weight` el modelo llegaba a **864**. Se ajustaron `learning_rate` a 3 valores y `min_samples_leaf` a 2, y quedó en 432. |
| `LinearDiscriminant` | El diseño cruzaba `solver` × `shrinkage`, pero `svd` no acepta `shrinkage`: 2 de 6 combinaciones lanzaban `NotImplementedError` siempre, y en una búsqueda eso es un `NaN` silencioso. `solver="lsqr"` queda fijo en el factory y solo `shrinkage` se busca. |
| `KernelRidge` | El diseño copiaba `gamma=("scale", "auto")` del `SVC`. En `KernelRidge` `gamma` es un `float` o `None`, así que las 6 combinaciones fallaban. Ahora es `(0.01, 0.1, 1.0)`. |
| `LinearSVC` | Con la `hinge` por defecto y `C=100` liblinear no converge ni con 10000 iteraciones. Se fija `squared_hinge` y `loss` sale del grid. |
| `RadiusNeighbors` | El radio por defecto de scikit-learn (1.0) deja muestras sin vecinos con datos sin escalar y predice `NaN`. El factory usa `radius=5.0` y el grid `(2.0, 5.0, 10.0)`. |
| Recuento de regresores | El diseño decía 11 nuevos y 17 en total; son **12** nuevos y **18** en total. El error estaba en el recuento de la spec, no en el código. |
| El grid como invariante | Se añadió `test_todas_las_combinaciones_del_grid_son_validas`, que ajusta las 1620 combinaciones del catálogo. Es lo que destapó los tres grids rotos anteriores. |

### Desviaciones registradas en 006

| Tema | Decisión tomada al implementar |
|---|---|
| Separador del CSV exportado | El diseño unía los nombres de columna con `","` para chamar a `detect_separator`; eso **inyecta** la coma que se quiere detectar y hacía que saliera siempre `,`. Se unen con un espacio: así solo cuentan los separadores que aparecen de verdad dentro de los nombres. Sin ninguno, `;` (C-007). |
| `predict_frame` sin `np.asarray` en el diseño | Se usa `np.asarray(valores)` para que las probabilidades queden como columna numérica y no como `object`. |
| `SchemaReport.renamed` | Campo añadido con valor por defecto, así que los constructores antiguos siguen valiendo. `describe()` añade su línea. |
| Comprobación de filas | Vive en `_predictor`, antes de construir el frame, para que `predict` y `predict_frame` fallen igual. |
| Tests de UI | Los de exportación (C-007 a C-010) viven en `tests/test_file_dialogs.py` porque necesitan un `QApplication` headless; los de `core` (C-001 a C-006) en `tests/test_persistence.py`. |

### Desviaciones registradas en 001

| Tema | Decisión tomada al implementar |
|---|---|
| Cadena de lectura | Un único bucle sobre una lista de candidatos `[detectado] + resto de SEPARADORES` × `ENCODINGS`. El resultado es el mismo que las dos fases de D1/D2, con menos código y sin caso especial: si el separador detectado no valida, se prueban los otros tres igual que antes. |
| API pública | Se añadieron `load_csv_table` y `load_excel` como helpers públicos además de `load_table`; el despacho queda legible y cada rama se prueba sola. `load_csv` sigue siendo el wrapper de siempre. |
| Índice de hoja | Un `sheet` entero se resuelve a nombre con `hojas[indice]` (admite negativos); fuera de rango cae en la primera hoja en lugar de fallar. El nombre inexistente sí es un `DataLoadError` que lista las hojas. |
| Validación | `_valida` también rechaza los DataFrames de cero filas, por R-010: un CSV con solo cabecera avisa en vez de subir al resto de la app un dataset inútil. |
| `.xls` en tests | `xlrd` solo lee, así que se añadió `xlwt` como dependencia **de desarrollo** para generar el fixture BIFF real; el test de `.xls` es de ida y vuelta de verdad, no un `importorskip` vacío. |
| `pytest` | `pytest.ini` ya fijaba `testpaths` y `pythonpath`; `pyproject.toml` no lo duplica. |
| `pyproject.toml` | El layout real es plano (`core/`, `ui/`, `main.py` en la raíz), no `automl_app/`; el `[tool.setuptools.packages.find]` refleja la realidad y el entry point es `main:main`. |
| Tests de UI | `tests/test_file_dialogs.py` (nuevo) prueba los diálogos y `DataTab` en modo headless con un fixture que **intercepta `QMessageBox`**: sin él, un test de error se cuelga para siempre esperando un `exec()`. Advertencia para las specs siguientes. |
| black/isort | El repositorio **no** estaba formateado antes de este trabajo (19 ficheros pendientes). Solo se formatean los ficheros nuevos o tocados por una spec; el resto se deja para una commit de limpieza aparte. |

## Registro de refinamientos (detectados al diseñar)

Estos cambios se hicieron al escribir el diseño, leyendo el código real. No
cambian el alcance de ninguna spec:

| Spec | Refinamiento |
|---|---|
| 002, R-005 | `build_preprocessor` **ya** aplica `StandardScaler`, así que la regla definitiva es insertar `MinMaxScaler` **siempre** tras el `ColumnTransformer` cuando el método es `chi2`, no “si no hay escalado”. Texto de la spec corregido. |
| 003 | `available_metrics` ya devuelve las tres métricas de clasificación en el catálogo actual; el trabajo real está en `_compute_metrics` (AUC multiclase, claves siempre presentes, motivo). Se detecta además que `ui/train_tab.py:454` imprimiría `auc_disponible` como número al ser `bool`; se centraliza el formato en `summarize_metrics`. |
| 003 | `tune_model` llama a `_compute_metrics` sin `pipe`/`X_test` en la rama sin grid: el AUC no se calcularía ahí. Corregido en T-005. |
| 004, R-012 | Se sustituye `apply_class_weight` por el campo declarativo `class_weight_path` en `ModelSpec`, coherente con AGENTS §3.4. La UI **filtra** la opción `class_weight` en vez de avisar y continuar. |
| 004, R-006 | `BalancedSampler.fit` no remuestrea y `fit_transform` sí; el remuestreo se hace por **índices** para soportar `ndarray`, `DataFrame` y matriz dispersa (que es lo que produce `ColumnTransformer`). |
| 005, R-014 | El progreso usa la convención ya existente de tres argumentos `(etapa, actual, total)` y la señal `Signal(str, int, int)`, para reutilizar `TuneWorker`/`WorkerThread`. |
| 005 | Holm se implementa a mano: `statsmodels` no es una dependencia permitida. |
| 002, 004 | Los `.automl` antiguos restauran un `BundleMetadata` sin los campos nuevos, así que `describe()` lanzaría `AttributeError`. Se resuelve en un punto único: `_upgrade_metadata` en `load_bundle`. `BUNDLE_VERSION` sigue en `1`. |
| 006 | `predict()` mantiene su firma; el trabajo compartido pasa a un `_predictor` privado para evitar implementaciones divergentes. `SchemaReport` gana `renamed: tuple = ()` para comunicar los nombres libres. |

## Desviaciones registradas en 007

| Tema | Decisión tomada |
|---|---|
| Empaquetado de `core/` y `ui/` | Se sospechó que `find_packages` devolvía `[]` por faltar los `__init__.py` y que `pip install .` empaquetaba una distribución vacía. **Era falso**: con setuptools ≥61 (aquí 84.0) `packages.find` usa paquetes de espacio de nombres por defecto. Se comprobó construyendo la rueda antes y después. Aun así se añadieron `core/__init__.py` y `ui/__init__.py`, que son los que AGENTS §2 documenta. |
| `resources/style.qss` | Se daría por hecho que estaba "pendiente de implementar". Está vacío y `main.py` nunca lo carga: es un fichero muerto. Se borra y el QSS pasa a `ui/theme.py`. |
| `.gitignore` con `*.spec` | Dejaba fuera `AutoML_MEC_Desktop.spec`, que es el fichero que hay que editar para incluir los recursos. Corregido en la fase 0 de la rama. |
| Empaquetar la fuente Ubuntu | Descartado. No está instalada en el sistema y añadirla sería una dependencia nueva (AGENTS §7.3). Se usa la lista de preferencia de `QFont.setFamilies`. |
| Sombras con `QGraphicsDropShadowEffect` | Descartado: repinta el canvas de matplotlib en cada cambio y no aporta nada a un tema plano. |

## Desviaciones registradas en 008

| Tema | Decisión tomada |
|---|---|
| Auditoría del layout | El primer análisis delegate-se apoya en un agente externo resulted incorrecto en varios puntos (`goto_train`/`goto_predict` no existen, `preprocess_tab` no tiene 9 filas densas). La investigación se repitió leyendo el código real: el problema es la **falta de scroll** y las **tablas sin `setMaximumHeight`**, no la densidad de las filas. La spec se escribió sobre los datos reales. |
| `_go_to_predict` con `QStackedWidget` | El `hasattr(tabs, "setCurrentIndex")` de la función seguiría siendo cierto con un `QStackedWidget`, así que el fallo se habría mantenido en silencio. Se elimina en favor de la señal `navigate_requested`. |

## Transversal a todas las specs

- `pyproject.toml` (ya contemplado en `AGENTS.md` §2, pero inexistente) pasa a
  declarar dependencias de ejecución y de desarrollo. La instalación en
  `entorno/` se pide como paso explícito al usuario.
- `AGENTS.md` §2 se actualiza para incluir `specs/` y los nuevos módulos de
  `core/`.
- Cada spec que añada campos a `AppState` los documenta en `core/state.py` y en
  el README (§3.2 de `AGENTS.md`).

## Dependencias nuevas previstas

| Paquete | Motivo | Spec |
|---|---|---|
| `openpyxl` | Lectura de `.xlsx` / `.xlsm` | 001, 006 |
| `xlrd` | Lectura de `.xls` (binario antiguo) | 001 |
| `odfpy` | Lectura de `.ods` (OpenDocument) | 001 |
| `imbalanced-learn` | SMOTE / SMOTEN | 004 |

`scipy`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `joblib` y `PySide6`
ya están instalados y solo se declararán en `pyproject.toml`.