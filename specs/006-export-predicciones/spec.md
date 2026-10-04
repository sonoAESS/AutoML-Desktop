# SDD-006 — Exportar los datos de entrada junto con la predicción

| Campo | Valor |
|---|---|
| Estado | **Implementada** (SDD-006 cerrada) |
| Orden de ejecución | 2 de 6 |
| Ámbito | `core/persistence.py`, `ui/predict_tab.py`, `tests/test_persistence.py`, `README.md` |
| Depende de | SDD-001 (reutiliza `data_loader.detect_separator`) |
| Bloquea a | — |

## 1. Problema

Hoy `core/persistence.py:340` construye la salida de `predict` como un
`DataFrame` con **solo** la columna `prediccion` (y `prob_<clase>`), y
`ui/predict_tab.py:197` exporta exactamente ese `DataFrame` con
`to_csv(index=False)`.

El resultado es que el usuario recibe un fichero con una sola columna de
valores sueltos, sin ninguna forma de saber a qué fila del dataset original
pertenece cada predicción. Para el uso típico de la herramienta —un profesor
que predice casos nuevos y necesita entregar la tabla completa, o un usuario
que cruza la predicción con sus propios datos— eso hace el resultado inútil.

## 2. Objetivo

Que el fichero exportado contenga **el dataset de entrada tal cual se cargó**
(todas sus columnas, orden y valores) más las columnas de predicción
añadidas al final, sin perder ni una fila ni una columna.

## 3. Alcance

### 3.1 Incluido

- Nueva función `predict_frame()` en `core/persistence.py`.
- `predict()` se mantiene con su comportamiento actual como wrapper.
- Exportación a CSV con `utf-8-sig` y separador detectado.
- Exportación a `.xlsx` reutilizando `openpyxl` (SDD-001).
- Tests de orden, columnas, colisiones e idempotencia de ida y vuelta.

### 3.2 No incluido

- Métricas sobre datos etiquetados, aunque el input traiga la columna
  objetivo: se conserva tal cual, sin recalcular nada.
- Exportar a JSON, Parquet o base de datos.
- Guardar la predicción dentro del `.automl`.
- Importar la predicción de vuelta en la app.

## 4. Requisitos

### API de `core`

- **R-001.** Se añade
  `predict_frame(bundle: Bundle, df: pd.DataFrame, strict: bool = True) -> tuple[pd.DataFrame, SchemaReport]`
  que devuelve el `DataFrame` de **entrada** con las columnas de predicción
  añadidas.
- **R-002.** El resultado parte de `df` (el `DataFrame` original que el
  usuario cargó, con sus columnas extra, su orden y sus dtypes) y **no** de
  `X` ni de `frame` de `align_features`, para no perder columnas ni aplicar la
  coerción de tipos que hace el pipeline internamente.
- **R-003.** Se conservan el índice y el orden de las filas de la entrada. Si
  `predict` devolviese un número de filas distinto, `predict_frame` lanza
  `ValueError` en español (no puede ocurrir con `align_features`, pero el test
  lo cubre con un doble).
- **R-004.** Las columnas añadidas son `PREDICTION_COLUMN` (`"prediccion"`) y,
  si el modelo expone `predict_proba`, una `prob_<clase>` por clase, tal y
  como define hoy `predict` (R-003 de SDD-003 no aplica aquí: son las mismas
  columnas).
- **R-005.** `predict(bundle, df, strict)` **mantiene su firma y su valor de
  retorno actuales** — `(predicciones, report, X, y)` con solo las columnas de
  predicción. Se reimplementa como wrapper de `predict_frame` que se queda con
  las columnas añadidas, de modo que ni la UI existente ni los tests ni
  ningún usuario de la API noten nada.
- **R-006.** Si la entrada ya contiene una columna `prediccion` o `prob_*`, la
  columna añadida se renombra con sufijo incremental (`prediccion`,
  `prediccion_2`, `prediccion_3`, …) y el renombrado se comunica al usuario en
  `lbl_resultado`. No se pisan columnas ni se lanza excepción.
- **R-007.** Si la entrada contiene la columna objetivo con valores reales, se
  conserva sin tocar. El `SchemaReport` no cambia respecto al actual.
- **R-008.** La documentación (`docstring`) de ambas funciones deja explícito
  que `predict_frame` devuelve la entrada más la predicción.

### UI

- **R-009.** `ui/predict_tab.py` guarda el `DataFrame` enriquecido en
  `self._predictions` (resultado de `predict_frame`) y usa ese objeto tanto
  para la tabla de salida como para la exportación.
- **R-010.** La tabla de salida sigue mostrando **solo** las columnas de
  predicción, para que siga siendo legible cuando el dataset tiene 30 columnas.
- **R-011.** `lbl_resultado` informa del formato de la exportación: “El
  fichero tendrá N columnas de entrada + M de predicción”, más los nombres
  añadidos si hubo colisión (R-006).
- **R-012.** El botón pasa a “Exportar resultados (datos + predicción)…” con
  nombre por defecto `predicciones.csv` y filtro
  `CSV (*.csv);;Excel (*.xlsx)`.
- **R-013.** La escritura a CSV usa `encoding="utf-8-sig"` y el separador
  detectado sobre las cabeceras mediante el helper de `data_loader` (con `;`
  ganando en caso de empate), para que Excel en español lo abra directamente.
- **R-014.** La escritura a `.xlsx` usa `df.to_excel(index=False)`. Si
  `openpyxl` no está instalado, la UI avisa con `QMessageBox.critical` en
  español en vez de fallar con un error interno.
- **R-015.** Cualquier error de escritura se captura y se muestra con
  `QMessageBox.critical`, nunca como traceback.

## 5. Invariantes de arquitectura

- La lógica de “unir entrada y predicción” vive en `core`, no en la UI.
- `predict()` no cambia de comportamiento: es API pública documentada.
- El bundle no se modifica al predecir.
- `core/persistence.py` sigue sin importar Qt.

## 6. Criterios de aceptación

- **C-001.** Dado un bundle entrenado y un dataset de entrada de 5 columnas,
  cuando se llama a `predict_frame`, entonces el resultado tiene 5 + 1 (+ N
  probabilidades) columnas y las 5 primeras son las de entrada, en el mismo
  orden y con los mismos valores.
- **C-002.** El `DataFrame` exportado conserva el **orden de las filas** de la
  entrada, incluso si el índice no es un `RangeIndex`.
- **C-003.** Las columnas extra del input que el modelo no necesita se
  conservan en el resultado.
- **C-004.** `predict(bundle, df)` sigue devolviendo un `DataFrame` con solo
  `prediccion` (+ `prob_*`), con la misma longitud y los mismos valores que
  antes de esta spec (test de regresión).
- **C-005.** Si el input ya trae una columna `prediccion`, el resultado tiene
  `prediccion` (la original) y `prediccion_2` (la nueva), sin excepción y sin
  sobrescribir el valor original.
- **C-006.** Si el input trae la columna objetivo con valores reales, esta
  aparece intacta y `report.missing` sigue vacío.
- **C-007.** Escritura a CSV con `utf-8-sig` y `;`, releída con
  `data_loader.load_table`, reproduce los mismos valores y columnas
  (ida y vuelta sin pérdida, tildes incluidas).
- **C-008.** Escritura a `.xlsx` y relectura con `load_table` reproducen los
  mismos valores (test marcado con `importorskip("openpyxl")`).
- **C-009.** La tabla de salida de la UI muestra únicamente las columnas de
  predicción y la etiqueta indica el número de columnas del fichero exportado.
- **C-010.** Si `openpyxl` no está instalado, elegir `.xlsx` muestra un
  `QMessageBox.critical` en español y no escribe un fichero corrupto.
- **C-011.** `pytest -q` en verde con al menos 6 tests nuevos o ampliados en
  `tests/test_persistence.py`.

## 7. Dependencias

- `openpyxl` para el export a `.xlsx`. **Reutiliza la dependencia de
  SDD-001**: si esa spec no se implementa, la exportación a `.xlsx` se
  desactiva con aviso y el CSV sigue funcionando. No se añade ninguna
  dependencia extra.

## 8. Riesgos

| Riesgo | Mitigación |
|---|---|
| Quebrar la API pública `predict` | R-005: `predict` se mantiene y se cubre con test de regresión (C-004) |
| Coerción de dtypes al unir (fechas, ints) | R-002: se parte de `df`, no de `X` |
| Nombre de columna duplicado al exportar | R-006 con sufijo incremental y aviso en la UI |
| CSV con acentos roto en Excel | R-013: `utf-8-sig` |
| Ficheros muy grandes en memoria | Se mantiene el flujo actual: un único `DataFrame` en memoria; se documenta el límite |

## 9. Decisiones que se cierran en `design.md`

- Nombre exacto de la función (`predict_frame` frente a `predict_dataset`).
- Si el sufijo de colisión es `_2` o `_dup1`.
- Si la exportación a `.xlsx` se ofrece ya en esta spec o cuando SDD-001 esté
  implementada.
- Si el separador de escritura se detecta o se fija en `;`.