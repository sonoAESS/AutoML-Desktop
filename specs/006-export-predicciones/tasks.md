# SDD-006 — Tareas de implementación

Estado: **implementada** (2026-10-04), con las desviaciones registradas en
`specs/README.md`. 144 tests en verde.

### [ ] T-001 · `_predictor` compartido
**Ficheros:** `core/persistence.py`
**Requisitos:** R-004, R-005

1. Extraer el cálculo actual de `predict` (líneas ~327-349) a
   `_predictor(bundle, df, strict) -> (X, y, report, columnas)`.
2. `predict` pasa a usar `_predictor` y a devolver exactamente lo mismo que
   antes: `(salida, report, X, y)` con solo columnas de predicción.
**Verificación:** `pytest -q tests/test_persistence.py` verde sin tocar los
tests existentes (C-004).

### [ ] T-002 · `_free_names` y `SchemaReport.renamed`
**Ficheros:** `core/persistence.py`
**Requisitos:** R-006

1. `_free_names(existing, nuevos)` con sufijos incrementales desde `_2`.
2. `SchemaReport.renamed: tuple = ()` como campo opcional.
3. `describe()` añade la línea de columnas renombradas.
**Verificación:** test de `_free_names` con `{"prediccion"}` presente y con
`{"prediccion", "prediccion_2"}` presentes.

### [ ] T-003 · `predict_frame`
**Ficheros:** `core/persistence.py`
**Requisitos:** R-001, R-002, R-003, R-007, R-008

1. Partir de `df.copy()` y añadir las columnas con los nombres libres.
2. `ValueError` si el número de predicciones no coincide con las filas.
3. Devolver el `report` con `renamed` relleno mediante `dataclasses.replace`.
4. Docstring que deja claro que devuelve la entrada más la predicción.
**Verificación:** C-001, C-002, C-003, C-005, C-006.

### [ ] T-004 · `detect_separator` con preferencia
**Ficheros:** `core/data_loader.py`
**Diseño §4.1**

1. Añadir el parámetro `preference=SEPARADORES` sin cambiar el comportamiento
   por defecto.
2. Test de que `detect_separator("a;b;c", preference=(";", ",")) == ";"` y de que
   el valor por defecto sigue ganando con `,`.
**Verificación:** `pytest -q tests/test_data_loader.py` verde.

### [ ] T-005 · Tests de persistencia
**Ficheros:** `tests/test_persistence.py`
**Criterios:** C-001 a C-006

Fixture: bundle en memoria con `LogisticRegression` entrenado sobre un dataset
pequeño (sin escribir a disco), con `metadata.feature_columns` de 2 columnas.

1. El frame devuelto tiene las columnas de entrada primero y luego `prediccion`
   (y `prob_*`).
2. Índice no `RangeIndex` → orden de filas conservado.
3. Columnas extra del input conservadas.
4. `predict()` idéntico a antes (longitud y valores).
5. Colisión: input con columna `prediccion` → `prediccion_2` y original intacta.
6. Input con la columna objetivo → aparece sin tocar.
**Verificación:** `pytest -q tests/test_persistence.py` verde.

### [ ] T-006 · `PredictTab` usa `predict_frame`
**Ficheros:** `ui/predict_tab.py`
**Requisitos:** R-009, R-010, R-011

1. `self._preview` (solo predicción) y `self._predictions` (entrada + predicción).
2. `predict()` llama a `persistence.predict_frame` con el mismo flujo de
   `try/except ValueError` → `_confirmar_sin_columnas()` → reintento con
   `strict=False`.
3. `lbl_resultado` con el recuento de columnas y `report.describe()`.
4. El botón se llama “Exportar resultados (datos + predicción)…” y el nombre
   por defecto es `predicciones.csv`.
**Verificación:** test headless que carga un bundle, predice sobre un CSV
temporal y comprueba que `_predictions` tiene las columnas de entrada.

### [ ] T-007 · Exportación CSV y XLSX
**Ficheros:** `ui/predict_tab.py`
**Requisitos:** R-012, R-013, R-014, R-015

1. `_exportar_csv(df, ruta)`: `sep` detectado con preferencia `;`, `index=False`,
   `encoding="utf-8-sig"`.
2. `_exportar_excel(df, ruta)`: comprobar `openpyxl` con
   `importlib.util.find_spec` y `RuntimeError` en español si falta.
3. Elegir según la extensión del destino; todo `except Exception` →
   `QMessageBox.critical`.
4. Mensaje final con la ruta y el número de filas.
**Verificación:** C-007 (ida y vuelta CSV en `tmp_path`, tildes incluidas) y
C-008 con `pytest.importorskip("openpyxl")`.

### [ ] T-008 · `README.md`
**Ficheros:** `README.md`

1. Sección de Predicción: el fichero exportado contiene los datos de entrada más
   la predicción y las probabilidades; formato CSV (`utf-8-sig`, `;`) y XLSX.
2. Mencionar el renombrado automático si el input ya trae `prediccion`.

## Verificación final de la spec

```bash
python -m pytest -q
grep -R "PySide6\|PyQt\|from ui\." core/   # vacío
```

- [x] C-001…C-011 cubiertos.
- [x] `git diff core/persistence.py` no cambia la firma de `predict`.