# SDD-001 — Carga multiformato (CSV y Excel)

| Campo | Valor |
|---|---|
| Estado | **Implementada** (SDD-001 cerrada) |
| Orden de ejecución | 1 de 6 |
| Ámbito | `core/data_loader.py`, `ui/data_tab.py`, `ui/predict_tab.py`, `tests/test_data_loader.py` |
| Depende de | — |
| Bloquea a | 006 (comparte filtro de archivos), 005 (usa el mismo dataset) |

## 1. Problema

`core/data_loader.py:load_csv` solo lee CSV y fija `encoding="utf-8"`. En la
práctica los usuarios de este tipo de herramienta suelen tener datos exportados
desde Excel, que fallan por dos motivos independientes:

- El fichero es `.xlsx`, `.xls` u `.ods`, no un CSV. No hay forma de cargarlo.
- El fichero es un CSV en `cp1252` con separador `;` (combinación habitual al
  exportar desde Excel en español). Hoy falla con `UnicodeDecodeError` para
  todos los candidatos de `SEPARADORES` y termina en
  `ValueError: No se pudo leer el archivo CSV correctamente.`

Además, `ui/predict_tab.py:116` ofrece únicamente el filtro `CSV (*.csv *.txt
*.tsv)`, de modo que aunque se soportase Excel habría que arreglar la UI en dos
sitios.

## 2. Objetivo

Una única función de carga en `core/` capaz de leer CSV, XLSX, XLS y ODS con
detección de separador y de codificación, manteniendo la API actual intacta
para no romper callers ni tests.

## 3. Alcance

### 3.1 Incluido

- API `load_table()` como punto de entrada único, con despacho por extensión.
- API `list_sheets()` paraEnumerator las hojas de un libro.
- Cadena de codificación con reintentos para CSV.
- Filtros de `QFileDialog` unificados en las dos pestañas que piden ficheros.
- Selección de hoja cuando el libro tiene más de una.
- Tests de los cinco formatos y de las cuatro codificaciones.

### 3.2 No incluido

- Guardado de datasets a disco desde la UI (solo se exportan predicciones, ver
  SDD-006).
- Lectura de `.xlsb`, `.parquet`, `.json` ni bases de datos.
- Editor de celdas o vista previa del libro antes de elegir la hoja.
- Detección automática *contenido* del separador más allá de los cuatro
  candidatos actuales.

## 4. Requisitos

### API de `core`

- **R-001.** Se añade `list_sheets(path: str) -> list[str]`. Devuelve los
  nombres de las hojas del libro. Para un CSV devuelve `[]` (no es un error) y
  para una extensión no tabular lanza `DataLoadError`.
- **R-002.** Se añade `load_table(path: str, sheet: str | int = 0, sep: str | None = None) -> pd.DataFrame`.
  `sep=None` significa autodetección. Si `sheet` es un entero se interpreta
  como índice posicional de la hoja.
- **R-003.** Despacho por extensión, case-insensitive:

  | Extensión | Motor | Paquete |
  |---|---|---|
  | `.csv`, `.txt`, `.tsv` | `pd.read_csv` | — |
  | `.xlsx`, `.xlsm` | `pd.read_excel` | `openpyxl` |
  | `.xls` | `pd.read_excel` | `xlrd` |
  | `.ods` | `pd.read_excel` | `odfpy` |

- **R-004.** Cualquier otra extensión lanza `DataLoadError` con mensaje en
  español que enumere las extensiones admitidas.
- **R-005.** `DataLoadError` es subclase de `ValueError`, para que los
  `except ValueError` y `pytest.raises(ValueError)` existentes sigan siendo
  válidos.
- **R-006.** Si el motor requerido no está instalado, `DataLoadError` explica
  qué paquete falta y cómo instalarlo, en lugar de propagar un `ImportError`
  o un error interno de `openpyxl`.
- **R-007.** La autodetección de separador mantiene `SEPARADORES = (",", ";",
  "\t", "|")` y su orden de preferencia actual, y conserva el comportamiento
  especial de los CSV de una sola columna (`core/data_loader.py:23`).
- **R-008.** La lectura de CSV prueba las codificaciones en el orden
  `utf-8-sig` → `cp1252` → `latin-1`. `latin-1` nunca falla, por lo que siempre
  se obtiene un `DataFrame`. Se conserva el criterio actual de aceptar el
  resultado cuando tiene más de una columna o cuando su único nombre no
  contiene separadores.
- **R-009.** `load_csv(path)` se mantiene con su firma y su semántica, y
  delegan en `load_table` con `sep=None`. Se conserva su docstring actualizado.
- **R-010.** `load_table` devuelve un `DataFrame` no vacío y con al menos una
  columna; un fichero vacío o sin cabecera lanza `DataLoadError` explicativo.
- **R-011.** Si la lectura falla, `DataLoadError` enumera las codificaciones y
  los separadores que se han probado, para que el usuario sepa qué se ha
  intentado antes de corregir el fichero a mano.

### UI

- **R-012.** `ui/data_tab.py` y `ui/predict_tab.py` comparten una única función
  de diálogo (definida en `ui/data_tab.py` y reutilizada, o en el módulo de
  widgets que corresponda) con este filtro:
  `CSV (*.csv *.txt *.tsv);;Excel (*.xlsx *.xlsm *.xls);;OpenDocument (*.ods);;Todos los archivos (*)`.
- **R-013.** Tras elegir el fichero, si `list_sheets(path)` devuelve más de un
  nombre, la UI pregunta con `QInputDialog.getItem` qué hoja usar, con la
  primera como opción por defecto. Con una sola hoja no se pregunta nada.
- **R-014.** `ui/data_tab.py` actualiza `state.source_path` con la ruta
  elegida y `state.raw_df` con el `DataFrame` devuelto, sin ninguna
  transformación adicional.
- **R-015.** El botón de Predicción pasa a decir “Cargar datos de entrada…” y
  llama a `load_table`, de modo que admita los mismos formatos que Datos.
- **R-016.** Todo error de carga se muestra con `QMessageBox.critical` y el
  texto de `DataLoadError`; nunca se muestra un traceback.

## 5. Invariantes de arquitectura

- `core/data_loader.py` solo hace E/S; no conoce `scikit-learn`, `ui/` ni Qt.
- `ui/` no llama a `pd.read_csv` / `pd.read_excel` directamente.
- `AppState.raw_df` sigue siendo el contenido literal del fichero: la
  detección de separador o de codificación es parte de la lectura, no una
  transformación posterior.
- No se muta el `DataFrame` devuelto por el llamante.

## 6. Criterios de aceptación

- **C-001.** Dado un `.xlsx` de una hoja, cuando se llama a
  `load_table(ruta)`, entonces devuelve un `DataFrame` con los mismos valores
  que el fichero y con los nombres de columna intactos.
- **C-002.** Ídem para `.xls` y `.ods`.
- **C-003.** Dado un CSV `;` en `cp1252` con acentos, cuando se llama a
  `load_table(ruta)`, entonces se lee sin `UnicodeDecodeError` y con las
  columnas separadas por `;`.
- **C-004.** Dado un CSV `,` en `utf-8` con BOM, cuando se llama a
  `load_table(ruta)`, entonces la primera columna no incluye el carácter BOM.
- **C-005.** Dado un CSV de una sola columna cuyo nombre contiene `_` o ` `,
  cuando se llama a `load_table(ruta)`, entonces se devuelve una única columna
  (comportamiento de `core/data_loader.py:23` intacto).
- **C-006.** Dado un CSV `|` o tabulado, cuando se llama a `load_table`, entonces
  se detecta el separador correcto.
- **C-007.** Dado un fichero con extensión `.txt`, cuando se llama a
  `load_table`, entonces se trata como CSV.
- **C-008.** Dado un libro de tres hojas, cuando la UI intenta cargarlo, entonces
  aparece un selector con los tres nombres y el `DataFrame` corresponde a la
  hoja elegida; con una sola hoja no aparece ninguna pregunta.
- **C-009.** Dado un fichero vacío, cuando se llama a `load_table`, entonces
  lanza `DataLoadError` con mensaje en español y el `DataFrame` original no se
  toca en `AppState`.
- **C-010.** Dado un fichero con extensión no tabular, cuando se llama a
  `load_table`, entonces `DataLoadError` es instancia de `ValueError`.
- **C-011.** `load_csv` sigue funcionando idéntico para CSV: los cinco tests
  existentes en `tests/test_data_loader.py` pasan sin modificarlos.
- **C-012.** `grep -R "PySide6\|PyQt\|from ui\." core/` no devuelve nada y
  `pytest -q` pasa completo.

## 7. Dependencias

- `openpyxl` (`.xlsx`, `.xlsm`) — no instalado.
- `xlrd` (`.xls`) — no instalado.
- `odfpy` (`.ods`) — no instalado.

Las tres se justifican porque son las únicas formas de leer esos formatos sin
reescribir un parser de OfML/BIFF. Se declaran en `pyproject.toml` y su
instalación en `entorno/` se pide al usuario como paso explícito. Los tests de
Excel se marcan con `pytest.importorskip` para que la suite siga siendo verde si
alguien no las instala.

## 8. Riesgos

| Riesgo | Mitigación |
|---|---|
| `latin-1` nunca falla y podría “enmascarar” un CSV corrupto producing mojibake | Se prueban `utf-8-sig` y `cp1252` primero; solo se cae en `latin-1` si ambos fallan, y el mensaje de error de carga lo indica |
| `.xls` antiguo con hojas ocultas o protección | `xlrd` las ignora; se documenta que solo se leen hojas visibles |
| Ficheros `.xlsx` muy grandes | `openpyxl` va íntegro en memoria; se documenta el límite orientativo (≈1–2 M de celdas) |
| Regresión en la detección de separador | Los tests existentes se conservan sin tocar y `SEPARADORES` no cambia |

## 9. Decisiones que se cierran en `design.md`

- Cuál es exactamente la cadena de lectura: ¿un solo `read_csv` con lista de
  `encodings` de pandas o un bucle explícito por codificación?
- Dónde vive el diálogo compartido de selección de fichero (helper en `ui/`
  o método de `PredictTab` reutilizado).
- Si `list_sheets` para CSV devuelve `[]` o `["CSV"]`.
- Si `.xlsm` se acepta o se rechaza con mensaje explícito (los macros no se
  ejecutan, solo se leen).