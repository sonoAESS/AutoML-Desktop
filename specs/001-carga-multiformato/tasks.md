# SDD-001 — Tareas de implementación

Estado: **implementada** (2026-10-04). 123 tests en verde.

Leyenda: `[ ]` pendiente · `[x]` hecha. Cada tarea indica fichero y verificación.

## Fase A — Entorno

### [ ] T-001 · Crear `pyproject.toml`
**Ficheros:** `pyproject.toml` (nuevo)

1. Declarar `[project]` con las dependencias del §7 del diseño.
2. Declarar `[project.optional-dependencies] dev` con `pytest`, `black`, `isort`.
3. Declarar `[tool.black] line-length = 88` e `[tool.isort] profile = "black"`.
4. Declarar `[tool.pytest.ini_options] testpaths = ["tests"]`.
**Verificación:** `python -c "import tomllib; tomllib.load(open('pyproject.toml','rb'))"` (3.9 usa `tomli`).

### [ ] T-002 · Instalar las dependencias en `entorno/`
**Ficheros:** ninguno (cambia `entorno/`)

1. Pedir confirmación explícita al usuario.
2. `pip install openpyxl xlrd odfpy` y `pip install pytest black isort`.
3. `pip freeze` no se commitea (`entorno/` está en `.gitignore`).
**Verificación:** `python -c "import openpyxl, xlrd, odfpy; print('ok')"` y `black --version`.

## Fase B — Núcleo

### [ ] T-003 · `DataLoadError`, `ENCODINGS` y `detect_separator`
**Ficheros:** `core/data_loader.py`
**Requisitos:** R-004, R-005, R-007, R-011

1. Añadir `DataLoadError(ValueError)`.
2. Añadir `ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")`.
3. Añadir `detect_separator(header_line)`.
**Verificación:** test unitario de los cuatro separadores y de una cabecera sin separador (`detect_separator("edad;altura") == ";"`, `detect_separator("edad") is None`).

### [ ] T-004 · `list_sheets`
**Ficheros:** `core/data_loader.py`
**Requisitos:** R-001, R-003, R-005, R-006

1. Tabla extensión → motor (`_EXCEL_ENGINES`).
2. CSV → `[]`.
3. Extensión no tabular → `DataLoadError`.
4. Motor ausente → `DataLoadError` con el paquete a instalar.
5. Usar `pd.ExcelFile(path, engine=motor)` en contexto `with`.
**Verificación:** test con `.xlsx` de 3 hojas (importorskip) y test de extensión inválida.

### [ ] T-005 · `load_table`
**Ficheros:** `core/data_loader.py`
**Requisitos:** R-002, R-003, R-008, R-009, R-010

1. Firmar `load_table(path, sheet=0, sep=None)`.
2. Rama CSV: lectura de la primera línea en `latin-1`, separador detectado o forzado, bucle de codificaciones, fallback al bucle heredado.
3. Rama Excel: validación de `sheet`, `read_excel` con motor explícito.
4. DataFrame vacío o sin columnas → `DataLoadError`.
5. `load_csv` pasa a ser `return load_table(path)`.
**Verificación:** `pytest -q tests/test_data_loader.py` con los 5 tests antiguos sin modificar + los nuevos.

### [ ] T-006 · Tests de `data_loader`
**Ficheros:** `tests/test_data_loader.py`
**Criterios:** C-001 a C-007, C-010, C-011

1. CSV `;` en `cp1252` con acentos (escribir con `encoding="cp1252"`).
2. CSV `,` en `utf-8` con BOM: primera columna sin `\ufeff`.
3. CSV de una columna y CSV `|` y tabulado (regresión de D2).
4. `.xlsx`, `.xls`, `.ods` de una hoja (los dos últimos con `importorskip`).
5. Fichero vacío → `DataLoadError`.
6. Extensión `.parquet` → `DataLoadError` y `isinstance(e, ValueError)`.
**Verificación:** `pytest -q tests/test_data_loader.py` verde.

## Fase C — Interfaz

### [ ] T-007 · `ui/file_dialogs.py`
**Ficheros:** `ui/file_dialogs.py` (nuevo)
**Requisitos:** R-012

1. `TABLE_FILTER` con el orden de la spec.
2. `choose_table_file(parent, caption)`.
3. `choose_sheet(parent, sheets)` con `QInputDialog.getItem`.
4. `choose_save_path(parent, caption, default_name, filters)` reutilizable por 006.
**Verificación:** importar el módulo con `QT_QPA_PLATFORM=offscreen` sin error.

### [ ] T-008 · `DataTab` usa el diálogo compartido
**Ficheros:** `ui/data_tab.py`
**Requisitos:** R-012, R-013, R-014, R-016

1. Renombrar `load_csv` → `load_data` y actualizar `btn_load.clicked`.
2. Sustituir el `QFileDialog` inline por `file_dialogs.choose_table_file`.
3. `list_sheets` → `choose_sheet` si hay más de una → `load_table`.
4. `state.raw_df`, `state.source_path` y `data_loaded.emit()` como antes.
5. `except Exception` → `QMessageBox.critical`.
**Verificación:** prueba headless que instancia `DataTab`, monkeypatchea el diálogo con un `.xlsx` temporal y comprueba `state.raw_df`.

### [ ] T-009 · `PredictTab` acepta los mismos formatos
**Ficheros:** `ui/predict_tab.py`
**Requisitos:** R-012, R-015

1. Botón “Cargar datos de entrada…” con el filtro compartido.
2. `load_table` + `choose_sheet`.
**Verificación:** prueba headless de `load_data` con un `.xlsx` temporal y un bundle en memoria.

## Fase D — Documentación

### [ ] T-010 · `AGENTS.md` y `README.md`
**Ficheros:** `AGENTS.md`, `README.md`

1. `AGENTS.md` §2: añadir `specs/` y `ui/file_dialogs.py`.
2. `AGENTS.md` §3.4: `data_loader.py` sigue siendo solo E/S, ahora multiformato.
3. `README.md`: sección de carga de datos con formatos, hojas y codificaciones.
**Verificación:** revisar que la estructura de `AGENTS.md` coincide con `find . -name "*.py" -not -path "./entorno/*"`.

## Verificación final de la spec

```bash
grep -R "PySide6\|PyQt\|from ui\." core/   # vacío
python -m pytest -q
```

- [x] `pytest -q` verde y sin tests saltados que no sean los de Excel sin motor.
- [x] `C-001`…`C-012` cubiertos por tests.