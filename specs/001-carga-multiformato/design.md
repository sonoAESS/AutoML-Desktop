# SDD-001 — Diseño técnico: carga multiformato

Estado: **implementada** (2026-10-04); desviaciones en el registro de
`specs/README.md`.

## 1. Mapa de cambios

| Fichero | Cambio |
|---|---|
| `pyproject.toml` (nuevo) | Declara dependencias de ejecución y desarrollo |
| `core/data_loader.py` | `DataLoadError`, `ENCODINGS`, `detect_separator`, `list_sheets`, `load_table`, `load_csv` como wrapper |
| `ui/file_dialogs.py` (nuevo) | Diálogos compartidos: filtro de ficheros, selector de hoja, destino de exportación |
| `ui/data_tab.py` | Usa `file_dialogs.choose_table_file`, `load_table` y `choose_sheet` |
| `ui/predict_tab.py` | Usa el mismo diálogo y `load_table` |
| `tests/test_data_loader.py` | Tests de formatos, codificaciones y errores |
| `AGENTS.md` | Añade `ui/file_dialogs.py` y `specs/` a la estructura |

## 2. Decisiones (cierra §9 de la spec)

| # | Decisión |
|---|---|
| D1 | **Cadena de lectura en dos fases**: se detecta el separador sobre la primera línea física (en bytes, sin decodificar) y después se itera solo sobre las codificaciones. Evita el bucle de 4 × 3 = 12 lecturas. |
| D2 | Si la lectura con el separador detectado no supera la validación de “más de una columna”, se cae al **bucle heredado** (4 separadores × 3 codificaciones) para no perder ningún caso que hoy funciona. |
| D3 | `list_sheets` devuelve `[]` para CSV (no es un error). Para extensión no tabular lanza `DataLoadError`. |
| D4 | Se acepta `.xlsm` (lectura; los macros no se ejecutan) y `.xlsb` se rechaza con mensaje explícito. |
| D5 | El motor de Excel se pide **explícito** por extensión (`openpyxl`, `xlrd`, `odfpy`) para poder dar un mensaje de instalación útil en vez de un error interno de pandas. |
| D6 | Los diálogos compartidos viven en `ui/file_dialogs.py`, módulo nuevo y acotado (no es un `utils.py`: solo knows de `QFileDialog`, `QInputDialog` y del filtro multiformato). |
| D7 | `DataLoadError(ValueError)`; el mensaje de error enumera los intentos (separadores y codificaciones) según R-011. |

## 3. API resultante

```python
# core/data_loader.py

SEPARADORES = (",", ";", "\t", "|")          # sin cambios
ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")

class DataLoadError(ValueError):
    """Fichero que no se puede leer como tabla."""

def detect_separator(header_line: str) -> str | None:
    """Separador más frecuente en la cabecera, o None si no hay ninguno."""

def list_sheets(path: str) -> list[str]:
    """Nombres de las hojas; [] para CSV; DataLoadError si no es tabular."""

def load_table(path: str, sheet: str | int = 0, sep: str | None = None) -> pd.DataFrame:
    """Lee CSV/XLSX/XLS/ODS. `sep=None` autodetecta."""

def load_csv(path: str) -> pd.DataFrame:
    """Wrapper retrocompatible: `load_table(path)`."""
```

`load_table` es el único punto de entrada. `load_csv` delega y conserva su
firma, de modo que `tests/test_data_loader.py` no se toca.

## 4. Algoritmos

### 4.1 `detect_separator`

```python
def detect_separator(header_line: str) -> str | None:
    conteos = {sep: header_line.count(sep) for sep in SEPARADORES}
    sep, n = max(conteos.items(), key=lambda par: par[1])
    return sep if n > 0 else None
```

Empate: gana el primero en el orden de `SEPARADORES` porque `max` es estable
respecto al orden de inserción del diccionario.

### 4.2 `load_table` para CSV

```
1. Primer separador = sep si se pasó, si no detect_separator(primera línea)
2. Para cada codificación de ENCODINGS:
     leer con read_csv(path, sep=sep, encoding=cod)
     si filas > 0 y columnas > 1                      -> devolver
     si columnas == 1 y el nombre no lleva separador   -> devolver
   Si ninguna vale -> bucle heredado (4 separadores x 3 codificaciones)
   Si tampoco -> DataLoadError con la lista de intentos
```

La lectura en bytes de la primera línea es
`open(path, "rb").readline().decode("latin-1")`: `latin-1` no falla nunca y los
bytes de los delimitadores son los mismos en todas las codificaciones.

### 4.3 `load_table` para Excel

```
1. ext = Path(path).suffix.lower()
2. motor = {"xlsx": "openpyxl", "xlsm": "openpyxl",
            "xls": "xlrd", "ods": "odfpy"}.get(ext)
   Si no hay motor -> DataLoadError con la lista de extensiones
3. try: ExcelFile(path, engine=motor) y comprobar sheet en sheet_names
   except ImportError -> DataLoadError con el paquete que falta
4. pd.read_excel(path, sheet_name=sheet, engine=motor, header=0)
5. df vacío o sin columnas -> DataLoadError explicativo
```

`sheet` acepta índice o nombre; si no existe, `DataLoadError` con el nombre
de las hojas disponibles.

### 4.4 `list_sheets`

```python
with pd.ExcelFile(path, engine=motor) as xls:
    return list(xls.sheet_names)
```

Para CSV devuelve `[]`. La UI solo pregunta cuando la lista tiene más de un
elemento, así que `[]` y un único elemento se comportan igual.

## 5. `ui/file_dialogs.py`

```python
TABLE_FILTER = ("CSV (*.csv *.txt *.tsv);;"
                "Excel (*.xlsx *.xlsm *.xls);;"
                "OpenDocument (*.ods);;"
                "Todos los archivos (*)")

def choose_table_file(parent, caption="Seleccionar fichero de datos") -> str | None
def choose_sheet(parent, sheets: list[str]) -> str | None     # None = cancelado
def choose_save_path(parent, caption, default_name, filters) -> str | None
```

`choose_sheet` usa `QInputDialog.getItem` con la primera hoja por defecto y
devuelve `None` si el usuario cancela; en ese caso la UI no carga nada.

## 6. UI

- `ui/data_tab.py::load_csv` → se renombra a `load_data` (se actualiza la
  conexión del botón en el mismo fichero). Flujo: diálogo → `list_sheets` →
  `choose_sheet` si hace falta → `load_table` → `state.raw_df` + `source_path`
  → `data_loaded.emit()`.
- `ui/predict_tab.py::load_data` → mismo flujo con `load_table`.
- Cualquier excepción de `core` se muestra con `QMessageBox.critical`.

## 7. Dependencias y pruebas

`pyproject.toml` (nuevo, primer tarea del plan) declara:

```toml
[project]
requires-python = ">=3.9"
dependencies = [
  "PySide6>=6.5", "pandas>=2.0", "numpy>=1.24", "scikit-learn>=1.3",
  "scipy>=1.10", "matplotlib>=3.7", "joblib>=1.3",
  "openpyxl>=3.1", "xlrd>=2.0", "odfpy>=1.4",
]

[project.optional-dependencies]
smote = ["imbalanced-learn>=0.11"]
dev = ["pytest>=7.4", "black>=24.1", "isort>=5.13"]
```

Instalar en `entorno/` requiere confirmación del usuario (cambio de entorno).

Tests de Excel con `pytest.importorskip("openpyxl")` para que la suite siga
verde sin los motores instalados. Para `.xls`/`.ods` se generan fixtures en
memoria solo si el motor está disponible.

## 8. Nota sobre `.automl`

Sin cambios en `BUNDLE_VERSION`: la carga multiformato no altera el formato
del bundle.