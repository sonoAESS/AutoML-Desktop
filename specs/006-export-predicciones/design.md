# SDD-006 — Diseño técnico: exportar entrada + predicción

Estado: **implementada** (2026-10-04); desviaciones en el registro de
`specs/README.md`.

## 1. Mapa de cambios

| Fichero | Cambio |
|---|---|
| `core/persistence.py` | `_prediction_columns`, `_free_names`, `predict_frame`, `predict` reescrito, `SchemaReport.renamed` |
| `core/data_loader.py` | `detect_separator` acepta `preference` (lo necesita la exportación) |
| `ui/predict_tab.py` | Usa `predict_frame`, botón renombrado, export CSV/XLSX |
| `tests/test_persistence.py` | Tests de unión, colisiones e idempotencia |

## 2. API resultante

```python
# core/persistence.py

def _prediction_columns(bundle: Bundle, X: pd.DataFrame) -> dict[str, np.ndarray]:
    """Columnas de predicción de un bundle: {nombre: valores}."""

def _free_names(existing, nuevos: dict[str, Any]) -> dict[str, str]:
    """Nombre libre para cada columna nueva: `prediccion`, `prediccion_2`…"""

def predict_frame(bundle: Bundle, df: pd.DataFrame, strict: bool = True
                  ) -> tuple[pd.DataFrame, SchemaReport]:
    """Devuelve `df` (tal cual se cargó) + columnas de predicción."""

def predict(bundle: Bundle, df: pd.DataFrame, strict: bool = True):
    """Igual que antes: (predicciones, report, X, y). Solo columnas nuevas."""
```

`predict` y `predict_frame` comparten **todo** el cálculo: `predict` llama a
`predict_frame`, se queda con las columnas añadidas y devuelve
`(frame[solo_nuevas], report, X, y)`. Así no hay dos implementaciones que
puedan divergir.

```python
def predict(bundle, df, strict=True):
    frame, report = predict_frame(bundle, df, strict=strict)
    X, y, _ = align_features(df, bundle.metadata, strict=strict)
    nuevas = [c for c in frame.columns if c not in set(df.columns)]
    return frame[nuevas], report, X, y
```

Cuidado: se llama a `align_features` dos veces (una dentro de `predict_frame` y
otra aquí). Se evita pasando `X, y` fuera de `predict_frame` mediante un helper
privado `_predictor(bundle, df, strict) -> (X, y, report, columnas)` que ambas
funciones usan. Diseño final:

```python
def _predictor(bundle, df, strict=True):
    X, y, report = align_features(df, bundle.metadata, strict=strict)
    modelo = bundle.pipeline
    predicciones = modelo.predict(X)
    columnas = {PREDICTION_COLUMN: predicciones}
    if hasattr(modelo, "predict_proba"):
        try:
            probas = modelo.predict_proba(X)
            clases = bundle.metadata.classes or getattr(modelo, "classes_", ())
            for indice, clase in enumerate(clases):
                columnas[f"prob_{clase}"] = probas[:, indice]
        except (AttributeError, ValueError, IndexError):
            pass
    return X, y, report, columnas

def predict(bundle, df, strict=True):
    X, y, report, columnas = _predictor(bundle, df, strict)
    nombres = _free_names(df.columns, columnas)
    salida = pd.DataFrame({nombres[n]: columnas[n] for n in nombres}, index=X.index)
    return salida, report, X, y

def predict_frame(bundle, df, strict=True):
    X, y, report, columnas = _predictor(bundle, df, strict)
    if X.shape[0] != len(df):
        raise ValueError(
            f"El modelo devolvió {X.shape[0]} predicciones para "
            f"{len(df)} filas de entrada."
        )
    nombres = _free_names(df.columns, columnas)
    frame = df.copy()
    renombradas = tuple((n, nombres[n]) for n in nombres if nombres[n] != n)
    for original in nombres:
        frame[nombres[original]] = columnas[original]
    return frame, replace(report, renamed=renombradas)
```

## 3. Nombres libres y colisiones

```python
def _free_names(existing, nuevos: dict) -> dict[str, str]:
    usados = set(map(str, existing))
    libres = {}
    for nombre in nuevos:
        candidato = nombre
        contador = 2
        while candidato in usados:
            candidato = f"{nombre}_{contador}"
            contador += 1
        usados.add(candidato)
        libres[nombre] = candidato
    return libres
```

`prediccion` → `prediccion_2` → `prediccion_3`… El sufijo empieza en 2 porque
`_2` es más legible que `_dup1` y coincide con lo habitual en pandas. La
columna original del usuario nunca se toca.

### 3.1 Cómo se comunica el renombrado

`SchemaReport` es un frozen dataclass; se le añade un campo con valor por
defecto, de modo que **todos** los constructores actuales siguen siendo válidos:

```python
@dataclass(frozen=True)
class SchemaReport:
    missing: tuple = ()
    unexpected: tuple = ()
    n_rows: int = 0
    n_converted: int = 0
    renamed: tuple = ()          # pares (nombre_pedido, nombre_asignado)

    def describe(self) -> str:
        ...
        if self.renamed:
           _pairs = ", ".join(f"{a} → {b}" for a, b in self.renamed)
            lineas.append(f"Columnas renombradas al añadir la predicción: {_pairs}")
```

Así la UI solo tiene que leer `report.describe()` y `report.renamed`, sin
lógica nueva.

## 4. Exportación

### 4.1 Separador

Se reutiliza `data_loader.detect_separator` con un parámetro nuevo:

```python
def detect_separator(header_line: str, preference=SEPARADORES) -> str | None:
```

Por defecto el comportamiento no cambia (orden de `SEPARADORES`, que empieza
por `,`). Para escribir se llama con
`preference=(";", ",", "\t", "|")` sobre la cabecera unida, de forma que en caso
de empate gana `;`, que es lo que Excel en español espera. Si no hay ningún
candidato con apariciones, se escribe con `,`.

### 4.2 `ui/predict_tab.py`

```python
self._preview  = DataFrame solo con columnas de predicción   # lo que se ve
self._predictions = frame entrada + predicción                # lo que se exporta
```

- `predict()` llama a `persistence.predict_frame`; si la entrada tiene la
  columna objetivo, se conserva intacta (no se calculan métricas: fuera de
  alcance).
- `_fill_table(self.table_salida, self._preview, max_rows=500)` mantiene la
  tabla legible (R-010).
- `lbl_resultado`: “N filas predichas: M columnas de entrada + K de predicción
  (exportables en un mismo fichero). {report.describe()}”.
- `export_predictions()`:
  ```python
  destino, _ = QFileDialog.getSaveFileName(
      self, "Exportar resultados (datos + predicción)",
      "predicciones.csv", "CSV (*.csv);;Excel (*.xlsx)")
  if not destino: return
  try:
      if destino.lower().endswith(".xlsx"):
          _exportar_excel(self._predictions, destino)
      else:
          _exportar_csv(self._predictions, destino)
  except Exception as e:
      QMessageBox.critical(self, "Error al exportar", str(e))
  ```
- `_exportar_csv` fija `encoding="utf-8-sig"`, `sep=` detectado e `index=False`.
- `_exportar_excel` comprueba `importlib.util.find_spec("openpyxl")` y, si falta,
  lanza un `RuntimeError` con el mensaje en español que la UI muestra en
  `QMessageBox.critical` antes de escribir nada.

Los helpers `_exportar_csv` / `_exportar_excel` son **funciones de módulo** de
`ui/predict_tab.py`: la escritura con los parámetros correctos es decisión de
presentación, pero la lógica de negocio (juntar columnas) vive en `core`. Con esto
`AGENTS.md` §3.5 se respeta: la UI no toma decisiones científicas, solo de
formato de fichero.

## 5. Compatibilidad

- `predict()` no cambia de firma ni de contenido → los tests actuales de
  `tests/test_persistence.py` siguen verdes sin tocarlos (C-004).
- `BUNDLE_VERSION` sigue en `1`: el bundle no cambia.
- Al ejecutarse después de 001, `openpyxl` ya es una dependencia declarada; aun
  así la exportación a `.xlsx` comprueba el paquete y falla con un mensaje claro
  en vez de escribir un fichero corrupto (decisión §9 de la spec).

## 6. Verificación

```bash
python -m pytest -q tests/test_persistence.py tests/test_data_loader.py
```