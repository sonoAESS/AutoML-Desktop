Guía para agentes automatizados (LLM, asistentes de código) y personas que
contribuyan a **AutoML Desktop**. Este archivo describe la arquitectura, las
convenciones y las reglas que **deben respetarse** al modificar el proyecto.

Si eres un agente que va a tocar código, lee este documento completo antes
de hacer cambios.

---

## 1. Visión general del proyecto

AutoML Desktop es una aplicación de escritorio con **PySide6** que permite a
usuarios sin experiencia en programación:

1. Cargar un CSV.
2. Preprocesarlo y limpiarlo.
3. Entrenar modelos de ML de clasificación o regresión.
4. Visualizar resultados y exportar el modelo.

La arquitectura sigue un patrón en capas, inspirado en librerías científicas
como MNE-Python:

- **`core/`** — Lógica pura de datos y ML. **No importa Qt.**
- **`ui/`** — Widgets y ventanas Qt. Consume `core/` pero no contiene lógica
  científica.
- **`main.py`** — Punto de entrada.

---

## 2. Estructura de archivos

```
automl_app/
├── main.py
├── core/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── profiling.py
│   ├── preprocessor.py
│   ├── model_specs.py
│   ├── model_trainer.py
│   ├── balancing.py
│   ├── feature_selection.py
│   ├── comparison.py
│   ├── persistence.py
│   └── state.py
├── ui/
│   ├── __init__.py
│   ├── main_window.py
│   ├── data_tab.py
│   ├── preprocess_tab.py
│   ├── train_tab.py
│   ├── results_tab.py
│   ├── predict_tab.py
│   ├── file_dialogs.py
│   ├── theme.py
│   ├── plotting.py
│   ├── layout.py
│   ├── sidebar.py
│   ├── enabled.py
│   ├── empty_state.py
│   └── workers.py
├── resources/            escudo e icono (solo binarios)
├── specs/
│   ├── README.md
│   └── NNN-slug/
│       ├── spec.md
│       ├── design.md
│       └── tasks.md
├── tests/
├── pyproject.toml
├── AGENTS.md
└── README.md
```

`ui/file_dialogs.py` contiene únicamente los diálogos compartidos de selección
de ficheros (filtro multiformato, elección de hoja, destino de exportación). No
es un módulo "catch-all": si alguna función no trata con diálogos de ficheros,
no pertenece aquí.

Cualquier archivo nuevo debe encajar en esta organización. **No crear
módulos "catch-all"** del tipo `utils.py` o `helpers.py` sin justificarlo.

---

## 3. Reglas de arquitectura (obligatorias)

### 3.1 `core/` no depende de `ui/` ni de Qt

**Regla dura**: ningún archivo en `core/` puede importar `PySide6`, `PyQt`,
ni ningún submódulo de `ui/`.

Motivo: la lógica debe poder ejecutarse en un notebook, un script o un
servidor sin Qt instalado. Esto es exactamente lo que hace MNE-Python con
sus submódulos puros.

Comprobación rápida antes de un commit:

```bash
grep -R "PySide6\|PyQt\|from ui\." core/
```

El comando no debe devolver resultados.

### 3.2 Estado compartido vía `core/state.AppState`

Toda comunicación entre pestañas **debe pasar por `AppState`** o por
señales Qt (`Signal`). No guardes datos en atributos globales ni en
variables de módulo.

Campos actuales de `AppState`:

| Campo             | Tipo                        | Descripción                              |
|-------------------|-----------------------------|------------------------------------------|
| `raw_df`          | `pd.DataFrame | None`       | Datos tal como se cargaron del CSV.      |
| `clean_df`        | `pd.DataFrame | None`       | Datos después del preprocesamiento.      |
| `export_df`       | `pd.DataFrame | None`       | Dataset transformado con el pipeline.    |
| `profiles`        | `list[ColumnProfile]`       | Perfil semántico por columna.            |
| `column_types`    | `dict[str, str]`            | Tipos forzados por el usuario.           |
| `normalizations`  | `dict[str, str]`            | Normalización elegida por columna.        |
| `target_column`   | `str | None`                | Nombre de la variable objetivo.          |
| `task_type`       | `"classification" | "regression" | None` | Tipo de tarea elegida.              |
| `family`          | `str | None`                 | Familia del modelo elegida.              |
| `model_name`      | `str | None`                 | Modelo elegido.                          |
| `pipeline`        | `sklearn.pipeline.Pipeline | None` | Pipeline completo entrenado.      |
| `trained_model`   | `Any | None`                | Alias del pipeline entrenado.            |
| `metrics`         | `dict`                      | Diccionario de métricas tras entrenar.   |
| `bundle`          | `persistence.Bundle | None`  | Proyecto `.automl` cargado.              |
| `bundle_path`     | `str | None`                | Ruta del último `.automl` guardado/cargado.|
| `source_path`     | `str | None`                | Ruta del CSV cargado.                    |

Si necesitas añadir un campo, hazlo aquí y documenta el cambio en el
README si afecta al flujo de usuario.

### 3.3 No mutar `raw_df`

El preprocesamiento **siempre parte de `state.raw_df.copy()`** y escribe el
resultado en `state.clean_df`. Nunca modificar `raw_df` in-place. Esto
permite al usuario rehacer el preprocesamiento sin recargar el CSV.

### 3.4 Submódulos por responsabilidad

- `data_loader.py` — solo E/S de CSV y resumen descriptivo.
- `profiling.py` — solo perfil semántico de columnas y sugerencias.
- `preprocessor.py` — solo transformaciones puras `DataFrame -> DataFrame`.
- `model_specs.py` — solo el catálogo declarativo de modelos, familias,
  hiperparámetros y reglas de compatibilidad.
- `model_trainer.py` — solo construcción de pipelines, entrenamiento y
  métricas.
- `balancing.py` — solo distribución de clases, catálogo de estrategias y el
  paso `BalancedSampler` que equilibra el pipeline.
- `feature_selection.py` — solo catálogo de métodos de selección, construcción
  del selector de scikit-learn y lectura de qué atributos se conservaron.
  El selector se ajusta **dentro** del pipeline, nunca sobre `clean_df` ni sobre
  el conjunto de prueba.
- `comparison.py` — solo comparativa de modelos: bloques pareados, rangos,
  test de Friedman, post-hoc con corrección de Holm y diferencia crítica.
  No dibuja el diagrama (eso vive en `ui/`) y es de solo lectura sobre
  `export_df`: nunca modifica `state.pipeline` ni `state.metrics`.
- `persistence.py` — solo guardado/carga de `.automl`, validación de esquema
  y predicción.

No mezcles responsabilidades: `model_trainer.py` no debe abrir archivos ni
`data_loader.py` debe conocer scikit-learn.

### 3.5 La UI no contiene lógica científica

En `ui/` solo se permite:

- Construir widgets.
- Leer entradas del usuario (combos, checkboxes, botones).
- Llamar a funciones de `core/`.
- Mostrar resultados y errores.

**Prohibido** en `ui/`:

- Hacer `pd.read_csv` directamente (usar `data_loader.load_csv`).
- Aplicar `fillna`, `get_dummies`, etc. (usar `preprocessor`).
- Instanciar modelos de scikit-learn (usar `model_trainer`).

---

## 4. Convenciones de código

- **Python**: 3.9+ (usa `list[str]`, `dict[str, ...]` con `from __future__
  import annotations` si hace falta compatibilidad).
- **Formato**: [black](https://black.readthedocs.io/) con longitud 88.
- **Imports**: ordenados con `isort` (perfil `black`).
- **Nombres**:
  - Módulos y funciones: `snake_case`.
  - Clases: `PascalCase`.
  - Constantes de módulo (`CLASSIFIERS`, `REGRESSORS`): `UPPER_SNAKE_CASE`.
- **Docstrings**: estilo Google, en español o inglés, pero **consistente
  dentro de un mismo archivo**.
- **Idioma de la UI**: **español**. Los mensajes al usuario deben estar en
  español. Los identificadores de código permanecen en inglés.

---

## 5. Patrones a respetar

### 5.4 La navegación va por señales, nunca subiendo al padre

**Regla dura**: ninguna pestaña accede a `self.window()`, a
`centralWidget()` ni al índice de otra pestaña. Para pedir un cambio de
paso se emite una señal y la ventana decide:

```python
class ResultsTab(QWidget):
    navigate_requested = Signal(str)   # "prediccion", "entrenamiento"…

# MainWindow
self.results_tab.navigate_requested.connect(self.ir_a_paso_nombre)
```

El motivo es concreto: el código antiguo hacía
`window.centralWidget().indexOf(window.predict_tab)`, y con el contenido en un
`QStackedWidget` ese `hasattr(tabs, "setCurrentIndex")` seguía siendo cierto
para el widget equivocado: el botón habría saltado en silencio a otra
pestaña. Hay un test que falla si `self.window()` aparece fuera de
`main_window.py`.

### 5.5 Layout: scroll y alturas acotadas

- Toda pestaña va envuelta con `ui.layout.envolver_scroll()`. El problema
  medido era que **no había scroll** y las tablas sin `setMaximumHeight`
  empujaban los botones fuera de la ventana.
- Toda tabla y lista nueva pasa por `ui.layout.acotar_tabla()`.
- Los lienzos de matplotlib usan `ui.layout.expandir_canvas()` y llaman a
  `tight_layout()` antes de dibujar.
- Los bloques largos van en un `QToolBox` (acordeón) y las acciones
  principales **fuera** del acordeón, siempre visibles.

### 5.1 Comunicación entre pestañas

Usa **señales Qt** para notificar cambios de estado, y que el receptor
consulte `AppState`:

```python
class DataTab(QWidget):
    data_loaded = Signal()      # se emite tras cargar un CSV

    def load_csv(self):
        ...
        self.data_loaded.emit()
```

Conexiones típicas en `MainWindow`:

```python
self.data_tab.data_loaded.connect(self.preprocess_tab.refresh)
self.data_tab.data_loaded.connect(self.train_tab.refresh)
self.preprocess_tab.data_processed.connect(self.train_tab.refresh)
self.train_tab.model_trained.connect(self.results_tab.refresh)
```

**Nunca** llamar directamente a métodos de otra pestaña para pasar datos;
usa siempre `AppState` + señal.

### 5.2 El estado de los controles se deriva de `AppState`

**Regla dura**: cada pestaña tiene un `_update_enabled_state()` que **solo lee
`self.state`**. No mira la visibilidad de otro widget, ni el texto de una
etiqueta, ni lo que pasó en la última operación.

Motivo: `preprocess_tab` no tenía ni una llamada a `setEnabled`, así que
mostraba trece controles activos para un dataset que no estaba cargado.

Reglas asociadas:

- Un control apagado **explica por qué** en su `toolTip` (R-010). Un control
  apagado sin explicación parece una aplicación rota.
- Si un grupo se desactiva, sus hijos también: `ui.enabled.habilitar_hijos()`.
- Los errores se siguen mostrando con `QMessageBox`; deshabilitar no sustituye
  a avisar (§5.2 siguiente).
- El texto visible de un botón no pasa de **28 caracteres**; el detalle largo va
  al tooltip.

### 5.6 Una pestaña sin datos explica qué falta

**Regla**: cada pestaña tiene un `EmptyState` (`ui/empty_state.py`) que sustituye
al contenido vacío. Una tabla con su cabecera y ninguna fila no dice nada, y
hace que la aplicación parezca rota.

El patrón es siempre el mismo:

```python
def _alternar_vacio(self):
    alternar_estado_vacio(self.vacio, (self.table, self.canvas), self.state.raw_df is None)
```

Tres reglas que los tests comprueban:

- El mensaje **nombra el paso siguiente** («Entrena un modelo en el paso 3»), y
  el número tiene que existir en la barra lateral.
- **Las acciones nunca se ocultan**: «Aplicar preprocesamiento», «Entrenar
  modelo» y «Aplicar el modelo» se quedan fuera del alternador.
- Se oculta con `setVisible(False)`, nunca sacando el widget del layout: los
  tests de UI los localizan por atributo.

El mensaje **no lleva botón**: un botón dentro del estado vacío que salte a
otra pestaña tendría que tocar el `QStackedWidget` del padre, que es justo el
acoplamiento que §5.4 elimina. El texto dice qué hacer y la barra lateral lo
hace.

### 5.3 Manejo de errores en la UI

Toda operación que pueda fallar debe ir envuelta en `try/except Exception`
y mostrar un `QMessageBox.critical(self, "Error", str(e))`. **No dejar que
un traceback llegue al usuario final.**

### 5.3 Bloqueo de la UI

**Regla**: nunca ejecutes entrenamiento, preprocesamiento pesado ni
búsqueda de hiperparámetros en el hilo principal.

- `train_model()` puede ejecutarse síncronamente si el dataset es pequeño,
  pero cualquier llamada a `tune_model()` **debe** hacerse desde
  `ui/workers.WorkerThread`.
- Los workers viven en `ui/workers.py`, extienden `QObject` y se comunican
  con la UI solo por señales (`finished`, `failed`, `progress`).
- No importes `core` directamente desde dentro de un worker para ejecutar
  lógica: el worker solo orquesta, la ciencia vive en `core/`.

---

## 6. Pruebas

- Ubicación: `tests/`.
- Framework: `pytest`.
- Regla: **todo lo de `core/` debe ser testeable sin Qt**. Si una función
  de `core/` necesita una app Qt para funcionar, está mal diseñada.
- Cobertura mínima esperada para `core/`: 80 %.

Ejemplo de prueba:

```python
# tests/test_preprocessor.py
import pandas as pd
from core import preprocessor

def test_fill_missing_median():
    df = pd.DataFrame({"a": [1.0, None, 3.0]})
    out = preprocessor.fill_missing(df, numeric_strategy="median")
    assert out["a"].isna().sum() == 0
    assert out["a"].iloc[1] == 2.0
```

Ejecutar:

```bash
pytest -q
```

---

## 7. Reglas para agentes automatizados

Si eres un agente (Claude, GPT, etc.) modificando este repositorio:

1. **Lee primero** `AGENTS.md` y `README.md`.
2. **No rompas** la separación `core/` ↔ `ui/`. Es la regla más importante.
3. **No introduzcas dependencias** nuevas en `pyproject.toml` sin
   justificarlo en la descripción del cambio. Evita añadir paquetes
   pesados (torch, tensorflow) salvo petición explícita.
4. **No cambies la API pública** de `core/` (nombres de funciones,
   firmas) sin actualizar el README y las pruebas.
5. **Añade pruebas** en `tests/` para cualquier función nueva en `core/`.
6. **Mantén los mensajes de la UI en español**.
7. **Ejecuta** antes de terminar:
   ```bash
   grep -R "PySide6\|PyQt\|from ui\." core/   # no debe devolver nada
   pytest -q
   ```
8. **Commits**: mensajes en imperativo, en español o inglés, con prefijo
   convencional (`feat:`, `fix:`, `refactor:`, `docs:`, `test:`).
9. **No generes archivos** fuera de la estructura definida en la sección 2.
10. Si una tarea requiere cambiar la arquitectura, **propón el cambio
    primero** en la descripción del PR en lugar de aplicarlo directamente.

---

## 8. Checklist antes de un Pull Request

- [ ] `core/` no importa Qt ni `ui/`.
- [ ] Las funciones nuevas de `core/` tienen pruebas en `tests/`.
- [ ] La UI solo llama a `core/`, no contiene lógica científica.
- [ ] Los errores se muestran con `QMessageBox`, no como tracebacks.
- [ ] El código pasa `black` e `isort`.
- [ ] `pytest -q` pasa.
- [ ] El README se actualizó si cambió el flujo de usuario o la API.
- [ ] Los mensajes al usuario están en español.

---

## 9. Glosario

| Término            | Significado                                                     |
|--------------------|-----------------------------------------------------------------|
| **core**           | Lógica pura de datos y ML, sin Qt.                              |
| **ui**             | Widgets y ventanas PySide6.                                     |
| **AppState**       | Objeto único que comparte datos entre pestañas.                 |
| **Pipeline**       | `sklearn.pipeline.Pipeline` con preprocesamiento + modelo.      |
| **Tab**            | Pestaña del `QTabWidget` principal (Datos, Preprocesamiento…).  |
| **Señal**          | `Signal` de Qt usada para notificar cambios entre pestañas.     |
