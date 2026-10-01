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
│   ├── preprocessor.py
│   ├── model_trainer.py
│   └── state.py
├── ui/
│   ├── __init__.py
│   ├── main_window.py
│   ├── data_tab.py
│   ├── preprocess_tab.py
│   ├── train_tab.py
│   └── results_tab.py
├── resources/
├── tests/
├── pyproject.toml
├── AGENTS.md
└── README.md
```

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
| `target_column`   | `str | None`                | Nombre de la variable objetivo.          |
| `task_type`       | `"classification" | "regression" | None` | Tipo de tarea elegida.              |
| `pipeline`        | `sklearn.pipeline.Pipeline | None` | Pipeline completo entrenado.      |
| `trained_model`   | `Any | None`                | Alias del pipeline entrenado.            |
| `metrics`         | `dict`                      | Diccionario de métricas tras entrenar.   |

Si necesitas añadir un campo, hazlo aquí y documenta el cambio en el
README si afecta al flujo de usuario.

### 3.3 No mutar `raw_df`

El preprocesamiento **siempre parte de `state.raw_df.copy()`** y escribe el
resultado en `state.clean_df`. Nunca modificar `raw_df` in-place. Esto
permite al usuario rehacer el preprocesamiento sin recargar el CSV.

### 3.4 Submódulos por responsabilidad

- `data_loader.py` — solo E/S de CSV y resumen descriptivo.
- `preprocessor.py` — solo transformaciones puras `DataFrame -> DataFrame`.
- `model_trainer.py` — solo construcción de pipelines, entrenamiento y
  métricas.

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

### 5.2 Manejo de errores en la UI

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
