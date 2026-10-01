# AutoML_MEC Desktop

Aplicación de escritorio con Python y Qt (PySide6) que permite cargar un
archivo CSV, aplicar preprocesamiento y limpieza de datos, y entrenar
modelos de Machine Learning **sin que el usuario necesite escribir código**.

---

## Características

- **Carga de CSV** con autodetección de separador y codificación.
- **Resumen automático** del dataset: filas, columnas, valores nulos,
  tipos de datos, columnas numéricas y categóricas.
- **Preprocesamiento guiado** mediante checkboxes y combos:
  - Eliminación de filas duplicadas.
  - Eliminación de columnas con alto porcentaje de nulos.
  - Imputación de nulos (mediana, media, moda, valor constante o drop).
  - Eliminación manual de columnas.
- **Entrenamiento sin código** para tareas de clasificación y regresión
  con una selección de modelos clásicos de scikit-learn.
- **Visualización de resultados**: matriz de confusión para clasificación,
  gráfico real vs. predicho para regresión.
- **Exportación del modelo entrenado** a `.joblib` para reutilizarlo.
- **Arquitectura extensible**: pestañas independientes (Datos,
  Preprocesamiento, Entrenamiento, Resultados) comunicadas por señales Qt.

---

## Estructura del proyecto

```
automl_app/
├── main.py                 # Punto de entrada de la aplicación
├── core/                   # Lógica independiente de la GUI
│   ├── __init__.py
│   ├── data_loader.py      # Carga y validación de CSV
│   ├── preprocessor.py     # Limpieza y transformación de datos
│   ├── model_trainer.py    # Entrenamiento y evaluación de modelos
│   └── state.py            # Estado compartido de la aplicación
├── ui/                     # Widgets y ventanas Qt
│   ├── __init__.py
│   ├── main_window.py      # Ventana principal con QTabWidget
│   ├── data_tab.py         # Pestaña "1. Datos"
│   ├── preprocess_tab.py   # Pestaña "2. Preprocesamiento"
│   ├── train_tab.py        # Pestaña "3. Entrenamiento"
│   └── results_tab.py      # Pestaña "4. Resultados"
├── resources/
│   └── styles.qss          # Hoja de estilos opcional
├── tests/                  # Pruebas unitarias de core/
├── pyproject.toml          # Configuración del paquete
├── AGENTS.md               # Guía para agentes y contribuidores
└── README.md
```

La separación `core/` ↔ `ui/` es intencional: toda la lógica científica y
de machine learning debe poder probarse sin instanciar Qt.

---

## Requisitos

- Python **3.9 o superior**
- Sistema operativo: Linux, macOS o Windows

### Dependencias

```bash
pip install PySide6 pandas scikit-learn matplotlib joblib seaborn
```

Opcionales según el uso:

```bash
pip install xgboost lightgbm   # Modelos de boosting adicionales
pip install shap               # Explicabilidad
```

---

## Instalación

### Desde el código fuente (recomendado durante el desarrollo)

```bash
git clone https://github.com/sonoAESS/automl-desktop.git
cd automl-desktop
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e .
```

### Ejecución

```bash
python main.py
```

Se abrirá la ventana principal con las cuatro pestañas del flujo de trabajo.

---

## Flujo de uso

1. **Datos** — Pulsa *Cargar CSV…* y elige un archivo. La app detecta
   automáticamente el separador y muestra una previsualización con el
   resumen del dataset.
2. **Preprocesamiento** — Marca las operaciones que quieras aplicar
   (duplicados, nulos, columnas a eliminar) y pulsa *Aplicar
   preprocesamiento*. El resultado queda disponible en memoria.
3. **Entrenamiento** — Selecciona la variable objetivo, el tipo de tarea
   (clasificación o regresión) y el modelo. Pulsa *Entrenar modelo*.
4. **Resultados** — Revisa las métricas y el gráfico generado. Si te
   convence, guarda el modelo con *Guardar modelo (.joblib)*.

---

## Modelos disponibles

### Clasificación

- Regresión Logística
- Árbol de Decisión
- Random Forest
- SVM (con probabilidad habilitada)
- KNN

### Regresión

- Regresión Lineal
- Ridge
- Árbol de Decisión
- Random Forest
- SVR
- KNN

Todos los modelos se envuelven en un `Pipeline` de scikit-learn que aplica
imputación + escalado a variables numéricas y one-hot encoding a
categóricas, evitando *data leakage* entre entrenamiento y prueba.

### Métricas reportadas

| Tarea           | Métricas                                  |
|-----------------|-------------------------------------------|
| Clasificación   | accuracy, f1_macro, roc_auc (si aplica)   |
| Regresión       | rmse, mae, r2                             |

---

## Uso programático (sin GUI)

La lógica en `core/` es reutilizable en scripts o notebooks:

```python
from core import data_loader, preprocessor, model_trainer

df = data_loader.load_csv("datos.csv")
df = preprocessor.drop_duplicates(df)
df = preprocessor.fill_missing(df, numeric_strategy="median",
                               categorical_strategy="mode")

pipe, metrics, (X_test, y_test, y_pred) = model_trainer.train_model(
    df, target="target", model_name="Random Forest",
    task_type="classification",
)
print(metrics)
```

---

## Filosofía de diseño

Este proyecto adopta varias decisiones inspiradas en librerías científicas
maduras como MNE-Python:

- **Submódulos por responsabilidad**: `io`, `preprocessing`, `modeling`
  (aquí `data_loader`, `preprocessor`, `model_trainer`).
- **API unificada**: los objetos principales exponen métodos coherentes
  (`load`, `apply`, `train`, `save`).
- **Operaciones in-place controladas**: el preprocesamiento parte siempre
  de `state.raw_df` y produce `state.clean_df` sin mutar el original.
- **Integración con el ecosistema científico**: NumPy, pandas,
  scikit-learn, Matplotlib, joblib.
- **Separación estricta entre lógica y presentación**: `core/` no importa
  nada de `ui/`.

---

## Roadmap

- [ ] Búsqueda de hiperparámetros con `GridSearchCV` / `RandomizedSearchCV`
      en un `QThread`.
- [ ] Comparación simultánea de varios modelos en una tabla.
- [ ] Integración con PyCaret / FLAML para AutoML real.
- [ ] Explicabilidad con SHAP.
- [ ] Exportar a script Python reproducible desde la UI.
- [ ] Perfiles de preprocesamiento guardables en JSON/YAML.
- [ ] Empaquetado con PyInstaller para distribución como ejecutable.

---

## Contribuir

1. Crea una rama: `git checkout -b feature/mi-mejora`.
2. Asegúrate de que `core/` sigue siendo importable sin Qt.
3. Añade pruebas en `tests/`.
4. Abre un Pull Request describiendo el cambio.

Lee `AGENTS.md` antes de modificar el código si vas a usar un agente
automatizado.

---

## Licencia

MIT. Ver `LICENSE` para más detalles.
```
