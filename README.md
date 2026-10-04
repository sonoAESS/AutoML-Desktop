# AutoML_MEC Desktop

Aplicación de escritorio con Python y Qt (PySide6) que permite cargar un
archivo CSV, aplicar preprocesamiento y limpieza de datos, y entrenar
modelos de Machine Learning **sin que el usuario necesite escribir código**.

---

## Características

- **Carga de CSV** con autodetección de separador y codificación.
- **Perfilado automático de cada columna**: tipo semántico (numérico,
  booleano, categórico, fecha, texto, identificador), cardinalidad, nulos,
  rango de valores y avisos de incompatibilidad.
- **Corrección manual del tipo de dato** desde la propia tabla, sin editar
  el CSV: cada columna se puede reinterpretar (p. ej. un ID como texto).
- **Sugerencias inteligentes**: recomienda la variable objetivo y el tipo de
  tarea (clasificación o regresión) a partir del perfil del dataset.
- **Preprocesamiento guiado** mediante checkboxes y combos:
  - Eliminación de filas duplicadas.
  - Eliminación de columnas con alto porcentaje de nulos.
  - Imputación de nulos (mediana, media, moda, valor constante o drop).
  - Conversión de tipos y expansión de fechas (año, mes, día).
  - Normalización por columna (min-max, estándar, logarítmica, ninguna).
  - Eliminación manual de columnas.
- **Modelos filtrados por compatibilidad**: solo se ofrecen los modelos que
  pueden usarse con esos datos (p. ej. se descartan los que necesitan más
  filas de las que hay) y se explica por qué se descartan.
- **Hiperparámetros filtrados por modelo**: cada modelo muestra solo los
  parámetros que admite, con valores por defecto, la opción «todos» o valores
  concretos; se puede buscar con `GridSearchCV` o `RandomizedSearchCV` en un
  `QThread`.
- **Visualización de resultados**: matriz de confusión para clasificación,
  gráfico real vs. predicho para regresión y curva de la búsqueda.
- **Guardado en un único archivo `.automl`** (modelo + metadatos + dataset ya
  transformado) y exportación del dataset a CSV.
- **Pestaña Predicción**: carga un `.automl`, comprueba el esquema del CSV
  nuevo y devuelve predicciones con sus probabilidades.
- **Arquitectura extensible**: pestañas independientes (Datos,
  Preprocesamiento, Entrenamiento, Resultados, Predicción) comunicadas por
  señales Qt.

---

## Estructura del proyecto

```
automl_app/
├── main.py                 # Punto de entrada de la aplicación
├── core/                   # Lógica independiente de la GUI
│   ├── __init__.py
│   ├── data_loader.py      # Carga y validación de CSV
│   ├── profiling.py        # Perfil semántico y sugerencias
│   ├── preprocessor.py     # Limpieza, tipos y normalización
│   ├── model_specs.py      # Catálogo de modelos e hiperparámetros
│   ├── model_trainer.py    # Entrenamiento y evaluación de modelos
│   ├── persistence.py      # Guardado/carga de .automl y predicción
│   └── state.py            # Estado compartido de la aplicación
├── ui/                     # Widgets y ventanas Qt
│   ├── __init__.py
│   ├── main_window.py      # Ventana principal con QTabWidget
│   ├── data_tab.py         # Pestaña "1. Datos"
│   ├── preprocess_tab.py   # Pestaña "2. Preprocesamiento"
│   ├── train_tab.py        # Pestaña "3. Entrenamiento"
│   ├── results_tab.py      # Pestaña "4. Resultados"
│   ├── predict_tab.py      # Pestaña "5. Predicción"
│   └── workers.py          # Hilos de entrenamiento y búsqueda
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

Se abrirá la ventana principal con las cinco pestañas del flujo de trabajo.

---

## Flujo de uso

1. **Datos** — Pulsa *Cargar CSV…* y elige un archivo. La app detecta
   automáticamente el separador, perfila cada columna y sugiere una variable
   objetivo. Si una columna está mal interpretada (un ID como texto libre,
   por ejemplo), cámbiala con el desplegable de su fila.
2. **Preprocesamiento** — Marca las operaciones que quieras aplicar
   (duplicados, nulos, columnas a eliminar) y, si lo necesitas, convierte
   tipos o normaliza columnas. Pulsa *Aplicar preprocesamiento*.
3. **Entrenamiento** — Se propone el objetivo y el tipo de tarea; puedes
   cambiarlos. Elige la familia y el modelo: solo aparecen los compatibles
   con tus datos, y el panel muestra los hiperparámetros propios de ese
   modelo. Activa *Buscar mejores hiperparámetros* para afinarlo en segundo
   plano y pulsa *Entrenar modelo*.
4. **Resultados** — Revisa las métricas y el gráfico generado. Guarda el
   proyecto con *Guardar modelo y dataset (.automl)*: ese archivo contiene el
   modelo, sus metadatos y el dataset ya transformado. Con *Exportar solo el
   dataset* obtienes el CSV.
5. **Predicción** — Carga un `.automl` (solo si es de confianza: contiene
   código compilado), carga el CSV de entrada y pulsa *Aplicar el modelo*.
   La app compara las columnas con las del entrenamiento, avisa de las que
   falten y devuelve las predicciones con sus probabilidades.

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
las conversiones de tipo, la normalización elegida, la imputación y el one-hot
encoding, evitando *data leakage* entre entrenamiento y prueba: los valores
usados para normalizar e imputar se aprenden solo con el conjunto de
entrenamiento.

El SVM usa `CalibratedClassifierCV` para ofrecer probabilidades sin depender
de `probability=True`.

### Métricas reportadas

| Tarea           | Métricas                                  |
|-----------------|-------------------------------------------|
| Clasificación   | accuracy, f1_macro, roc_auc (si aplica)   |
| Regresión       | rmse, mae, r2                             |

---

## Uso programático (sin GUI)

La lógica en `core/` es reutilizable en scripts o notebooks:

```python
from core import data_loader, model_trainer, persistence, profiling

df = data_loader.load_csv("datos.csv")

# Perfil y sugerencias
for profile in profiling.profile_dataframe(df):
    print(profile.name, profile.label, profile.value_range)

pipe, metrics, (X_test, y_test, y_pred) = model_trainer.train_model(
    df, target="target", model_name="Random Forest",
    task_type="classification",
    casts={"fecha_alta": "fecha"}, normalizations={"sueldo": "standard"},
)
print(metrics)

# Guardar modelo + dataset transformado y aplicarlo a datos nuevos
export = model_trainer.transform_with_pipeline(df, pipe)
metadata = persistence.build_metadata(
    pipe, target_column="target", task_type="classification",
    model_name="Random Forest", metrics=metrics, df=export,
)
persistence.save_bundle("modelo.automl", pipe, metadata, dataset=export)

bundle = persistence.load_bundle("modelo.automl")
predicciones, report, _, _ = persistence.predict(bundle, nuevos_datos)
```

### Sobre el archivo `.automl`

`.automl` es un `joblib` que incluye el pipeline y los metadatos del
entrenamiento. **Cargar un `.automl` ejecuta código**: abre solo archivos
propios o de confianza. Si solo necesitas el dataset, exporta el CSV.

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

- [x] Búsqueda de hiperparámetros con `GridSearchCV` / `RandomizedSearchCV`
      en un `QThread`.
- [x] Perfilado del dataset, corrección de tipos y sugerencia de objetivo.
- [x] Modelos e hiperparámetros filtrados por tipo de tarea y datos.
- [x] Guardado del proyecto en `.automl` (modelo + dataset) y predicción.
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
