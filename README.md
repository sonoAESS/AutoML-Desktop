# AutoML_MEC Desktop

Aplicación de escritorio con Python y Qt (PySide6) que permite cargar un
archivo CSV o de Excel, aplicar preprocesamiento y limpieza de datos, y
entrenar
modelos de Machine Learning **sin que el usuario necesite escribir código**.

---

## Identidad visual

Azul, rojo y blanco: los colores del Ministerio de Educación Superior de
Cuba, tomados del sitio institucional del proyecto ERCE. El escudo va en la
cabecera y como icono de la aplicación.

| | | |
|---|---|---|
| `marino` `#12223b` | `azul_med` `#14448c` | `azul_clar` `#446dab` |
| `acento` `#d34223` | `oro` `#81660d` | `fondo` `#f3f4f7` |

Los botones tienen tres clases —**principal**, secundario y de peligro— que se
asignan con `ui.theme.marcar(boton, "primario")`, no con estilos escritos a
mano.

Todos los colores viven en `ui/theme.py`: `COLORES` para la paleta y `TONOS`
para los tintes de hover, selección y deshabilitado. Si añades un color, va
ahí, no en el fichero donde se use.

Características

- **Carga de CSV, XLSX, XLS y ODS** con autodetección de separador y
  codificación (`utf-8-sig` → `cp1252` → `latin-1`). Si el libro de Excel
  tiene varias hojas, se elige cuál cargar.
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
- **Tabla de distribución de clases** en la pestaña de preprocesamiento:
  instancias y porcentaje de cada clase, **índice de desequilibrio (IR)**
  (clase mayoritaria / minoritaria, al estilo Weka), entropía de Shannon
  normalizada y el nivel de imbalance calculado.
- **Balanceo de clases** opcional al entrenar, con seis estrategias: ninguna,
  pesos de clase, submuestreo aleatorio, sobremuestreo aleatorio, SMOTE y
  SMOTEN. Solo en clasificación, y solo con modelos que lo admitan.
- **Selección de atributos** opcional antes de entrenar, con cuatro métodos
  (chi-cuadrado, ANOVA F, información mutua y embebido con bosque aleatorio),
  corte por número de atributos o por porcentaje, y un botón «Analizar
  selección» que muestra la puntuación de cada atributo sin llegar a entrenar.
- **Comparativa de modelos con el test de Friedman**: puntúa todos los modelos
  compatibles sobre los mismos bloques de validación cruzada, da su ranking por
  rangos medios, la diferencia crítica de Nemenyi y un post-hoc por pares con
  corrección de Holm, más el diagrama de diferencias críticas.
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
- **Pestaña Predicción**: carga un `.automl`, comprueba el esquema del
  fichero nuevo y devuelve predicciones con sus probabilidades.
- **Exportación con los datos dentro**: el fichero exportado (CSV o XLSX)
  contiene **el dataset de entrada completo** más las columnas `prediccion` y
  `prob_<clase>`, para poder analizar el resultado sin volver a unir nada a
  mano. El CSV se escribe en `utf-8-sig` con `;` para que Excel en español lo
  abra directamente.
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
│   ├── data_loader.py      # Carga de CSV, XLSX, XLS y ODS
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
│   ├── file_dialogs.py      # Diálogos compartidos de ficheros
│   └── workers.py           # Hilos de entrenamiento y búsqueda
├── resources/
│   └── styles.qss          # Hoja de estilos opcional
├── specs/                 # Especificaciones por funcionalidad (SDD)
├── tests/                  # Pruebas unitarias de core/
├── pyproject.toml          # Dependencias y configuración
├── AGENTS.md               # Guía para agentes y contribuidores
└── README.md
```

La separación `core/` ↔ `ui/` es intencional: toda la lógica científica y
de machine learning debe poder probarse sin instanciar Qt.

---

## Requisitos

- Python **3.10 o superior**
- Sistema operativo: Linux, macOS o Windows

### Dependencias

Las dependencias están declaradas en `pyproject.toml`:

```bash
pip install -e .            # núcleo + lectura de Excel (openpyxl, xlrd, odfpy)
pip install -e ".[smote]"   # añade imbalanced-learn para SMOTE
pip install -e ".[dev]"     # añade pytest, black, isort, coverage y xlwt
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

1. **Datos** — Pulsa *Cargar datos…* y elige un archivo. La app detecta
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
   plano y pulsa *Entrenar modelo*. Mientras busca, una barra animada indica
   que está trabajando, el texto de estado dice cuántas combinaciones y
   ajustes faltan y cuánto lleva, y los controles quedan bloqueados para no
   cambiar nada a medias. Al acabar verás *Búsqueda terminada en N s* (o
   *Búsqueda fallida*) y las métricas con los mejores parámetros.
4. **Resultados** — Revisa las métricas y el gráfico generado. Guarda el
   proyecto con *Guardar modelo y dataset (.automl)*: ese archivo contiene el
   modelo, sus metadatos y el dataset ya transformado. Con *Exportar solo el
   dataset* obtienes el CSV.
5. **Predicción** — Carga un `.automl` (solo si es de confianza: contiene
   código compilado), carga el fichero de entrada (CSV o Excel) y pulsa
   *Aplicar el modelo*.
   La app compara las columnas con las del entrenamiento, avisa de las que
   falten y devuelve las predicciones con sus probabilidades. Con
   *Exportar resultados* se guardan los datos de entrada **junto con** la
   predicción en un CSV (`utf-8-sig`, `;`) o en un XLSX. Si el fichero de
   entrada ya trae una columna `prediccion`, la nueva se llama
   `prediccion_2` y el aviso lo indica.

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

| Tarea           | Métricas                                        |
|-----------------|-------------------------------------------------|
| Clasificación   | exactitud, F1 (macro), AUC (ROC One-vs-Rest)   |
| Regresión       | RMSE, MAE, R²                                   |

Las tres métricas de clasificación se muestran **siempre**, también cuando el
AUC no se puede calcular (el modelo no da probabilidades, o el conjunto de
prueba solo tiene una clase): en ese caso la tabla de resultados muestra `n/d`
y el motivo, en lugar de esconder la métrica.

En multiclase el AUC es *One-vs-Rest* promediado (*macro*). El orden de las
columnas y los nombres de las clases se toman de `pipe.named_steps["model"].classes_`.

### Comparar modelos (test de Friedman)

Elegir el modelo "mejor" con una sola métrica sobre un solo reparto de datos
engaña: el ganador puede ser suerte de esa partición. La comparativa puntúa
**todos los modelos compatibles sobre los mismos bloques** y pregunta si las
diferencias entre ellos son más de lo que se esperaría por azar.

Cómo se lee el resultado:

| Concepto | Qué significa |
|---|---|
| **Puntuación media** | La métrica elegida (exactitud, RMSE…) promediada sobre los bloques |
| **Rango medio** | Posición media en cada bloque; **menor rango = mejor modelo** |
| **Friedman χ², p** | Si `p < 0.05` las diferencias son significativas; si no, no se puede afirmar que un modelo sea mejor |
| **Diferencia crítica** | Rango que un modelo puede apartarse del mejor sin que se considere peor (Nemenyi) |
| **Equivalente** | `sí` si el rango cae dentro de la diferencia crítica del mejor |
| **Post-hoc** | Por qué par concreto difiere, con ajuste de Holm |

Detalles que importan:

- Los bloques están **pareados**: los mismos folds para todos los modelos, que
  es la condición de validez del test. Por eso un modelo que falle en un bloque
  se excluye entero con su motivo, en vez de compararse con menos datos.
- El post-hoc solo es concluyente si el test de Friedman resulta significativo;
  con `p` alto, los pares "significativos" son ruido y así se avisa.
- La comparación es **de solo lectura**: no toca el modelo entrenado ni sus
  métricas.

**Aviso de tiempo**: con 5 modelos, 5 folds y 3 repeticiones son 75 ajustes de
pipeline. Empieza con **2 repeticiones** y sube si el resultado queda justo.

---

### Selección de atributos

En datasets con muchas columnas, Quite sabe más quitando: se puede pedir que el
pipeline se quede solo con las más útiles. La selección ocurre **dentro** del
pipeline, después de codificar las variables, así que el conjunto de prueba
conserva todas las columnas y las métricas siguen siendo comparables.

| Método      | Qué es                          | Cuándo usarlo                                  | Coste   |
|-------------|---------------------------------|------------------------------------------------|---------|
| `chi2`      | Chi-cuadrado                    | Clasificación con valores no negativos          | Bajo    |
| `anova`     | Test F                          | Punto de partida: rápido y con datos limpios   | Bajo    |
| `mutual_info` | Información mutua              | Cuando sospechas relaciones no lineales        | Alto    |
| `embedded`  | Bosque aleatorio               | Buscas la mejor exactitud y aceptas el coste    | Muy alto|

Notas:

- `chi2` no admite datos negativos, así que el pipeline añade un `MinMaxScaler`
  antes de seleccionar. En regresión desaparece: no sirve para variables
  continuas.
- `embedded` no admite el corte por porcentaje, porque necesita conocer el
  número de columnas al construir el bosque; la opción aparece desactivada.
- «Analizar selección» ajusta el selector sobre la parte de entrenamiento y
  pinta la tabla de puntuaciones. No entrena el modelo final: sirve para ver
  qué se va a quedar antes de decidir.
- Si eliges `class_weight` como forma de balanceo, el control manual de pesos
  de clase desaparece de los hiperparámetros para que no se contradigan.

---

### Balanceo de clases

Cuando la distribución está desequilibrada, el entrenamiento puede
compensarlo. El balanceo vive **dentro del pipeline**, así que solo se aplica
a los datos de entrenamiento: el conjunto de prueba conserva su tamaño y sus
desequilibrios, y las métricas siguen siendo comparables.

| Estrategia         | Qué hace                                                        | Requiere    |
|--------------------|-----------------------------------------------------------------|-------------|
| `none`             | No hace nada (valor por defecto)                                 | —           |
| `class_weight`     | El modelo recibe pesos inversos a la frecuencia de cada clase     | —           |
| `random_under`     | Submuestrea la clase mayoritaria hasta igualarla con la menor      | —           |
| `random_over`      | Sobremuestrea replicando filas de la clase minoritaria            | —           |
| `smote`            | Crea ejemplos sintéticos entre vecinos de la clase minoritaria     | `imbalanced-learn` |
| `smoten`           | Como SMOTE, con variantes de entropía en los bordes            | `imbalanced-learn` |

Notas:

- Solo hay balanceo en **clasificación**; en regresión el desplegable no aparece.
- `class_weight` solo se ofrece en modelos que aceptan ese parámetro (no en KNN).
- Al elegir `class_weight` desaparece el control manual de pesos de clase de los
  hiperparámetros, para que no se contradigan.
- SMOTE necesita al menos 6 filas de la clase minoritaria; con menos, avisa en vez
  de fallar.
- La pestaña de resultados indica el efecto real: `Balanceo: smote (240 → 480 instancias)`.

**El dataset exportado no se balancea.** `export_df` y el `.automl` guardan los
datos tal y como salen del preprocesamiento, sin remuestrear ni duplicar filas:
el balanceo solo existe dentro del modelo y no altera los datos que descarga el
usuario.

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
