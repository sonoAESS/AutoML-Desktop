# Guía de uso de AutoML Desktop

Esta guía explica cómo usar la aplicación de principio a fin, con tres
ejemplos reales que están en la carpeta `example/data/`:

| Ejemplo | Archivo | Qué vas a aprender |
|---|---|---|
| Clasificación binaria | `heart.csv` | El flujo completo: cargar, limpiar, entrenar, guardar y predecir |
| Regresión | `ford.csv` | Elegir un objetivo continuo e interpretar RMSE, MAE y R² |
| Multiclase | `Mobiles.csv` | Selección de atributos, comparativa de modelos y balanceo |

No hace falta saber programar ni escribir código. Todo se hace con botones,
desplegables y casillas.

> Los tres archivos están en la carpeta `example/data/` del repositorio. Si
> usas el ejecutable y no los tienes, cualquier CSV con una columna objetivo
> similar sirve para seguir los mismos pasos.



> Si buscas información para modificar el programa, consulta el
> [`README.md`](README.md) y [`AGENTS.md`](AGENTS.md). Esta guía es solo para
> quien va a **usar** la aplicación.

---

## 1. Abrir la aplicación

### Si tienes el programa compilado

Descarga `AutoML-Desktop-windows-x64.zip` o
`AutoML-Desktop-linux-x86_64.tar.gz` desde la página de releases y
**descomprime la carpeta entera**. Ejecuta el programa desde dentro de esa
carpeta: el ejecutable necesita la carpeta `_internal/` que tiene al lado.
Sacar solo el `.exe` deja un programa que no arranca.

No hace falta instalar Python.

### Si ejecutas desde el código fuente

Necesitas Python 3.12 y las dependencias del proyecto:

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[smote]"
python main.py
```

El extra `[smote]` solo es necesario si vas a usar SMOTE o SMOTE-NC como
balanceo. Sin él, esas dos opciones fallan al entrenar.

### Formatos de archivo admitidos

- **Texto delimitado**: `.csv`, `.txt` y `.tsv`. Detecta automáticamente el
  separador (`,`, `;`, tabulador, `|`) y la codificación (`utf-8-sig`,
  `cp1252`, `latin-1`).
- **Excel**: `.xlsx`, `.xlsm` y `.xls`. Si el libro tiene varias hojas, la
  aplicación pregunta cuál cargar.
- **OpenDocument**: `.ods`.

El formato `.xlsb` no está admitido. Ábrelo en Excel y guárdalo como `.xlsx`.

---

## 2. La ventana en un vistazo

```
┌────────────────────────────────────────────────────────────┐
│  Paso 2 de 5 · Preprocesamiento                            │  cabecera
├──────────────┬─────────────────────────────────────────────┤
│ AutoML       │  Calidad (duplicados y nulos)          ▸    │
│ Desktop      │  Tipos de datos y fechas               ▸    │  acordeón
│              │  Normalización de variables            ▸    │
│ ✓ Datos      │  Columnas a eliminar                   ▸    │
│ ▸ Preproces. │  Distribución de clases                ▸    │
│   Entrenam.  │                                             │
│   Resultados │  [ Aplicar preprocesamiento ]               │  acción fija
│   Predicción │                                             │
├──────────────┴─────────────────────────────────────────────┤
│ Dataset preprocesado. Ya puedes entrenar en el paso 3.     │  barra de estado
└────────────────────────────────────────────────────────────┘
```

- La **barra lateral** es la navegación. El paso actual se resalta; los que ya
  completaste llevan un **✓**. Puedes hacer clic en cualquier paso.
- La **cabecera** dice siempre en qué paso estás.
- La **barra de estado**, abajo, resume lo que acaba de ocurrir.
- Los grupos de Preprocesamiento y Entrenamiento son un **acordeón**: se abre
  uno cada vez. La acción principal («Aplicar preprocesamiento», «Entrenar
  modelo») queda siempre fuera y a la vista.
- En Resultados hay tres **sub-pestañas**: *Métricas*, *Comparativa* y
  *Exportar*.

### Atajos de teclado

| Atajo | Acción |
|---|---|
| `Ctrl+1` … `Ctrl+5` | Ir al paso 1 a 5 |
| `Ctrl+Tab` | Avanzar al paso siguiente (vuelve al 1 al final) |

### Cuando un botón está apagado

Un control deshabilitado **explica el motivo en su tooltip**: pasa el ratón por
encima. Por ejemplo, «Entrenar modelo» está apagado hasta que haya dataset
limpio, objetivo y modelo elegidos. Esto es normal, no es un fallo.

---

## 3. Tutorial 1: clasificación con `heart.csv`

**Objetivo:** predecir si un paciente tiene una enfermedad cardíaca
(`HeartDisease`: `0` = no, `1` = sí).

El archivo tiene **918 filas y 12 columnas**, sin valores nulos y sin filas
duplicadas. La clase está razonablemente repartida: 508 casos «1» (55 %) y 410
casos «0» (45 %).

### Paso 1 · Datos

1. Pulsa **Cargar datos…** y elige `example/data/heart.csv`.
2. La línea de información, junto al botón de carga, mostrará algo como
   `heart.csv · 918 filas · 12 columnas · 0 valores nulos`.
3. Debajo aparece la **tabla de perfil**, con una fila por columna:

   | Columna | Tipo detectado | Dtype real | Únicos | % nulos | Tarea sugerida |
   |---|---|---|---|---|---|
   | Age | Numérico | int64 | 50 | 0 % | regresión |
   | Sex | Categórico | object | 2 | 0 % | clasificación |
   | ChestPainType | Categórico | object | 4 | 0 % | clasificación |
   | HeartDisease | Numérico | int64 | 2 | 0 % | clasificación |

   - **Tipo detectado** es cómo interpreta la aplicación la columna:
     numérico, booleano, categórico, fecha, texto o identificador.
   - **Dtype real** es el tipo que trae el archivo. Si no coinciden, suele
     hacer falta una corrección.
   - **Únicos** y **% nulos** avisan de columnas con demasiada variedad o
     demasiados huecos.
   - **Tarea sugerida** indica si esa columna serviría para clasificación o
     para regresión.

4. **Corregir un tipo** (opcional): cambia el desplegable de la columna. Por
   ejemplo, si un identificador numérico como `id` aparece como *Numérico*,
   cámbialo a *Identificador* para que no se use como variable predictora. Con
   **Volver a detectar tipos** se descartan todas las correcciones manuales.

5. **Sobre la variable objetivo:** la aplicación muestra una sugerencia en
   forma de texto, pero no es infalible: elige la columna con menos valores
   distintos. En `heart.csv` propondrá `Sex` (2 valores), que **no** es lo que
   queremos predecir.

   La sugerencia es solo un aviso: el objetivo se confirma en el desplegable
   **Variable objetivo** del paso 3. Elige ahí `HeartDisease`. Si prefieres
   aceptar la sugerencia, usa el botón **Usar objetivo sugerido** cuando esté
   disponible y cambia la columna después.

6. La **vista previa** muestra las primeras 200 filas. No es editable: solo
   sirve para comprobar que el archivo se ha leído bien.

> Consejo: si una columna de texto contiene números («1.200», «3,5»), la
> aplicación lo avisa en el tooltip de la columna. Puedes convertirla a
> *Numérico* desde el desplegable.

### Paso 2 · Preprocesamiento

Abre el acordeón **Calidad (duplicados y nulos)**. Para este archivo puedes
dejar los valores por defecto:

- **Eliminar filas duplicadas**: marcado. Aquí no hay duplicadas, así que no
  cambia nada.
- **Eliminar columnas con >50 % nulos**: marcado. Aquí no hay nulos.
- **Estrategia numérica**: `median` (rellena huecos con la mediana).
- **Estrategia categórica**: `mode` (rellena con el valor más frecuente).

En **Normalización de variables** verás una fila por columna numérica
(`Age`, `RestingBP`, `Cholesterol`, `FastingBS`, `MaxHR`, `Oldpeak`). La
normalización es opcional: los modelos de conjuntos y los árboles no la
necesitan para funcionar. Si quieres probarla, elige `estándar (z-score)` en
`Age` y `Cholesterol`.

En **Distribución de clases** aparece la tabla del objetivo **si ya está
elegido**. Si todavía no lo has hecho, el bloque permanece oculto: elige
`HeartDisease` en el paso 3 y vuelve a este paso para verlo. Para
`HeartDisease` mostrará:

```
2 clases · 918 instancias · IR = 1.24 (n_mayor/n_minor) ·
desbalance equilibrado · entropía normalizada 0.99
```

- **IR** (índice de desbalance) es el cociente entre la clase mayoritaria y la
  minoritaria. Por debajo de 1,5 se considera *equilibrado*; entre 1,5 y 3,
  *moderado*; por encima de 3, *severo*.
- Aquí no hace falta balancear.

Pulsa **Aplicar preprocesamiento**. Deberías ver:

```
Listo: 918 filas × 12 columnas (este es el dataset que se usará para
entrenar y se guardará junto al modelo).
```

> Si cambias de opinión, vuelve a este paso y pulsa otra vez el botón. El
> preprocesamiento siempre parte del archivo original, así que no se acumulan
> transformaciones.

### Paso 3 · Entrenamiento

En el acordeón **Qué modelo entrenar**:

1. **Variable objetivo:** elige `HeartDisease`.
2. **Tipo de tarea:** `Clasificación` (se selecciona solo al elegir una
   columna con pocos valores distintos).
3. **Familia de modelo:** `Conjuntos de modelos`.
4. **Modelo:** `Random Forest`.

Debajo del formulario, la línea de compatibilidad explica qué modelos se han
descartado y por qué. Por ejemplo, si el dataset tuviera menos de 50 filas, los
SVM no aparecerían.

En **Hiperparámetros**, `Random Forest` ofrece:

| Parámetro | Valores | Qué controla |
|---|---|---|
| Número de árboles | 100, 200, 400 | Más árboles = más estable, más lento |
| Profundidad máxima | ninguno, 10, 20 | Cuánto puede crecer cada árbol |
| Mínimo para dividir | 2, 5, 10 | Filas mínimas para crear una rama |
| Variables por división | sqrt, log2 | Cuántas columnas prueba en cada corte |

Para una primera prueba, deja todos en `Probar todos…`.

> **Importante:** estos controles solo se aplican cuando está marcada
> **Buscar la mejor combinación**. Un entrenamiento sin búsqueda usa siempre
> los valores por defecto del modelo, elijas lo que elijas aquí. Para fijar un
> hiperparámetro concreto, activa la búsqueda y deja ese parámetro en el valor
> deseado y los demás en `Probar todos…`.

**Balanceo de clases:** déjalo en `Sin balancear`. Con IR = 1,24 no aporta.

**Selección de atributos:** déjala desmarcada en esta primera prueba.

Pulsa **Entrenar modelo**. Al terminar, la línea de resultado muestra las tres
métricas:

```
Exactitud: 0.9022 · F1 (macro): 0.9005 · AUC (ROC One-vs-Rest): 0.9331
```

La partición de datos usa una semilla fija (`random_state=42`), así que estos
valores son reproducibles con esta versión y estos parámetros. Si cambias el
modelo, los hiperparámetros o el preprocesamiento, cambiarán. Lo importante es
cómo se leen:

| Métrica | Rango | Interpretación |
|---|---|---|
| **Exactitud** | 0 a 1 | Porcentaje de aciertos. 0,90 = 90 % |
| **F1 (macro)** | 0 a 1 | Media del F1 de cada clase, sin favorecer a la mayoritaria |
| **AUC (ROC One-vs-Rest)** | 0,5 a 1 | Capacidad de separar las clases. 0,5 = azar; 1 = perfecto |

#### Probar la búsqueda de hiperparámetros

Abre **Hiperparámetros**, marca **Buscar la mejor combinación** y elige:

- **Estrategia:** `Todas (grid)` prueba todas las combinaciones del dominio;
  `Aleatoria (random)` prueba un número fijo de combinaciones al azar.
- **Folds (validación cruzada):** 5 por defecto.
- **Iteraciones (si es aleatoria):** 20 por defecto.
- **Métrica a optimizar:** el desplegable muestra los nombres internos
  `accuracy`, `f1_macro` y `roc_auc` (en la tabla de resultados aparecen como
  Exactitud, F1 (macro) y AUC).

Pulsa **Entrenar modelo**. Mientras trabaja:

- La barra animada indica que la aplicación sigue viva.
- El texto de estado dice cuántas combinaciones y ajustes faltan y cuánto
  lleva: `Buscando: 54 combinaciones, 270 ajustes · 12/270 ajustes · 8 s`.
- Los controles se bloquean para que no cambies la configuración a medias.

Al acabar verás `Búsqueda terminada en N s. Los mejores parámetros están
arriba.` y las métricas con los mejores parámetros elegidos.

> **Aviso de tiempo:** con Random Forest, «Todas (grid)» y 5 folds son 270
> ajustes. En un portátil normal puede tardar minutos. Para explorar, empieza
> por `Aleatoria (random)` con 10–20 iteraciones.

### Paso 4 · Resultados

Abre la sub-pestaña **Métricas**:

- Arriba, un resumen: modelo, filas de prueba, balanceo y selección.
- La tabla **Métrica / Valor**.
- El **gráfico**:
  - Si entrenaste **sin** búsqueda de hiperparámetros, muestra la **matriz de
    confusión**: filas = clase real, columnas = clase predicha. La diagonal son
    los aciertos; lo que queda fuera de ella son los errores.
  - Si entrenaste **con** búsqueda, muestra la **puntuación de cada
    combinación** probada. La matriz de confusión no se dibuja en ese caso,
    aunque las métricas sí corresponden al mejor modelo encontrado.

> Si el AUC no se puede calcular, la tabla muestra `n/d` y el motivo en el
> tooltip. No es un error: ocurre cuando el modelo no da probabilidades o
> cuando el conjunto de prueba tiene una sola clase.

Abre la sub-pestaña **Exportar**:

- **Guardar proyecto** crea un archivo `.automl` con el modelo, los metadatos y
  (si la casilla **Incluir el dataset transformado** está marcada) los datos ya
  preparados. Es el archivo que usarás para predecir.
- **Exportar dataset** guarda solo los datos transformados en CSV.
- **Ir a Predicción** salta al paso 5.

Pulsa **Guardar proyecto** y guarda, por ejemplo, `heart_random_forest.automl`.

> Un `.automl` contiene código compilado. Guárdalo y ábrelo solo si es tuyo o
> de una fuente de confianza.

### Paso 5 · Predicción

1. Pulsa **Cargar modelo (.automl)…**, elige el archivo que acabas de guardar
   y confirma el aviso de seguridad.
2. Pulsa **Cargar datos de entrada…** y elige `example/data/heart.csv`.

La aplicación compara las columnas del archivo con las que espera el modelo:

- Si coinciden: `Las columnas coinciden con el esquema del modelo.`
- Si sobran columnas: `Columnas ignoradas (no las usa el modelo): …`
- Si falta alguna: `Faltan columnas obligatorias: …` y el texto se pinta en
  rojo.

3. Pulsa **Aplicar el modelo**. Verás una tabla con las columnas nuevas:

   | prediccion | prob_0 | prob_1 |
   |---|---|---|
   | 1 | 0.08 | 0.92 |
   | 0 | 0.81 | 0.19 |

   - `prediccion` es la clase estimada.
   - `prob_0` y `prob_1` son las probabilidades de cada clase.

4. **Exportar resultados** guarda **todas las columnas de entrada más las de
   predicción**. El CSV se escribe en `utf-8-sig` y, salvo que los nombres de
   columna sugieran otro separador, con `;`, para que Excel en español lo abra
   con las tildes y las columnas correctas. También puedes elegir `.xlsx`.
5. **Añadir al dataset** incorpora la predicción a la tabla cargada. Es un
   cambio que no se puede deshacer sin volver a cargar el archivo; la
   aplicación pide confirmación.

> Si el archivo de entrada ya trae una columna llamada `prediccion`, la nueva
> se llamará `prediccion_2` y el aviso lo indicará. Tu columna original no se
> toca.

---

## 4. Tutorial 2: regresión con `ford.csv`

**Objetivo:** predecir el precio de un coche (`price`, en libras).

`ford.csv` tiene **17 965 filas y 9 columnas**. `price` va de 495 a 54 995, y
hay **154 filas duplicadas**.

### Pasos

1. **Datos:** carga `example/data/ford.csv`. La sugerencia automática de
   objetivo propondrá `transmission`; ignórala, porque queremos `price`.
2. **Preprocesamiento:** deja marcado **Eliminar filas duplicadas**. Al
   aplicar verás `Listo: 17 811 filas × 9 columnas`: se han quitado los 154
   duplicados.
   - Normaliza `mileage`, `year`, `engineSize` y `mpg` con
     `estándar (z-score)` si vas a usar modelos lineales (Ridge, Lasso,
     ElasticNet) o SVM. Los árboles y bosques no lo necesitan.
3. **Entrenamiento:**
   - **Variable objetivo:** `price`.
   - **Tipo de tarea:** `Regresión` (se elige solo).
   - **Familia:** `Conjuntos de modelos` → `Random Forest`, o
     `Modelos lineales` → `Ridge`.
4. **Resultados:** las métricas son distintas a las de clasificación:

   | Métrica | Qué significa |
   |---|---|
   | **RMSE** | Error cuadrático medio, en las mismas unidades que `price`. Penaliza mucho los errores grandes |
   | **MAE** | Error absoluto medio, también en libras. Más fácil de interpretar |
   | **R²** | Proporción de la variación explicada. 1 = perfecto; 0 = tan bueno como predecir la media; negativo = peor que la media |

   El gráfico **Real vs. Predicho** (visible si entrenaste sin búsqueda) muestra
   cada coche como un punto. La línea roja discontinua es la predicción
   perfecta: cuanto más cerca de ella, mejor. Si usaste búsqueda, el gráfico
   muestra la puntuación de las combinaciones probadas.

5. **Predicción:** igual que en el tutorial 1. Un archivo de entrada puede
   incluir o no la columna `price`; si la incluye, se conserva, y si no, la
   aplicación solo necesita las variables predictoras.

> La columna `model` tiene 23 valores distintos y es categórica: el pipeline la
> codifica automáticamente (one-hot encoding). No tienes que convertirla a
> números a mano.

---

## 5. Tutorial 3: multiclase, selección y comparativa con `Mobiles.csv`

**Objetivo:** clasificar móviles en cuatro rangos de precio
(`price_range`: 0, 1, 2 o 3).

`Mobiles.csv` tiene **2 000 filas y 21 columnas**, sin nulos ni duplicados. Las
cuatro clases están perfectamente equilibradas: 500 cada una.

### Preparación

1. Carga el archivo y ve al paso 3.
2. Elige `price_range` como objetivo. El tipo de tarea será `Clasificación`.
3. Elige `Conjuntos de modelos` → `Random Forest`.

### Selección de atributos

Abre el acordeón **Selección de atributos** y marca **Seleccionar solo los
atributos más útiles**. Después:

- **Método:** `ANOVA F` para empezar.
- **Criterio:** `Número de atributos (k)`.
- **Valor:** `10`.

Pulsa **Analizar selección**. La tabla muestra, para cada atributo, cuántas
columnas codificadas aporta, su puntuación y si quedaría seleccionado. Este
análisis **no entrena el modelo final**: sirve para ver qué se va a conservar.

Pulsa **Entrenar modelo** con la selección activa. En Resultados aparecerá una
línea como:

```
Selección: ANOVA F (10 atributos conservados)
```

| Método | Cuándo usarlo |
|---|---|
| **Chi-cuadrado** | Clasificación con valores no negativos. Rápido |
| **ANOVA F** | Punto de partida: rápido y con datos limpios |
| **Información mutua** | Sospechas relaciones no lineales. Más lento |
| **Embebido (bosque aleatorio)** | Buscas la mejor exactitud y aceptas el coste. No admite corte por porcentaje |

> La selección ocurre **dentro** del pipeline, después de codificar las
> variables. El conjunto de prueba conserva todas las columnas, así que las
> métricas siguen siendo comparables.

### Comparativa de modelos (Friedman)

Ve al paso 4 y abre la sub-pestaña **Comparativa**. Esta herramienta responde
a una pregunta distinta: *¿de verdad un modelo es mejor que los demás, o la
diferencia es casualidad del reparto de datos?*

1. **Métrica:** `accuracy` (la tabla y el resumen la llaman Exactitud).
2. **Repeticiones:** empieza con `2`. Cada modelo se ajusta una vez por fold y
   por repetición: con 5 folds y 2 repeticiones son 10 ajustes **por modelo**.
   Con 10 modelos compatibles, 100 ajustes.
3. Deja marcado **Todos los modelos compatibles** (o desmárcalo para comparar
   solo la familia actual).
4. Pulsa **Comparar modelos**.

Cómo leer la tabla:

| Columna | Significado |
|---|---|
| **Puntuación media** | La métrica promediada sobre todos los bloques |
| **Desviación** | Cuánto varía entre bloques |
| **Rango medio** | Posición media en cada bloque; **menor es mejor** |
| **Equivalente** | `sí` si su rango cae dentro de la diferencia crítica del mejor |

Y el resumen:

- **Friedman χ² y p**: si `p < 0,05`, las diferencias son significativas. Si
  `p` es alto, no se puede afirmar que un modelo sea mejor que otro.
- **Diferencia crítica**: cuánto puede alejarse un modelo del mejor sin
  considerarse peor (método de Nemenyi).
- **Post-hoc con corrección de Holm**: qué pares concretos difieren. Solo es
  concluyente si Friedman ha salido significativo; si no, la aplicación avisa
  de que esos pares son ruido.
- **Diagrama de diferencias críticas**: barras con el rango medio de cada
  modelo. Los que quedan dentro de la diferencia crítica del mejor se pintan
  en azul; el resto, en rojo.

La comparativa es de **solo lectura**: no cambia el modelo entrenado ni sus
métricas.

### Balanceo (para probar el efecto)

Con `Mobiles.csv` no hace falta, porque las clases están equilibradas. Para
verlo funcionar, usa un dataset desequilibrado o reduce una clase a mano.

En el acordeón **Balanceo de clases**:

| Estrategia | Qué hace | Requiere |
|---|---|---|
| Sin balancear | Conserva la distribución original | — |
| Pesos de clase | El modelo recibe pesos inversos a la frecuencia | Que el modelo admita `class_weight` |
| Submuestreo aleatorio | Reduce las mayoritarias al tamaño de la minoritaria | — |
| Sobremuestreo aleatorio | Repite filas de la minoritaria | — |
| SMOTE | Crea ejemplos sintéticos entre vecinos | `imbalanced-learn` |
| SMOTE-NC | SMOTE para variables mixtas, respeta las categóricas | `imbalanced-learn` |

- Solo aparece en **clasificación**; en regresión no hay balanceo.
- `Pesos de clase` solo se ofrece en los modelos que aceptan ese parámetro.
  Al elegirlo, desaparece el control manual de pesos en los hiperparámetros
  para que no se contradigan.
- SMOTE necesita al menos 2 instancias de la clase minoritaria. Si hay muy
  pocas, reduce el número de vecinos o usa sobremuestreo aleatorio.
- En Resultados verás el efecto real:
  `Balanceo: smote (240 → 480 instancias)`.

> **El dataset exportado no se balancea.** El balanceo vive dentro del modelo:
> el CSV y el `.automl` guardan los datos tal como salen del preprocesamiento,
> sin duplicar ni inventar filas.

---

## 6. Referencia de parámetros

### 6.1 Tipos de columna

La aplicación asigna a cada columna un tipo semántico. Puedes corregirlo desde
la tabla del paso 1.

| Tipo | Qué es | Se usa como predictor |
|---|---|---|
| **Numérico** | Números continuos o discretos | Sí |
| **Booleano** | Verdadero/falso, sí/no, 0/1 | Sí |
| **Categórico** | Pocas categorías repetidas | Sí, se codifica |
| **Fecha** | Fechas reconocibles | Sí, si se expande en año/mes/día |
| **Texto** | Texto libre o con muchas variantes | **No** sin vectorizar |
| **Identificador** | Un valor distinto por fila | **No** |

Si una columna de tipo *Texto* o *Identificador* queda entre las variables
predictoras, el paso 3 avisa y deja el desplegable de modelos vacío:
`requiere convertir o eliminar: <columna>`. La solución es corregir el tipo o
eliminar la columna en el paso 2.

### 6.2 Preprocesamiento

| Opción | Qué hace | Cuándo usarla |
|---|---|---|
| **Eliminar filas duplicadas** | Quita filas idénticas | Casi siempre; los duplicados inflan la clase repetida |
| **Eliminar columnas con >50 % nulos** | Descarta columnas medio vacías | Cuando hay columnas con muchos huecos |
| **Estrategia numérica** | `median`, `mean` o `drop` | `median` si hay valores atípicos; `drop` si prefieres perder filas |
| **Estrategia categórica** | `mode`, `constant` o `drop` | `mode` es lo habitual |
| **Convertir fechas en año, mes y día** | Sustituye la fecha por tres columnas numéricas | Series temporales o fechas con estacionalidad |
| **Normalización por columna** | Ver 6.3 | Modelos lineales, SVM y KNN |
| **Columnas a eliminar** | Quita las columnas marcadas | IDs, texto libre, fugas de información |

> Cuidado con las columnas que contienen la respuesta. Si predices `price` y
> dejas una columna `price_usd`, el modelo parecerá perfecto y no servirá para
> datos nuevos.

### 6.3 Normalización

| Método | Qué hace | Cuándo usarla |
|---|---|---|
| **Sin normalizar** | Deja los valores originales | Árboles, bosques, boosting |
| **Estándar (z-score)** | Media 0, desviación 1 | Modelos lineales y SVM; la opción más habitual |
| **Mín-máx (0-1)** | Escala al rango [0, 1] | Modelos basados en distancias; sensible a atípicos |
| **Robusta (mediana-IQR)** | Usa mediana y rango intercuartílico | Datos con muchos atípicos |
| **Logaritmo** | Comprime colas largas | Variables positivas muy sesgadas (salarios, precios) |

La normalización se aprende **solo con el conjunto de entrenamiento** y se
aplica igual al de prueba, así que no hay fuga de información.

### 6.4 Familias y modelos

| Familia | Modelos | Cuándo elegirla |
|---|---|---|
| **Modelos lineales** | Regresión Logística, SGD, Regresión Lineal, Ridge, Lasso, ElasticNet, Huber, TheilSen | Pocas variables, relaciones aproximadamente lineales, resultados explicables |
| **Árboles de decisión** | Árbol de Decisión | Quieres reglas legibles; suele sobreajustar |
| **Conjuntos de modelos** | Random Forest, ExtraTrees, Bagging, AdaBoost, GradientBoosting, HistGradientBoosting | Buen rendimiento general sin ajuste fino; empieza aquí |
| **Máquinas de soporte vectorial** | SVM, LinearSVC, SVR, KernelRidge | Datasets medianos, fronteras complejas; conviene normalizar |
| **Vecinos más cercanos** | KNN, RadiusNeighbors | Datos densos y homogéneos; conviene normalizar |
| **Naive Bayes** | GaussianNB, BernoulliNB | Variables discretas o poco correlacionadas; muy rápido |
| **Análisis discriminante** | LinearDiscriminant, QuadraticDiscriminant | Clasificación con clases gaussianas |

El catálogo completo, con las decisiones de por qué cada modelo ofrece ciertos
parámetros, está en el [`README.md`](README.md#modelos-disponibles).

### 6.5 Hiperparámetros frecuentes

| Modelo | Parámetro | Qué hace |
|---|---|---|
| Árbol / bosques | `Profundidad máxima` | Limita el tamaño; menor = menos sobreajuste |
| Bosques y boosting | `Número de árboles` / `estimadores` | Más árboles = más estable y más lento |
| Bosques | `Variables por división` | `sqrt` es el valor habitual en clasificación |
| KNN | `Número de vecinos` | Impar para binaria; mayor = más suave |
| KNN | `Peso de vecinos` | `distance` da más peso a los cercanos |
| KNN | `Métrica` | 1 = Manhattan, 2 = Euclídea |
| SVM (kernel) | `C`, `gamma` | `C` alto = menos margen y más sobreajuste |
| Lineales | `C` / `alpha` | Fuerza de la regularización |

Deja el valor `Probar todos…` si no quieres fijar uno: durante la búsqueda se
explorará todo el dominio del parámetro. Sin búsqueda, el modelo usa sus
valores por defecto y estos controles no intervienen.

### 6.6 Búsqueda de hiperparámetros

| Control | Qué significa |
|---|---|
| **Todas (grid)** | Prueba todas las combinaciones del dominio. Exhaustivo y lento |
| **Aleatoria (random)** | Prueba `Iteraciones` combinaciones al azar. Más rápido |
| **Folds** | Particiones de la validación cruzada. 5 es el valor habitual |
| **Iteraciones** | Solo para búsqueda aleatoria |
| **Métrica a optimizar** | La métrica que decide qué combinación gana |

Consejos:

- Ningún grid del catálogo supera las 432 combinaciones. Aun así, 5 folds
  multiplican los ajustes por cinco.
- Si la búsqueda tarda demasiado, baja folds a 3, usa búsqueda aleatoria o
  reduce el número de árboles.
- La búsqueda se ejecuta en un hilo aparte: la ventana sigue respondiendo y los
  controles se bloquean para no cambiar la configuración a medias.

### 6.7 Métricas

| Tarea | Métricas | Cómo interpretarlas |
|---|---|---|
| Clasificación | Exactitud | Aciertos / total. Fácil de leer, engaña con clases desequilibradas |
| Clasificación | F1 (macro) | Equilibrio entre precisión y exhaustividad, promediado por clase |
| Clasificación | AUC (ROC One-vs-Rest) | Capacidad de separar clases. En multiclase se promedia *macro* |
| Regresión | RMSE | Error en unidades del objetivo; castiga los errores grandes |
| Regresión | MAE | Error medio absoluto, en unidades del objetivo |
| Regresión | R² | Variación explicada; 0 = predecir la media |

`n/d` significa «no disponible». El motivo aparece en el tooltip: el modelo no
expone probabilidades o el conjunto de prueba tiene una sola clase.

### 6.8 Comparativa de modelos

| Concepto | Qué significa |
|---|---|
| **Bloques pareados** | Los mismos folds para todos los modelos; condición de validez del test |
| **Rango medio** | Posición media en cada bloque. Menor = mejor |
| **Friedman χ², p** | `p < 0,05` indica diferencias significativas |
| **Diferencia crítica** | Margen de Nemenyi para considerar dos modelos equivalentes |
| **Post-hoc Holm** | Comparaciones por pares con corrección por comparaciones múltiples |

Si un modelo falla en un bloque, se excluye entero con su motivo, en vez de
compararse con menos datos.

### 6.9 Archivos y exportación

| Archivo | Qué contiene | Cuándo usarlo |
|---|---|---|
| **`.automl`** | Modelo + metadatos + (opcional) dataset transformado | Predecir y reproducir el entrenamiento |
| **CSV del dataset** | Solo los datos transformados | Revisar o reutilizar los datos |
| **CSV/XLSX de predicciones** | Datos de entrada + `prediccion` + `prob_<clase>` | Entregar resultados |

- El `.automl` se guarda con `joblib` y **contiene código compilado**. Ábrelo
  solo si confías en su origen.
- La exportación de predicciones escribe CSV en `utf-8-sig` con `;` para Excel
  en español, o XLSX si eliges esa extensión.
- La exportación del dataset desde Resultados es un CSV estándar.

---

## 7. Problemas frecuentes

### «Ningún modelo puede usarse con estos datos»

El desplegable de modelos está vacío y la línea de compatibilidad explica el
motivo. Las dos causas habituales son:

- **`requiere convertir o eliminar: <columna>`**: hay una columna de texto
  libre o un identificador entre las variables predictoras. Corrige su tipo en
  el paso 1 o elimínala en el paso 2.
- **`necesita al menos N filas (hay M)`**: el dataset es demasiado pequeño
  para esa familia. SVM pide 50 filas, vecinos 30, Naive Bayes 10 y el
  análisis discriminante 20. Prueba otra familia.

### «Faltan columnas obligatorias»

El archivo de predicción no tiene todas las columnas con las que se entrenó.
Puedes cancelar y corregir el archivo, o aceptar el aviso: la aplicación
rellena las columnas ausentes con valores vacíos, que el modelo ya sabe
imputar. El resultado será menos fiable.

### «Columnas ignoradas (no las usa el modelo)»

El archivo trae columnas de más. No es un error: se conservan en la exportación
pero no entran en la predicción.

### SMOTE falla o no aparece

- SMOTE y SMOTE-NC necesitan `imbalanced-learn`. Instálalo con
  `pip install -e ".[smote]"` o `pip install imbalanced-learn`.
- Con muy pocas instancias de la clase minoritaria, baja el número de vecinos
  o usa **Sobremuestreo aleatorio**.

### El CSV se abre mal en Excel

- La aplicación detecta el separador y la codificación al leer, así que no
  tienes que convertir nada antes de cargar.
- Al **exportar predicciones**, el CSV usa `utf-8-sig` y normalmente `;`, a
  propósito.
- La exportación del dataset desde Resultados no añade esas marcas. Si Excel
  junta las columnas, impórtalo desde *Datos → Desde texto/CSV* indicando `;`.

### No puedo exportar a Excel

La exportación a `.xlsx` necesita `openpyxl`. Si no está instalado, la
aplicación lo dice y ofrece guardar en CSV. Instálalo con
`pip install openpyxl`.

### Un botón está apagado y no sé por qué

Pasa el ratón por encima: el tooltip dice qué falta. Las reglas son:

| Estado | Qué se apaga |
|---|---|
| Sin dataset | Preprocesamiento completo, menos la carga |
| Sin variables numéricas | Estrategia de imputación numérica |
| Sin dataset limpio, objetivo o modelo | «Entrenar modelo» |
| Sin modelo entrenado | Guardar, exportar e ir a Predicción |
| Sin modelo guardado | «Aplicar el modelo» |
| Modelo cargado, pero sin datos de entrada | «Aplicar el modelo» |
| Sin predicciones | «Exportar resultados» |

### La aplicación tarda mucho entrenando

- Baja los folds de 5 a 3.
- Usa búsqueda aleatoria con 10–20 iteraciones en vez de grid.
- Reduce el número de árboles.
- La comparativa de Friedman multiplica el coste por modelos × folds ×
  repeticiones: empieza con 2 repeticiones.

### El modelo acierta demasiado

Sospecha de una **fuga de información**: una columna que contiene la respuesta
o se calcula después de conocerla. Elimínala en el paso 2 y vuelve a entrenar.

---

## 8. Glosario

| Término | Significado |
|---|---|
| **Clasificación** | Predecir una categoría (sí/no, tipo A/B/C) |
| **Regresión** | Predecir un número continuo (precio, temperatura) |
| **Variable objetivo** | La columna que quieres predecir |
| **Predictor / atributo** | Cada columna que usa el modelo para predecir |
| **Pipeline** | Cadena de pasos: imputación, codificación, normalización y modelo |
| **Hiperparámetro** | Ajuste del modelo que se elige antes de entrenar |
| **Validación cruzada** | Dividir los datos en folds para evaluar sin depender de un único reparto |
| **Fold** | Cada una de las particiones de la validación cruzada |
| **Sobreajuste** | El modelo memoriza el entrenamiento y falla con datos nuevos |
| **Fuga de información** | Usar al entrenar datos que no estarán disponibles al predecir |
| **Balanceo** | Compensar clases desequilibradas |
| **IR** | Índice de desbalance: clase mayoritaria / minoritaria |
| **One-hot encoding** | Convertir cada categoría en una columna 0/1 |
| **AUC** | Área bajo la curva ROC; mide la separación entre clases |
| **RMSE / MAE** | Medidas de error de regresión, en unidades del objetivo |
| **`.automl`** | Archivo de proyecto con el modelo, sus metadatos y, opcionalmente, los datos transformados |

---

## 9. Atajos y flujo recomendado

1. Carga los datos y revisa el perfil.
2. Corrige tipos y elige el objetivo.
3. Limpia y aplica el preprocesamiento.
4. Entrena un modelo sencillo, sin búsqueda.
5. Mira las métricas y el gráfico.
6. Si hace falta, prueba búsqueda de hiperparámetros, balanceo o selección.
7. Compara modelos antes de dar uno por bueno.
8. Guarda el `.automl`.
9. Aplica el modelo a datos nuevos y exporta los resultados.

| Atajo | Acción |
|---|---|
| `Ctrl+1` | Datos |
| `Ctrl+2` | Preprocesamiento |
| `Ctrl+3` | Entrenamiento |
| `Ctrl+4` | Resultados |
| `Ctrl+5` | Predicción |
| `Ctrl+Tab` | Paso siguiente |

---

## 10. Para terminar

La aplicación no decide por ti: te muestra qué ha entendido de cada columna,
qué modelo vas a entrenar y qué métricas obtienes. Revisa el perfil antes de
entrenar y desconfía de una exactitud perfecta: casi siempre significa que el
objetivo se ha colado entre los predictores.

Si quieres practicar, empieza por `heart.csv` (el flujo completo), sigue con
`ford.csv` (un objetivo continuo) y termina con `Mobiles.csv` (selección,
comparativa y balanceo). Los tres están en `example/data/`.
