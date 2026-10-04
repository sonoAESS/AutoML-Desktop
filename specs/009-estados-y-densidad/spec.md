# SDD-009 — Estados de los controles y densidad de opciones

| Campo | Valor |
|---|---|
| Estado | Especificación — escrita |
| Orden de ejecución | 3 de 3 (de la rama `design/redesign-ui`) |
| Ámbito | las cinco pestañas de `ui/`, `ui/main_window.py`, `tests/test_estados_ui.py` |
| Depende de | SDD-007 (tokens), SDD-008 (estructura) |
| Bloquea a | — |

## 1. Problema

La interfaz comunica estados que no existen. En Preprocesamiento hay **13
controles interactivos** (casillas, combos, spins, casillas de operación)
que están habilitados **siempre**, incluso sin dataset cargado. En Predicción,
«Predecir» y «Exportar» se pueden pulsar sin modelo. En Resultados, los tres
botones de guardado se pueden pulsar sin modelo entrenado.

El usuario ve casillas activas para un dataset que aún no ha cargado, y
descubre que no funciona al pulsar. El coste es doble: ruido visual y
descubrimiento tardío de los errores.

El recuento real de `setEnabled` por fichero lo confirma:

| Fichero | Llamadas a `setEnabled` |
|---|---|
| `ui/preprocess_tab.py` | **0** |
| `ui/data_tab.py` | 2 |
| `ui/train_tab.py` | 15 |
| `ui/results_tab.py` | 8 |
| `ui/predict_tab.py` | 7 |

Preprocesamiento es el caso extremo: **ningún** control cambia de estado.

Hay un segundo problema, de espacio y de lenguaje: los botones tienen textos
largos («Añadir predicciones al dataset», 36 caracteres) que ocupan la fila
entera y obligan a comprimir los demás controles, y muchos controles no
explican qué hacen porque no tienen `toolTip`.

## 2. Objetivo

Que **cada control esté habilitado solo cuando tenga sentido**, con una única
regla derivada de `AppState` por pestaña; y que cada opción ocupe el mínimo
espacio posible, con el detalle largo en el tooltip en vez de en el botón.

## 3. Alcance

### 3.1 Incluido

- `_update_enabled_state()` en cada pestaña, con la condición de habilitación
  de cada control explícita y legible.
- Textos de botón cortos y detalle en tooltips.
- Tooltips en los controles que hoy no los tienen.
- Mensajes de estado en la barra de la ventana que dicen en qué paso está la
  aplicación.

### 3.2 No incluido

- Ocultar controles en vez de deshabilitarlos (sería otra decisión de flujo).
- Validación de entrada con mensajes propios: la validación sigue siendo de
  `core/` y se muestra como error.
- Reordenar de nuevo las pestañas: es SDD-008.

## 4. Requisitos

### Estados

- **R-001.** Cada pestaña tiene un método `_update_enabled_state()` que **solo
  lee `self.state`** y fija el estado de sus controles. Es idempotente.
- **R-002.** Se invoca desde `refresh()` y tras cualquier acción que cambie
  `AppState` (aplicar preprocesamiento, entrenar, cargar, guardar, predecir).
- **R-003.** **Datos**: sin `raw_df`, «Usar objetivo sugerido» y «Volver a
  detectar tipos» deshabilitados; con dataset, habilitados.
- **R-004.** **Preprocesamiento**: sin `raw_df`, todos los controles
  deshabilitados salvo el de carga. Con dataset, habilitados los que aplican al
  perfil semántico: si no hay columnas numéricas, la estrategia numérica se
  deshabilita; si no hay categóricas, la categórica.
- **R-005.** **Preprocesamiento**: las opciones que dependen de otra se
  habilitan en cascada. Si «Eliminar duplicados» está desmarcado, su detalle no
  se muestra. La jerarquía se respeta: si un padre está deshabilitado, sus
  hijos también.
- **R-006.** **Entrenamiento**: «Entrenar modelo» exige dataset limpio, objetivo
  y modelo. El grupo de búsqueda se habilita solo con la casilla marcada (ya
  existe `_toggle_tune_options`; se unifica con el mismo criterio).
- **R-007.** **Resultados**: sin modelo entrenado, deshabilitados «Guardar
  modelo y dataset», «Exportar solo el dataset» e «Ir a Predicción»; también
  el informe y el canvas quedan con su mensaje de estado vacío.
- **R-008.** **Resultados**: el grupo de comparativa se habilita con dataset
  transformado **y** ≥3 modelos compatibles (ya implementado; se conserva).
- **R-009.** **Predicción**: sin bundle cargado, deshabilitados «Predecir» y
  «Exportar»; sin datos de entrada, «Predecir» sigue deshabilitado aunque haya
  bundle.
- **R-010.** Un control deshabilitado **explica por qué** en su tooltip: no
  basta con silenciarlo.

### Densidad y textos

- **R-011.** El texto visible de un botón no supera los **28 caracteres**. Si
  hace falta más, el detalle va al `toolTip`.
- **R-012.** Botones largos que se converting en verbos claros: «Añadir
  predicciones al dataset» → «Añadir al dataset», con el detalle completo en
  el tooltip.
- **R-013.** Todo control interactivo tiene `toolTip` en español que explica
  **qué hace y cuándo conviene usarlo**, no qué es. Sin tooltip solo se
  aceptan los controles autoexplicativos (una etiqueta junto al control).
- **R-014.** Las filas de controles se reparten con `QFormLayout` cuando la
  pareja es etiqueta + control, y las acciones van en su propia fila al final.
  Ninguna fila(superficial) supera los **5 widgets**.
- **R-015.** Los botones de acción destructiva (limpiar, reiniciar, eliminar)
  llevan la clase `peligro` de SDD-007 y confirmación si hay algo que perder.

### Barra de estado

- **R-016.** La barra de estado dice en qué paso está la aplicación y qué acaba
  de pasar, sin repetir lo que ya se ve en la etiqueta de la pestaña.
- **R-017.** Los mensajes son distintos según el resultado: operación con
  éxito (texto `suave`), operación fallida (rojo `acento`).

## 5. Invariantes

| Invariante | Origen |
|---|---|
| `core/` no se toca: los estados se derivan de `AppState` | AGENTS §7.4 |
| Ninguna condición de habilitación inventa datos: solo lee `AppState` | AGENTS §5.1 |
| Los errores se siguen mostrando con `QMessageBox`, no deshabilitando en silencio | AGENTS §5.2 |
| Sin dependencias nuevas | AGENTS §7.3 |
| Mensajes en español | AGENTS §4 |
| Los 314 tests existentes siguen en verde | AGENTS §8 |

## 6. Criterios de aceptación

- **C-001.** Sin dataset, Preprocesamiento tiene todos sus controles
  deshabilitados menos el de carga (recuento exacto verificado por test).
- **C-002.** Con dataset categórico, la estrategia numérica queda
  deshabilitada y la categórica habilitada.
- **C-003.** `_update_enabled_state()` es idempotente: llamarlo dos veces no
  cambia ningún estado.
- **C-004.** Desactivar un grupo desactiva a todos sus hijos (jerarquía).
- **C-005.** Sin modelo, los tres botones de Resultados están deshabilitados y
  su tooltip dice qué falta.
- **C-006.** Sin bundle, «Predecir» está deshabilitado; con bundle y sin datos,
  también; con bundle y datos, habilitado.
- **C-007.** Todos los textos de botón tienen ≤28 caracteres (test que recorre
  los botones construidos).
- **C-008.** Todo control interactivo sin etiqueta tiene `toolTip` no vacío.
- **C-009.** Ninguna fila horizontal de nivel superior tiene más de 5 widgets.
- **C-010.** La barra de estado cambia de texto en cada transición de estado
  y distingue éxito de fallo.
- **C-011.** Los 314 tests existentes siguen en verde sin modificarlos.
- **C-012.** La suite completa pasa.

## 7. Riesgos

| Riesgo | Mitigación |
|---|---|
| Deshabilitar de más y bloquear un flujo válido | Cada condición se prueba con el `AppState` mínimo que la habilita; C-003 y C-006 cubren los casos límite |
| Ocultar información: el usuario no sabe que la opción existe | R-010 obliga a que el tooltip diga por qué está deshabilitado |
| `_update_enabled_state()` invocado en el sitio equivocado deja estados rancios | R-002 obliga a unificar los puntos de invocación; los tests los recorren todos |
| Perder funcionalidad por acortar textos | R-011 mueve el detalle al tooltip, nunca lo descarta |