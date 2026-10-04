# SDD-008 — Estructura, navegación y densidad de las pestañas

| Campo | Valor |
|---|---|
| Estado | Especificación — escrita |
| Orden de ejecución | 2 de 3 (de la rama `design/redesign-ui`) |
| Ámbito | `ui/main_window.py`, `ui/navigation.py` (nuevo), `ui/sidebar.py` (nuevo), las cinco pestañas, `tests/test_navigation_ui.py` |
| Depende de | SDD-007 (tokens y estilos) |
| Bloquea a | SDD-009 |

## 1. Problema

La navegación es una `QTabWidget` plana en la que los cinco pasos se ven
igual: no se distingue en cuál está el usuario ni qué pasos ya están
hechos. Para un usuario sin experiencia, ése es el punto de fricción
principal.

El problema de espacio es concreto y medido, no una impresión:

| Pestaña | Altura mínima estimada | Why |
|---|---|---|
| Resultados | ~1100 px | tabla de métricas (140) + canvas 5×4 in (~330) + grupo de comparativa (tabla 160 + canvas 180) + botones y etiquetas |
| Entrenamiento | ~950 px | formulario (4 filas) + 2 etiquetas + rejilla de hiperparámetros + 3 grupos con formularios + botón Entrenar |
| Preprocesamiento | ~800 px | 6 grupos apilados, dos con tablas sin altura acotada |

En un portátil de 1366×768, útil ≃ 700 px de alto: **no hay scroll en
ninguna pestaña**, así que el botón «Entrenar modelo» y el diagrama de
comparativa quedan literalmente inalcanzables. Las tablas
(`table_norm`, `tbl_distribucion`, `list_cols`, `tbl_seleccion`,
`tbl_comparativa`) no tienen `setMaximumHeight` y empujan el resto fuera
del área visible.

A esto se suma un acoplamiento frágil: `ResultsTab._go_to_predict` sube por
`self.window().centralWidget()` y busca el índice de la pestaña. Cualquier
cambio en el orden de las pestañas lo rompe **en silencio** y salta a la
pantalla equivocada.

## 2. Objetivo

Una navegación lateral con los cinco pasos, que muestre en cuál está el
usuario y qué pasos ya completó; un panel de contenido que siempre entre en
la pantalla; y la decoupling de la navegación por señales, para que
reordenar pasos no rompa nada.

## 3. Alcance

### 3.1 Incluido

- Barra lateral de pasos con estado (pendiente / activo / completado).
- `QStackedWidget` como panel de contenido.
- Sub-pestañas internas en Resultados y acordeón plegable en
  Preprocesamiento y Entrenamiento.
- `QScrollArea` en las cinco pestañas.
- Alturas acotadas para tablas y listas.
- Señal de navegación en lugar del acceso a `self.window()`.

### 3.2 No incluido

- Mover controles entre pestañas: esta spec reorganiza, no cambia el flujo.
- Asistente guiado (wizard) de un paso por pantalla: se descartó por
  ocultar las opciones avanzadas.
- Habilitación/deshabilitación de controles por estado: es SDD-009.
- Cualquier cambio en `core/`.

## 4. Requisitos

### Barra lateral

- **R-001.** `ui/sidebar.py` define `StepSidebar(QFrame)`, una columna fija
  de ancho constante (~210 px) con la marca arriba (escudo + título) y los
  cinco pasos debajo.
- **R-002.** Cada paso es un `QPushButton` checkable con: número, título y
  una propiedad dinámica `estado` ∈ {`pendiente`, `activo`, `completado`}.
  El color lo decide la hoja de estilo de SDD-007 según el estado.
- **R-003.** El paso activo se distingue por color de fondo `azul_med` y texto
  blanco; los completados, con el número relleno en `oro`; los pendientes, con
  texto `suave`. Ningún estado se distingue **solo** por color (R-012 exige
  doble señal).
- **R-004.** `set_estado(indice, estado)` actualiza la propiedad y repinta con
  `unpolish/polish`, igual que `marcar()` de SDD-007.
- **R-005.** Emitir `paso_elicido = Signal(int)` al pulsar. `MainWindow` es el
  único que decide a qué paso se navega.
- **R-006.** Un paso completado no se puede deseleccionar desde la barra; el
  contenido sigue siendo de solo lectura para el usuario.

### Panel de contenido

- **R-007.** `MainWindow` usa un `QStackedWidget` con las cinco pestañas en el
  orden del flujo: Datos → Preprocesamiento → Entrenamiento → Resultados →
  Predicción.
- **R-008.** La navegación entre pasos se hace **solo** con
  `AppState` + señales. Se elimina `ResultsTab._go_to_predict` y se sustituye
  por `navigate_requested = Signal(str)`, que `MainWindow` escucha.
  **Prohibido** acceder a `self.window()`, `centralWidget()` o al índice de
  otra pestaña desde dentro de una pestaña (AGENTS §5.1).
- **R-009.** Los botones que yaTTPnavegaban siguen funcionando: «Ir a
  Predicción» en Resultados y el equivalente de los pasos anteriores.
- **R-010.** Atajos de teclado `Ctrl+1` … `Ctrl+5` para saltar de paso, y
  `Ctrl+Tab` para avanzar al siguiente paso.

### Ajuste al espacio

- **R-011.** Las cinco pestañas se envuelven en un `QScrollArea` con
  `setWidgetResizable(True)` y `setFrameShape(QFrame.NoFrame)`, de modo que el
  contenido nunca queda recortado.
- **R-012.** Resultados se divide en un `QTabWidget` interno de tres
  sub-pestañas: **Métricas** (etiqueta de estado, tabla, canvas, informe),
  **Comparativa** (todo el grupo de Friedman) y **Exportar** (opciones,
  botones y etiqueta de guardado). Cada sub-pestaña entra en la pantalla.
- **R-013.** Preprocesamiento y Entrenamiento usan un `QToolBox` plegable
  (acordeón) con sus grupos, de modo que solo uno esté abierto a la vez y no
  haya scroll horizontal.
- **R-014.** Toda tabla y lista nueva o existente lleva `setMaximumHeight`
  coherente (por defecto ~180 px) para que no empuje los botones hacia abajo.
- **R-015.** Los canvas de matplotlib llevan
  `SizePolicy(Expanding, Expanding)` y `tight_layout()` en cada dibujo, para
  que no se recorten las etiquetas largas (nombres de atributos, clases).
- **R-016.** El tamaño de ventana es `1240×720` con mínimo `1000×640`, para
  entrar en 1366×768 sin barra de desplazamiento de ventana.

### Contrato con los tests

- **R-017.** No se renombra ningún atributo de widget existente. Los tests de
  UI hacen de unos 40 atributos (`tbl_comparativa`, `gb_comparativa`,
  `cmb_balancing`, `spn_sel`…); si un control se mueve de contenedor, conserva
  su nombre.
- **R-018.** Las señales públicas de cada pestaña (`data_loaded`,
  `data_processed`, `model_trained`, `comparison_ready`, `model_saved`)
  mantienen su firma.

## 5. Invariantes

| Invariante | Origen |
|---|---|
| `core/` no se toca | AGENTS §7.4 |
| Comunicación entre pestañas por `AppState` + señales, nunca por acceso directo | AGENTS §5.1 |
| Nada de lógica científica en `ui/` | AGENTS §3.5 |
| Trabajo pesado sigue en `ui/workers.py` | AGENTS §5.3 |
| Sin dependencias nuevas | AGENTS §7.3 |
| Los 314 tests existentes siguen en verde | AGENTS §8 |

## 6. Criterios de aceptación

- **C-001.** `MainWindow` no contiene ningún `QTabWidget` de nivel superior;
  el contenido se cambia con `setCurrentIndex` sobre un `QStackedWidget`.
- **C-002.** Ninguna pestaña accede a `self.window()`: un test greps `ui/*.py`
  y falla si aparece fuera de `ui/main_window.py`.
- **C-003.** Los cinco pasos aparecen en la barra lateral en el orden del flujo
  y con su número.
- **C-004.** `set_estado(2, "completado")` deja el paso 2 con esa propiedad y
  con doble señal visual (color **y** texto/símbolo).
- **C-005.** Pulsar un paso cambia el contenido del panel y no pierde el estado
  de los datos (`AppState` intacto).
- **C-006.** `Ctrl+3` lleva al paso 3.
- **C-007.** Con la ventana a 1280×720, el botón «Entrenar modelo» y
  «Aplicar preprocesamiento» quedan dentro del área visible (comprobado con
  `geometry()`), sin necesidad de hacer scroll.
- **C-008.** Resultados tiene tres sub-pestañas y solo una está activa.
- **C-009.** Preprocesamiento y Entrenamiento son un `QToolBox` y, tras
  construir la ventana, hay **como máximo una** página visible.
- **C-010.** Toda tabla y lista de las cinco pestañas tiene
  `setMaximumHeight` mayor que cero.
- **C-011.** Toda figura se dibuja con `tight_layout()` y el canvas es
  `Expanding`.
- **C-012.** Los botones «Ir a Predicción» siguen cambiando de paso a través de
  la señal, sin tocar `centralWidget()`.
- **C-013.** Ningún atributo de widget de los usados por los tests ha
  desaparecido (test que recorre la lista de nombres conocidos).
- **C-014.** La suite completa pasa.

## 7. Riesgos

| Riesgo | Mitigación |
|---|---|
| Reorganizar rompe los 40 atributos que usan los tests | R-017 lo prohíbe explícitamente y C-013 lo vigila |
| El acordeón esconde opciones que antes se veían | Solo se pliega lo que es Advanced; «Distribución de clases» y las acciones principales quedan visibles |
| `QToolBox` con muchos grupos se siente pesado | Cada página se corresponde con una decisión del usuario, no con un control suelto |
| Señal de navegación mal conectada deja botones muertos | C-012 prueba el recorrido real de señal a contenido |