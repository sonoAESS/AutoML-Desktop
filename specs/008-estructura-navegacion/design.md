# SDD-008 — Diseño

## 1. Arquitectura

```
MainWindow (QMainWindow)
├── cabecera: QFrame + QHBoxLayout   → escudo + título + info de estado
├── cuerpo:   QHBoxLayout
│   ├── StepSidebar (QFrame, ~210 px)  ← ui/sidebar.py
│   └── QStackedWidget (contenido)     ← las 5 pestañas, cada una en un QScrollArea
└── QStatusBar
```

```
StepSidebar
└── QVBoxLayout
    ├── marca:      QLabel(píxel del escudo) + QLabel("AutoML Desktop")
    ├── separador
    ├── 5 × StepButton (QPushButton checkable, propiedad "estado")
    └── separador
    └── QLabel(marca del Ministerio, texto small)
```

### Por qué `QStackedWidget` y no `QTabWidget`

Un `QTabWidget` de nivel superior obliga a consumir 30–40 px de altura para las
pestañas, que es exactamente lo que falta. Con `QStackedWidget` + barra lateral
el contenido gana ese alto. Además el `QTabWidget` no admite una columna
lateral con estado; la barra lateral sí, porque cada botón lleva su propia
propiedad de estado.

## 2. `StepSidebar`

```python
class StepButton(QPushButton):
    def __init__(self, indice, titulo):
        super().__init__(f"{indice}.  {titulo}")
        self.setCheckable(True)
        self.setProperty("estado", "pendiente")
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setCursor(Qt.PointingHandCursor)
```

### Estados y su doble señal (R-003)

| Estado | Fondo | Texto | Señal no cromática |
|---|---|---|---|
| `pendiente` | transparente | `suave` | Número en gris claro |
| `activo` | `azul_med` | blanco | **Barra izquierda de 3 px** en `acento` |
| `completado` | transparente | `texto` | Número con fondo `oro` |

La barra izquierda se hace con `border-left` en QSS. Es la señal no
colórica que exige R-003: en una pantalla gris se distingue el paso activo
también por esa barra.

`set_estado(indice, estado)` valida el estado contra las tres opciones
conocidas y lanza `ValueError` si recibe otro, para que un typo no deje un
paso en un estado invisible.

## 3. Señal de navegación (R-008)

El problema real de `ResultsTab._go_to_predict`:

```python
# HOY — frágil
window = self.window()
tabs = window.centralWidget()
indice = tabs.indexOf(window.predict_tab)
```

Con barra lateral `centralWidget()` es un `QWidget`, no un `QTabWidget`: el
`hasattr(tabs, "setCurrentIndex")` de esa función **seguiría siendo cierto**
porque un `QStackedWidget` también lo tiene, pero `window.predict_tab` solo
existe mientras el atributo se llame así. El fallo sería silencioso.

Solución, en dos piezas:

```python
# ResultsTab
navigate_requested = Signal(str)     # "prediccion" | "entrenamiento" | ...

# MainWindow
self.results_tab.navigate_requested.connect(self._ir_a)
def _ir_a(self, paso):
    self.ir_a_paso(INDICE_POR_NOMBRE.get(paso))
```

`ir_a_paso(indice)` es el **único** punto que cambia de vista, y es público
para que los tests puedan verificarlo sin pulsar.

Se aplica el mismo patrón a cualquier navegación entre pestañas.

## 4. Scroll (R-011)

Función única reutilizable, porque las cinco pestañas la necesitan:

```python
def envolver_scroll(widget) -> QScrollArea:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setFrameShape(QFrame.NoFrame)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    area.setWidget(widget)
    return area
```

Se envuelve el **contenido**, no la pestaña: `QStackedWidget.addWidget(area)`.

Con `setWidgetResizable(True)` y políticas de tamaño expansivas, las barras
aparecen solo cuando hacen falta y desaparecen al ampliar la ventana.

## 5. Resultados en tres sub-pestañas (R-012)

| Sub-pestaña | Contenido |
|---|---|
| **Métricas** | `lbl` de estado, `tbl_metricas`, `canvas` con ROC/matriz/importancia, `txt_report` |
| **Comparativa** | `gb_comparativa` completa: opciones, botón, progreso, `tbl_comparativa`, etiquetas, `canvas_cd` |
| **Exportar** | `chk_dataset`, `btn_save`, `btn_export`, `btn_load_in_predict`, `lbl_saved` |

Motivo de agrupar así: «Comparativa» es una tarea opcional y larga (Friedman
tarda); dejarla junto a las métricas obligaba a hacer scroll para verla.
«Exportar» son tres botones, así que por sí solo no merece una sub-pestaña,
pero sí un sitio propio donde el usuario termina.

La sub-pestaña de comparativa se crea con el constructor (`show()`), pero
**oculta** (`setCurrentIndex(0)`), igual que hoy: la disponibilidad la decide
`_refresh_comparativa_disponibilidad`.

Los atributos no cambian de nombre: `gb_comparativa`, `tbl_comparativa`,
`canvas_cd`, `figure_cd`, `btn_comparar`, `cmb_metrica_comparativa`,
`spn_repeticiones`, `chk_todos_los_modelos`, `progress_comparativa`,
`lbl_comparativa`, `lbl_excluidos` (R-017).

## 6. Acordeón en Preprocesamiento y Entrenamiento (R-013)

`QToolBox` con una página por decisión del usuario.

Preprocesamiento:

| Página | Contenido |
|---|---|
| Calidad | `gb_dups`, `gb_null` |
| Tipos y fechas | `gb_tipos` |
| Normalización | `gb_norm` |
| Columnas a eliminar | `gb_cols` |
| Distribución de clases | `gb_distribucion` (siempre visible: informa, no configura) |

El botón «Aplicar preprocesamiento» y su etiqueta quedan **fuera** del
acordeón, siempre visibles: es la acción que cierra el paso.

Entrenamiento:

| Página | Contenido |
|---|---|
| Modelo | `form` objetivo/tarea/familia/modelo, etiquetas de compatibilidad y de modelo |
| Ajustes | `gb_params`, `gb_tune` |
| Balanceo | `gb_balanceo` |
| Selección de atributos | `gb_seleccion` |

«Entrenar modelo», `progress` y los mensajes quedan fuera del acordeón.

`QToolBox` empieza con la primera página abierta. `gb_distribucion` se manages
mediante `setVisible` como hoy, y se añade como página solo si aplica; si no
aplica se deja la página fuera para no mostrar un hueco vacío.

**Riesgo detectado:** mover widgets dentro de un `QToolBox` requiere quitar su
pariente (un `QGroupBox` dentro de un `QVBoxLayout` no puede reparentarse
limpiamente). Se resuelve construyendo la página con `addItem(title, widget)`
donde `widget` es el `QGroupBox` ya creado pero **sin** haberlo añadido al
layout raíz.

## 7. Alturas y gráficos (R-014, R-015)

- `ALTURA_TABLA = 180` px por defecto, con `setMaximumHeight` explícito en cada
  tabla y lista nueva o existente.
- En cada `_pintar*` / `_plot*`: `self.figure.tight_layout()` antes de
  `canvas.draw_idle()`. Sin esto, las etiquetas largas de nombres de atributos
  se cortan (es el defecto visible hoy en la importancia de variables).
- `canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)` en los
  5 canvas.

## 8. Tamaño de ventana (R-016)

`resize(1240, 720)` y `setMinimumSize(1000, 640)`.

Justificación: 1366×768 es la resolución de portátil más común en el entorno
de uso; descontando la barra de tareas y un panel lateral quedan ~1360×700.
Con 720 de alto la ventana entra justa y el contenido se desplaza dentro con
el `QScrollArea`, que es preferible a que la propia ventana se recorte.

Se elimina el `resize(1180, 820)` actual porque 820 de alto **no** entra.

## 9. Atajos (R-010)

`QShortcut` con contexto `WindowShortcut`:

| Atajo | Efecto |
|---|---|
| `Ctrl+1` … `Ctrl+5` | Ir al paso N |
| `Ctrl+Tab` | Avanzar al paso siguiente (envuelve al último) |

## 10. Contrato de atributos (R-017)

Se extrae la lista de los atributos que los tests tocan y se congela en un
test de `tests/test_navigation_ui.py`:

```
refresh, cmb_metodo_sel, cmb_criterio_sel, cmb_balancing, spn_vecinos,
spn_sel, chk_seleccion, tbl_seleccion, btn_analizar, lbl_sel, gb_balanceo,
gb_distribucion, tbl_distribucion, lbl_distribucion, cmb_task, cmb_model,
tbl_comparativa, lbl_comparativa, figure_cd, btn_comparar, spn_repeticiones,
chk_todos_los_modelos, progress_comparativa, cmb_metrica_comparativa,
gb_comparativa, _set_busy_comparativa, _modelos_a_comparar,
_avisar_comparativa_desfasada, _show_comparison, _param_widgets,
_balancing_config, _selection_config, _on_task_changed, load_data,
```

El test comprueba que **todos** existen en la pestaña correspondiente tras el
rediseño. Es la red de seguridad de R-017.

## 11. Alternativas descartadas

| Alternativa | Por qué no |
|---|---|
| Mantener `QTabWidget` y añadir scroll | No resuelve el estado de los pasos, que es el problema de navegación |
| Wizard de un paso por pantalla | Esconde las opciones avanzadas, que son el valor de la herramienta |
| `QToolBox` en Datos y Predicción | Con 2–3 grupos no aporta nada y añade un clic |
| `QStackedLayout` en vez de `QStackedWidget` | `QStackedWidget` da índice y señal `currentChanged`, que necesita la navegación |