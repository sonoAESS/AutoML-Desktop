# SDD-008 — Tareas

Estado: **esperando puerta 2**. Depende de SDD-007.

## Fase A — Infraestructura compartida

- [ ] **T-001.** `ui/layout.py` (nuevo) con `envolver_scroll(widget)` y `ALTURA_TABLA = 180`. Docstring: envuelve el contenido de una pestaña en un `QScrollArea` sin marco.
- [ ] **T-002.** `ui/sidebar.py` (nuevo): `StepButton` y `StepSidebar` con `paso_elicido = Signal(int)`, `set_estado(indice, estado)` que valida los tres estados y `marca_institucional` para el escudo.
- [ ] **T-003.** `INDICE_POR_NOMBRE` y `PASOS` (nombre, título) como fuente única de los cinco pasos.

## Fase B — MainWindow

- [ ] **T-004.** Sustituir el `QTabWidget` raíz por `StepSidebar` + `QStackedWidget`. Sin pestañas superiores.
- [ ] **T-005.** `ir_a_paso(indice)` como único punto que cambia de vista; conecta con `paso_elicido`.
- [ ] **T-006.** Conectar `navigate_requested` de `ResultsTab` a `ir_a_paso` y **borrar** `_go_to_predict`.
- [ ] **T-007.** Estado de los pasos: `Datos` completado al cargar CSV, `Preprocesamiento` al aplicar, `Entrenamiento` al entrenar, `Resultados` al tener modelo.
- [ ] **T-008.** `resize(1240, 720)`, `setMinimumSize(1000, 640)`, cabecera con escudo y título.
- [ ] **T-009.** Atajos `Ctrl+1`–`Ctrl+5` y `Ctrl+Tab` con `QShortcut`.
- [ ] **T-010.** Cada pestaña se envuelve con `envolver_scroll` antes de añadirla al `QStackedWidget`.

## Fase C — Resultados

- [ ] **T-011.** `QTabWidget` interno con tres páginas: Métricas, Comparativa, Exportar. Redistribuir los widgets existentes **conservando sus nombres de atributo**.
- [ ] **T-012.** `_refresh_comparativa_disponibilidad` selecciona la página de comparativa en vez de hacer `setVisible` del grupo, y oculta la página cuando no aplica.
- [ ] **T-013.** `ALTURA_TABLA` en `tbl_metricas` y `tbl_comparativa`; `SizePolicy` expandente en los dos canvas.
- [ ] **T-014.** `tight_layout()` en cada `_pintar*` / `_plot*` de la pestaña.
- [ ] **T-015.** Botón «Ir a Predicción» pasa a emitir `navigate_requested("prediccion")`.

## Fase D — Preprocesamiento

- [ ] **T-016.** `QToolBox` con las páginas Calidad, Tipos y fechas, Normalización, Columnas a eliminar y Distribución de clases. Los `QGroupBox` se construyen sin añadirlos al layout raíz antes de meterlos en `addItem`.
- [ ] **T-017.** «Aplicar preprocesamiento», `progress` y `lbl_result` **fuera** del acordeón, siempre visibles.
- [ ] **T-018.** La página de distribución solo se añade si aplica; si no, no queda un hueco vacío.
- [ ] **T-019.** `ALTURA_TABLA` en `table_norm`, `tbl_distribucion` y `list_cols`.

## Fase E — Entrenamiento

- [ ] **T-020.** `QToolBox` con las páginas Modelo, Ajustes, Balanceo y Selección de atributos.
- [ ] **T-021.** «Entrenar modelo», `progress`, `lbl_status` y `lbl_result` fuera del acordeón.
- [ ] **T-022.** `ALTURA_TABLA` en `tbl_seleccion`; el canvas con `SizePolicy` expandente.

## Fase F — Datos y Predicción

- [ ] **T-023.** `ALTURA_TABLA` en `data_tab.table` y `table_perfil`; en `predict_tab.tbl_preview` y `tbl_predictions`.
- [ ] **T-024.** Revisar `predict_tab` para que su contenido entre en 720 px sin que las tablas empujen los botones.

## Fase G — Limpieza del acoplamiento

- [ ] **T-025.** Buscar y eliminar cualquier acceso a `self.window()` / `centralWidget()` dentro de las pestañas (R-008).

## Fase H — Tests

- [ ] **T-026.** `tests/test_navigation_ui.py`:
  - C-001: no hay `QTabWidget` de nivel superior.
  - C-002: `self.window()` solo aparece en `main_window.py`.
  - C-003: los cinco pasos con su número y título.
  - C-004: `set_estado` y doble señal visual.
  - C-005: ir a un paso conserva el `AppState`.
  - C-006: `Ctrl+3` lleva al paso 3.
  - C-008: tres sub-pestañas en Resultados.
  - C-009: como máximo una página del `QToolBox` visible.
  - C-010: toda tabla y lista con `maximumHeight > 0`.
  - C-013: los atributos congelados existen (R-017).
- [ ] **T-027.** Test de geometría a 1280×720 (C-007): «Entrenar modelo» y «Aplicar preprocesamiento» dentro del área visible.
- [ ] **T-028.** Test de que el canvas es `Expanding` y la figura usa `tight_layout` (C-011).

## Fase I — Documentación

- [ ] **T-029.** `AGENTS.md`: documentar `ui/layout.py`, `ui/sidebar.py`, el patrón `envolver_scroll`, la navegación por señal y la prohibición de `self.window()`.
- [ ] **T-030.** `README.md`: describir la navegación lateral y el acordeón.
- [ ] **T-031.** `specs/README.md`: roadmap 008.

## Fase J — Verificación

- [ ] **T-032.** `black` e `isort` solo sobre los ficheros tocados.
- [ ] **T-033.** `pytest -q` en verde.
- [ ] **T-034.** `grep -R "PySide6\|PyQt\|from ui\." core/` vacío.
- [ ] **T-035.** Captura de las cinco pestañas a 1280×720.

## Criterios cubiertos

| Criterio | Tarea |
|---|---|
| C-001 | T-004, T-026 |
| C-002 | T-025, T-026 |
| C-003 | T-002, T-003, T-026 |
| C-004 | T-002, T-026 |
| C-005 | T-005, T-026 |
| C-006 | T-009, T-026 |
| C-007 | T-008, T-027 |
| C-008 | T-011, T-026 |
| C-009 | T-016, T-020, T-026 |
| C-010 | T-013, T-019, T-022, T-023, T-026 |
| C-011 | T-014, T-022, T-028 |
| C-012 | T-006, T-015 |
| C-013 | T-026 |
| C-014 | T-033 |

## Commits previstos

1. `feat(ui): barra lateral de pasos y panel de contenido con scroll`
2. `refactor(results): sub-pestañas de métricas, comparativa y exportación`
3. `refactor(preprocess): acordeón de bloques de preprocesamiento`
4. `refactor(train): acordeón de configuración de entrenamiento`
5. `fix(ui): desacopla la navegación del acceso a centralWidget`
6. `test: cubre la navegación, el scroll y la densidad de las pestañas`
7. `docs: documenta la estructura de la interfaz y cierra la spec 008`