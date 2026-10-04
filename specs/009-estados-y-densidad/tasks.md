# SDD-009 — Tareas

Estado: **esperando puerta 2**. Depende de SDD-007 y SDD-008.

## Fase A — Utilidad compartida

- [ ] **T-001.** `ui/enabled.py` (nuevo) con `habilitar_hijos(padre, habilitado)`; docstring en español. Sin Qt fuera de `ui/`.

## Fase B — Preprocesamiento (el caso extremo)

- [ ] **T-002.** `_update_enabled_state()`: sin `raw_df`, todo deshabilitado menos la carga.
- [ ] **T-003.** Condiciones por perfil semántico: sin numéricas, `cb_num` fuera; sin categóricas, `cb_cat` fuera.
- [ ] **T-004.** Tooltips con el motivo cuando se deshabilita.
- [ ] **T-005.** Invocar desde `refresh()` y desde `apply()`.

## Fase C — Datos

- [ ] **T-006.** `_update_enabled_state()` para `btn_usar_objetivo`, `btn_reperfil` y las tablas.

## Fase D — Entrenamiento

- [ ] **T-007.** `_update_enabled_state()`: `btn_train` exige `clean_df`, objetivo y modelo.
- [ ] **T-008.** Unificar `_toggle_tune_options` y `_refresh_balancing` con el mismo criterio de `_update_enabled_state`, sin duplicar reglas.
- [ ] **T-009.** Invocar desde `refresh()` y tras `train()`/`analizar_seleccion()`.

## Fase E — Resultados

- [ ] **T-010.** `_update_enabled_state()`: `btn_save`, `btn_export`, `btn_load_in_predict`, `tbl_metricas` y canvas exigen `pipeline`.
- [ ] **T-011.** Conservar la condición de ≥3 modelos de la comparativa.

## Fase F — Predicción

- [ ] **T-012.** `_update_enabled_state()`: `btn_predict` exige bundle **y** datos; `btn_export` y `btn_add_to_data` exigen predicciones.
- [ ] **T-013.** Invocar desde `refresh()`, `load_bundle()`, `load_data()` y `predict()`.

## Fase G — Densidad

- [ ] **T-014.** Acortar los botones >28 caracteres según la tabla del diseño, moviendo el detalle al tooltip.
- [ ] **T-015.** `toolTip` en todo control interactivo sin etiqueta.
- [ ] **T-016.** Clase `peligro` en los botones que destruyen datos.
- [ ] **T-017.** Ninguna fila horizontal de nivel superior con más de 5 widgets.

## Fase H — Barra de estado

- [ ] **T-018.** `MainWindow._estado(texto, ok)` con color `suave`/`acento`.
- [ ] **T-019.** Mensajes distintos por paso y por resultado; sin repetir la etiqueta de la pestaña.

## Fase I — Tests

- [ ] **T-020.** `tests/test_estados_ui.py`:
  - C-001: recuento exacto de controles deshabilitados en Preprocesamiento sin datos.
  - C-002: sin numéricas, `cb_num` deshabilitado y `cb_cat` habilitado.
  - C-003: idempotencia de `_update_enabled_state()` en las cinco pestañas.
  - C-004: la jerarquía deshabilita a los hijos.
  - C-005: sin modelo, los tres botones de Resultados fuera y con tooltip.
  - C-006: recorrido de Predicción sin bundle, con bundle y con bundle + datos.
  - C-007: ningún texto de botón >28 caracteres.
  - C-008: todo control interactivo sin etiqueta tiene tooltip.
  - C-009: ninguna fila de primer nivel con más de 5 widgets.
  - C-010: la barra de estado cambia y distingue éxito de fallo.

## Fase J — Documentación

- [ ] **T-021.** `AGENTS.md`: la regla de que el estado de los controles se deriva solo de `AppState`.
- [ ] **T-022.** `README.md`: sección «Controles activos y por qué».
- [ ] **T-023.** `specs/README.md`: roadmap 009.

## Fase K — Verificación

- [ ] **T-024.** `black` e `isort` solo sobre los ficheros tocados.
- [ ] **T-025.** `pytest -q` en verde.
- [ ] **T-026.** `grep -R "PySide6\|PyQt\|from ui\." core/` vacío.

## Criterios cubiertos

| Criterio | Tarea |
|---|---|
| C-001 | T-002, T-020 |
| C-002 | T-003, T-020 |
| C-003 | T-020 |
| C-004 | T-001, T-020 |
| C-005 | T-010, T-020 |
| C-006 | T-012, T-020 |
| C-007 | T-014, T-020 |
| C-008 | T-015, T-020 |
| C-009 | T-017, T-020 |
| C-010 | T-018, T-019, T-020 |
| C-011 | T-025 |
| C-012 | T-025 |

## Commits previstos

1. `feat(ui): los controles de preprocesamiento solo se activan si aplican`
2. `feat(ui): estado de los botones de entrenamiento, resultados y predicción`
3. `refactor(ui): textos de botón cortos con el detalle en el tooltip`
4. `feat(ui): la barra de estado dice en qué paso estás`
5. `test: cubre los estados de los controles y la densidad de las filas`
6. `docs: documenta los estados de la interfaz y cierra la spec 009`