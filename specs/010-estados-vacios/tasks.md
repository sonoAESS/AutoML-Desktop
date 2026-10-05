# SDD-010 — Tareas

Estado: **completada**.

## Fase A — Componente

- [x] **T-001.** `ui/empty_state.py` con `EmptyState`: título, detalle y siguiente paso, con los roles `titulo`, `suave` y `seccion` de la hoja de estilo.
- [x] **T-002.** `mostrar(visible)` idempotente, sin recrear el widget.
- [x] **T-003.** `alternar_estado_vacio(estado, widgets, vacio)`.

## Fase B — Las cinco pestañas

- [x] **T-004.** `data_tab`: oculta `table` y `table_perfil` sin dataset.
- [x] **T-005.** `preprocess_tab`: oculta el acordeón sin datos; **`btn_apply` nunca**.
- [x] **T-006.** `train_tab`: oculta el formulario de «Modelo» sin dataset limpio; el acordeón y «Entrenar» nunca.
- [x] **T-007.** `results_tab`: oculta `tbl_metricas` y `canvas` sin modelo.
- [x] **T-008.** `predict_tab`: oculta las dos tablas sin bundle ni datos.

## Fase C — Textos

- [x] **T-009.** Los cinco mensajes nombran el paso siguiente.
- [x] **T-010.** `_alternar_vacio()` llamado desde `refresh()` y desde la construcción.

## Fase D — Tests

- [x] **T-011.** `tests/test_empty_state_ui.py`:
  - C-001: niveles de texto e idempotencia.
  - C-002: las cinco pestañas muestran el mensaje sin datos.
  - C-003: ninguna lo muestra con datos.
  - C-004: el mensaje cita el número del paso correcto según `PASOS`.
  - C-005: «Aplicar preprocesamiento» sigue visible sin datos.
  - C-006: los atributos del contrato siguen existiendo.
  - C-007: ninguna tabla queda visible y vacía a la vez que el mensaje.

## Fase E — Documentación

- [x] **T-012.** `AGENTS.md`: el patrón del estado vacío y por qué no lleva botón.
- [x] **T-013.** `README.md`: sección «Cuando una pestaña está vacía».
- [x] **T-014.** `specs/README.md`: roadmap 010.

## Fase F — Verificación

- [x] **T-015.** `black` e `isort` sobre los ficheros tocados.
- [x] **T-016.** `pytest -q` en verde.
- [x] **T-017.** `grep -R "PySide6\|PyQt\|from ui\." core/` vacío.
- [x] **T-018.** Capturas de las cinco pestañas vacías y con datos.

## Commits previstos

1. `feat(ui): componente de estado vacío reutilizable`
2. `feat(ui): las cinco pestañas explican qué les falta`
3. `test: cubre los estados vacíos de las cinco pestañas`
4. `docs: documenta los estados vacíos y cierra la spec 010`
