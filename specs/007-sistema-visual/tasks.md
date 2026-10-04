# SDD-007 — Tareas

Estado: **esperando puerta 2**. Depende de nada.

## Fase A — Fundamentos (núcleo del tema)

- [ ] **T-001.** `ui/theme.py` con `COLORES`, `ESPACIOS`, `RADIOS` y `RADIO_BASE`, con los valores de R-002. Docstring en español explicando que es la fuente única de verdad de la paleta.
- [ ] **T-002.** `build_stylesheet()`: función pura que devuelve el QSS completo interpolando `COLORES`. Debe cubrir `QWidget`, `QGroupBox`, `QPushButton[clase=...]`, `QPushButton:disabled`, `QLineEdit`, `QComboBox`, `QSpinBox`, `QCheckBox`, `QRadioButton`, `QTableWidget` (cabecera, rejilla, selección), `QTabBar`/`QTabWidget`, `QToolBox`, `QListWidget`, `QTextEdit`/`QPlainTextEdit`/`QTextBrowser`, `QProgressBar`, `QScrollBar`, `QStatusBar`, `QToolTip`, `QMessageBox`, `QScrollArea`.
- [ ] **T-003.** `aplicar_fuente(app)` con la lista de preferencia `FAMILIAS`, 10 pt.
- [ ] **T-004.** `apply_theme(app)` = hoja de estilo + fuente + paleta de matplotlib.
- [ ] **T-005.** `marcar(w, clase)` y `propiedad(w, nombre, valor)`: asignan propiedad dinámica y repintan con `unpolish/polish`.
- [ ] **T-006.** `main.py` llama a `apply_theme(app)` antes de `MainWindow(...)`.

## Fase B — Marca y recursos

- [ ] **T-007.** Copiar `analisis_3er_grado/web/public/img/mined-logo.png` → `resources/escudo-mec.png` y `favicon.svg` → `resources/icono-app.svg`.
- [ ] **T-008.** `resource_path(nombre)` con soporte `sys._MEIPASS` y `FileNotFoundError` explícito.
- [ ] **T-009.** `icono_app()` devuelve un `QIcon` desde `resources/icono-app.svg`; `escudo_pixmap(px)` para la cabecera.
- [ ] **T-010.** `MainWindow` fija el icono de la ventana; `main.py` fija `setWindowIcon` en la aplicación.
- [ ] **T-011.** Borrar `resources/style.qss` (vacío y sin cargar).

## Fase C — Gráficos

- [ ] **T-012.** `ui/plotting.py` con `aplicar_paleta_matplotlib()`: `plt.style.use("default")` + `rcParams` de eje, rejilla, tipografía y ciclo de colores.
- [ ] **T-013.** Verificar que las 8 figuras existentes heredan la paleta sin cambiar una línea de los módulos que las crean.

## Fase D — Empaquetado

- [ ] **T-014.** `AutoML_MEC_Desktop.spec`: `datas=[("resources", "resources")]` e `icon="resources/icono-app.svg"`.
- [ ] **T-015.** `pyproject.toml`: `include-package-data = true` (ya aplicado en la fase 0) y comprobar que los recursos viajan en la rueda.

## Fase E — Aplicación del tema a la interfaz

- [ ] **T-016.** Marcar los botones principales: `Entrenar modelo`, `Aplicar preprocesamiento`, `Guardar modelo y dataset`, `Comparar modelos compatibles` → `primario`.
- [ ] **T-017.** Marcar como `peligro` los que destruyen datos: `Reset`, `Limpiar operaciones`, `Eliminar` de la lista de columnas.
- [ ] **T-018.** Títulos de los `QGroupBox` con el color `azul_med` y peso 600.

## Fase F — Tests

- [ ] **T-019.** `tests/test_theme.py` con `qapp` fixture y `QT_QPA_PLATFORM=offscreen`.
  - C-001: hoja no vacía, sin `@token` ni `var(--x)` sin sustituir.
  - C-002: todo hex de la hoja está en `COLORES`.
  - C-003: idempotencia de `apply_theme`.
  - C-004: los recursos son PNG/SVG válidos.
  - C-005: `resource_path` en modo desarrollo y con `_MEIPASS` simulado.
  - C-006: `resource_path` de un fichero inexistente lanza y nombra el fichero.
  - C-007: `MainWindow` tiene icono no nulo.
  - C-008: `marcar` deja la propiedad y repinta.
  - C-009: `aplicar_paleta_matplotlib` cambia `rcParams`.
  - C-010: el `.spec` declara `datas` e `icon`.
- [ ] **T-020.** Test de contraste: `texto` sobre `papel` y blanco sobre `azul_med` por encima de 4.5:1.

## Fase G — Documentación

- [ ] **T-021.** `AGENTS.md`: documentar `ui/theme.py`, `ui/plotting.py`, la limitación del QSS (sin variables CSS) y la regla de que los colores salen de `COLORES`.
- [ ] **T-022.** `README.md`: sección «Identidad visual» con la paleta y la marca.
- [ ] **T-023.** `specs/README.md`: añadir 007/008/009 al roadmap.
- [ ] **T-024.** `ui/__init__.py` y estructura de `resources/` coherentes con AGENTS §2.

## Fase H — Verificación

- [ ] **T-025.** `black` e `isort` solo sobre los ficheros tocados (nunca `.`).
- [ ] **T-026.** `pytest -q` en verde.
- [ ] **T-027.** `grep -R "PySide6\|PyQt\|from ui\." core/` vacío.
- [ ] **T-028.** Captura de la ventana en 1280×720 para revisar de verdad el aspecto.

## Criterios cubiertos

| Criterio | Tarea |
|---|---|
| C-001 | T-019 |
| C-002 | T-019 |
| C-003 | T-019 |
| C-004 | T-019 |
| C-005 | T-008, T-019 |
| C-006 | T-008, T-019 |
| C-007 | T-010, T-019 |
| C-008 | T-005, T-019 |
| C-009 | T-012, T-019 |
| C-010 | T-014, T-015 |
| C-011 | T-026, T-027 |

## Commits previstos

1. `feat(theme): tokens de color, tipografía y hoja de estilo corporativa`
2. `feat(brand): escudo e icono del ministerio, con recursos empaquetados`
3. `feat(plot): paleta de matplotlib alineada con la interfaz`
4. `test: cubre el sistema visual y el empaquetado de recursos`
5. `docs: documenta el sistema visual y cierra la spec 007`