# SDD-007 — Sistema visual y marca institucional

| Campo | Valor |
|---|---|
| Estado | Especificación — **esperando puerta 1** |
| Orden de ejecución | 1 de 3 (de la rama `design/redesign-ui`) |
| Ámbito | `ui/theme.py` (nuevo), `main.py`, `resources/` (nuevo), `AutoML_MEC_Desktop.spec`, `pyproject.toml`, `ui/plotting.py` (nuevo), `tests/test_theme.py` |
| Depende de | — |
| Bloquea a | SDD-008, SDD-009 |

## 1. Problema

La aplicación se ve exactamente como PySide6 por defecto: fondo gris del
sistema, botones grises, tipografía por defecto, sin identidad. No hay ni un
ni un solo color propio, y el usuario finales no tiene forma de saber que la
herramienta pertenece a la institución.

Hay dos señales de que esto ya se anticipó y se quedó a medias:

- `resources/style.qss` existe pero tiene **0 bytes** y nadie lo carga.
- `main.py` no aplica hoja de estilo, ni fuente, ni icono.

Además, la marca (el escudo del Ministerio) vive en otro proyecto y nunca se
copió, así que la app distribuible se abre con el icono genérico de Python.

## 2. Objetivo

Una identidad visual coherente con el Ministerio de Educación Superior de
Cuba: **azules, rojos y blancos**, con el escudo en la cabecera, aplicada de
forma centralizada y reproducible (los mismos tokens tanto en desarrollo
como en la app compilada).

## 3. Alcance

### 3.1 Incluido

- `ui/theme.py` como fuente única de verdad: paleta, espaciados, radios,
  tipografía y la hoja de estilo completa.
- Escudo e icono de la aplicación, y empaquetado de los recursos.
- Paleta de matplotlib alineada con la de la interfaz.
- Jerarquía visual de botones: primario, secundario y de peligro.

### 3.2 No incluido

- Tema oscuro conmutable (decidido: solo tema claro).
- Empaquetar tipografías. La familia Ubuntu **no está instalada** en el
  sistema y añadirla como dependencia contradice AGENTS §7.3.
- Sombras, degradados animados ni animaciones: el QSS de Qt no las soporta
  de forma fiable y `QGraphicsDropShadowEffect` penaliza el repintado de los
  canvas de matplotlib.
- Rediseño del layout de las pestañas: es SDD-008.

## 4. Requisitos

### Tokens y hoja de estilo

- **R-001.** `ui/theme.py` expone `COLORES: dict[str, str]` con la paleta
  completa y `build_stylesheet() -> str`. **No existirán literales de color
  fuera de ese diccionario** en el resto de `ui/`.
- **R-002.** La paleta es la ya usada en el sitio del proyecto ERCE, para que
  ambas piezas compartan identidad:

  | Token | Valor | Uso |
  |---|---|---|
  | `marino` | `#12223b` | Fondo de la cabecera, texto de la barra lateral |
  | `azul_osc` | `#193867` | Bordes oscuros, hover |
  | `azul_med` | `#14448c` | Botón primario, enlaces, pestaña activa |
  | `azul_clar` | `#446dab` | Bordes de foco, acentos secundarios |
  | `acento` | `#d34223` | Rojo institucional: peligro, borrados |
  | `oro` | `#81660d` | Avisos no bloqueantes |
  | `fondo` | `#f3f4f7` | Fondo de la ventana |
  | `papel` | `#ffffff` | Superficies de tarjetas y tablas |
  | `texto` | `#20242c` | Texto principal |
  | `suave` | `#5b6168` | Texto secundario, texto deshabilitado |
  | `borde` | `#d7dbe2` | Bordes de 1 px |

- **R-003.** `apply_theme(app: QApplication) -> None` aplica la hoja de estilo
  y la fuente, y es **idempotente**: llamarlo dos veces no duplica reglas.
- **R-004.** `main.py` invoca `apply_theme(app)` **antes** de construir
  `MainWindow`.
- **R-005.** El QSS se genera desde Python porque **Qt no soporta variables
  CSS**: no se usará `var(--token)` ni `@token` en la hoja. Esta limitación
  queda documentada en `AGENTS.md` para que nadie intente usarla después.

### Tipografía

- **R-006.** La familia se elige de una lista de preferencia
  `["Ubuntu", "Noto Sans", "DejaVu Sans", "Segoe UI"]` tomando la primera
  disponible, con `QFont.setFamilies`. Tamaño base 10 pt.
- **R-007.** La hoja de estilo fija tamaño y familia en `QWidget` para que los
  widgets creados después hereden los valores.

### Marca y recursos

- **R-008.** `resources/` contiene el escudo (`escudo-mec.png`) y el icono de
  la aplicación (`icono-app.svg`), copiados del proyecto ERCE.
- **R-009.** `resource_path(nombre: str) -> str` resuelve rutas relativas tanto
  en desarrollo como en la app compilada con PyInstaller
  (`sys._MEIPASS`), y **lanza `FileNotFoundError` con un mensaje claro** si el
  recurso no está, en lugar de devolver una ruta inválida.
- **R-010.** La ventana y la aplicación usan el icono; el `QApplication` fija
  `setWindowIcon`.
- **R-011.** `AutoML_MEC_Desktop.spec` declara `datas` con `resources` e `icon`
  con el icono de la aplicación, de modo que la app compilada muestre el escudo
  y encuentre los recursos.
- **R-012.** `resources/style.qss` **se elimina**: está vacío y nunca se
  cargó. `resources/` pasa a contener solo binarios.

### Jerarquía visual

- **R-013.** Tres clases de botón, marcadas con una propiedad dinámica
  `clase` y no con estilos inline:
  - `primario`: fondo `azul_med`, texto blanco.
  - `secundario`: fondo `papel`, borde `borde`, texto `azul_med`.
  - `peligro`: fondo `acento`, texto blanco (borrados, reinicios).
- **R-014.** `marcar(w: QWidget, clase: str)` asigna la propiedad y llama a
  `style().unpolish/polish`, porque Qt no repinta los cambios de propiedad
  dinámico por su cuenta. Un test lo verifica.
- **R-015.** Los componentes con los que más se interactúa (botón «Entrenar
  modelo», «Aplicar preprocesamiento», «Guardar modelo») son `primario`.

### Gráficos

- **R-016.** `ui/plotting.py` expone `aplicar_paleta_matplotlib()`, que fija
  `rcParams` (colores de ejes, rejilla, tipografía y ciclo de colores) con la
  misma paleta de la interfaz. Vive en `ui/` porque `core/` no puede importar
  nada de Qt ni depender de la capa visual.
- **R-017.** El ciclo de colores por defecto de los gráficos usa `azul_med`,
  `acento`, `azul_clar` y `oro`, para que las series de un gráfico no
  aparezcan con los colores genéricos de matplotlib.

## 5. Invariantes

| Invariante | Origen |
|---|---|
| `core/` no importa PySide6, PyQt ni `ui/` | AGENTS §3.1 |
| `core/` no cambia: esta spec no lo toca | AGENTS §7.4 |
| Sin dependencias nuevas | AGENTS §7.3 |
| Mensajes al usuario en español | AGENTS §4 |
| Los 314 tests existentes siguen en verde | AGENTS §8 |
| Atributos de widget existentes no se renombran | AGENTS §8 |

## 6. Criterios de aceptación

- **C-001.** `build_stylesheet()` devuelve una cadena no vacía y sin ninguna
  secuencia `@token` o `var(--x)` sin sustituir.
- **C-002.** Todos los colores que aparecen en la hoja de estilo pertenecen a
  `COLORES`; ningún hex suelto fuera del diccionario.
- **C-003.** `apply_theme(app)` puede llamarse dos veces y el resultado es
  idéntico (test de idempotencia).
- **C-004.** El escudo y el icono existen en `resources/` y son PNG/SVG válidos.
- **C-005.** `resource_path("escudo-mec.png")` devuelve una ruta existente en
  modo desarrollo, y con `sys._MEIPASS` simulado también.
- **C-006.** `resource_path("no-existe.png")` lanza `FileNotFoundError` con el
  nombre del fichero en el mensaje.
- **C-007.** La ventana tiene icono no nulo tras `MainWindow()`.
- **C-008.** `marcar(boton, "primario")` deja `boton.property("clase")` igual a
  `"primario"` y el estilo repintado.
- **C-009.** `aplicar_paleta_matplotlib()` cambia los `rcParams` y no affecta a
  `core/`.
- **C-010.** El `.spec` declara `datas` e `icon`, y `pyproject.toml` incluye
  los recursos como package-data.
- **C-011.** La suite completa pasa y `grep -R "PySide6\|from ui\." core/` no
  devuelve nada.

## 7. Riesgos

| Riesgo | Mitigación |
|---|---|
| El QSS de Qt no soporta variables CSS, flexbox ni `box-shadow` | Se documenta en AGENTS §5; el diseño se apoya en layouts, no en CSS |
| El escudo es de 104×117 px y se verá pequeño en pantallas HiDPI | Se usa a 32 px lógicos en la cabecera; para HiDPI se documenta el escalado |
| Cambiar colores puede reducir contraste | `texto` sobre `papel` y blanco sobre `azul_med` se comprueban con el test de contraste |
| El `.spec` no se puede probar en el CI | Se valida que las rutas del `.spec` existen en el repo |