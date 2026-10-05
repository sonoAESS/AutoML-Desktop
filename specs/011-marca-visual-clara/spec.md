# 011 — Marca visual clara

**Titular:** Aclarar el azul de la barra lateral, agrandar y centrar el escudo,
y quitar el texto «Ministerio de Educación» duplicado.

**Orden:** 1 · **Depende de:** — · **Rama:** `design/redesign-ui`

---

## Contexto

La barra lateral usa `marino` (`#12223b`) como fondo. Es un azul muy oscuro que
cansa en pantallas grandes y que además está sobrecargado: el mismo token se
usa en diez sitios distintos (fondo de la barra, cabecera de tabla, estado
`pressed` del botón primario, color de título, tooltip, …). Aclarar el token
entero cambiaría nueve cosas que el usuario no pidió.

El escudo mide 34 px y está pegado al borde izquierdo en `x = 0`, sin centrar.
El texto «Ministerio de Educación Superior» aparece dos veces: una bajo el
título de la barra y otra en el pie.

## Requisitos

### R-001 · Un token propio para el fondo de la barra

Se añade `COLORES["lateral"] = "#24507f"`. Lo usa **solo** `QFrame#lateral`.
`marino` conserva su valor `#12223b` y sus otros nueve usos.

**Motivo:** un token por responsabilidad. `marino` significa «azul marino» y
lo usan la cabecera de tabla, el `pressed` del botón primario y el color de
título; el fondo de la barra lateral es otra cosa y merece su propio nombre.

### R-002 · Contraste del texto sobre el fondo nuevo

`#24507f` con texto blanco da **8.30:1** (antes 15.94:1). Pero el texto
secundario `TONOS["sidebar_texto"]` (`#a9bbd4`) sobre `#24507f` da solo
**4.25:1**, por debajo de WCAG AA (4.5:1). Se aclara a `#dbe6f3`, que da
**6.57:1**.

`TONOS["sidebar_hover"]` (`#1c3054`) se re-deriva de la base nueva como
`#1c3c66`, para que el hover siga siendo visible sin quedar huérfano.

### R-003 · Escudo más grande y centrado

`escudo_pixmap(34)` → `escudo_pixmap(64)`. La etiqueta pasa de 44 px a 72 px
de alto y se alinea con `AlignHCenter | AlignVCenter`. El escudo deja de estar
pegado al borde izquierdo.

**Comprobación de espacio:** el sidebar mide 210 px de ancho y el escudo a
64 px de alto ocupa ~57 px de ancho (proporción 104:117). La altura mínima del
sidebar medida es 370 px sobre un presupuesto de 720 px, así que hay ~350 px
de margen.

### R-004 · Marca centrada y sin duplicados

Se elimina la etiqueta `ORGANIZACION` de arriba. El texto completo
«Ministerio de Educación Superior — Cuba» vive solo en el pie (`pie`).

El bloque de marca (escudo, título y pie) se centra en conjunto, de modo que
«centrado como el texto de abajo» sea literalmente cierto.

### R-005 · Icono de aplicación nuevo

Se sustituye el icono anterior (un cubo y un círculo genéricos) por uno propio:
fondo azul `#14448c` con esquinas redondeadas, tres barras blancas ascendentes
y una flecha de tendencia roja `#d34223`. Se generan `icono-app.png` (256 px)
e `icono-app.ico` (16–256 px, siete tamaños) y el `.spec` de PyInstaller pasa
a usar el `.ico` (PyInstaller no acepta `.svg` como icono en Windows).

## Invariantes que no se rompen

| Invariante | Origen |
|---|---|
| `core/` no importa PySide6, PyQt ni `ui/` | `AGENTS.md` §3.1 |
| Todo color vive en `ui/theme.py` (`COLORES` / `TONOS`) | `AGENTS.md` §4 |
| Ningún color de la hoja fuera de los tokens | spec 007, R-001 |
| El título de la marca es blanco sobre azul | spec 007 |
| `resources/` solo contiene binarios | spec 007 |
| La UI no contiene lógica científica | `AGENTS.md` §3.5 |

## Criterios de aceptación

| ID | Criterio |
|---|---|
| C-001 | `theme.COLORES["lateral"] == "#24507f"` y `theme.COLORES["marino"] == "#12223b"` (este último sigue existiendo). |
| C-002 | La hoja de estilo contiene `QFrame#lateral { background-color: #24507f }` y ningún otro selector usa `marino` como fondo de la barra. |
| C-003 | Contraste de `COLORES["blanco"]` sobre `COLORES["lateral"]` ≥ 7:1 y de `TONOS["sidebar_texto"]` sobre `COLORES["lateral"]` ≥ 4.5:1, calculado en el test y no hardcodeado. |
| C-004 | `escudo_pixmap()` por defecto devuelve un pixmap de 64 px de alto; el sidebar lo muestra centrado horizontalmente (su `x` más la mitad de su ancho coincide con la mitad del sidebar, con tolerancia de 2 px). |
| C-005 | El sidebar contiene exactamente una etiqueta con el texto «Ministerio de Educación Superior — Cuba» y ninguna con «Ministerio de Educación Superior» a secas. |
| C-006 | `resources/icono-app.ico` existe, es un ICO válido y contiene los siete tamaños 16/24/32/48/64/128/256. |
| C-007 | `AutoML_MEC_Desktop.spec` referencia `resources/icono-app.ico`. |
| C-008 | `tests/test_navigation_ui.py::test_el_sidebar_cabe_en_720` sigue pasando (altura mínima del sidebar ≤ 720). |
| C-009 | La línea duplicada `QLabel[role="titulo"]` de `ui/theme.py:413-414` desaparece y la hoja sigue estilizando `role="titulo"`. |

## No incluido

- Aclarar `azul_med`, `azul_osc` ni `azul_clar`: el usuario pidió solo el fondo
  de la barra lateral.
- Mover el escudo a la cabecera superior.
- Cambiar la tipografía ni el tamaño base.
- Tocar `ui/main_window.py`.
