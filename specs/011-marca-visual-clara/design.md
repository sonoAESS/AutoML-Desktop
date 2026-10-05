# 011 — Marca visual clara · Diseño

**Spec:** [spec.md](spec.md) · **Orden:** 1 · **Depende de:** —

---

## D1 · Token `lateral` en lugar de reutilizar `marino`

`marino` (`#12223b`) se usa en diez sitios. Solo uno es el fondo de la barra
lateral. Se añade un token nuevo en lugar de tocar el existente:

```python
COLORES["lateral"] = "#24507f"
```

La hoja pasa de `QFrame#lateral { background-color: marino }` a
`QFrame#lateral { background-color: lateral }`. Ningún otro selector cambia.

**Alternativa descartada:** aclarar `marino` directamente. Cambiaría la
cabecera de tabla, el `pressed` del botón primario, el color de título, el
tooltip y el `axes.titlecolor` de matplotlib — nueve efectos no pedidos.

## D2 · Contraste del texto secundario

| Token | Antes | Después | Contraste sobre `#24507f` |
|---|---|---|---|
| `sidebar_texto` | `#a9bbd4` | `#dbe6f3` | 4.25:1 → **6.57:1** |
| `sidebar_hover` | `#1c3054` | `#1c3c66` | re-derivado de la base nueva |

El título (`blanco` sobre `lateral`) queda en **8.30:1**. Los tres superan
WCAG AA (4.5:1 para texto normal).

El contraste se **calcula en el test** con la fórmula WCAG relativa, no se
hardcodea un número: si alguien cambia el token, el test sigue verificando la
regla.

## D3 · Escudo a 64 px y centrado

`escudo_pixmap(px: int = 32)` → `escudo_pixmap(px: int = 64)`. El sidebar llama
`escudo_pixmap()` sin argumento, así que hereda el nuevo valor.

La etiqueta pasa de `setFixedHeight(44)` a `setFixedHeight(72)` y de
`AlignLeft | AlignVCenter` a `AlignHCenter | AlignVCenter`.

**Comprobación de espacio:** el escudo original es 104×117 (proporción 0.889).
A 64 px de alto ocupa ~57 px de ancho, dentro de los 210 px del sidebar. La
altura mínima del sidebar medida es 370 px sobre un presupuesto de 720 px.

## D4 · Marca centrada y sin duplicados

Se elimina la etiqueta `organizacion` (la que mostraba `ORGANIZACION` bajo el
título). El texto completo «Ministerio de Educación Superior — Cuba» vive
solo en `pie`, que pasa a usar `ORGANIZACION` en lugar del literal recortado
actual.

El bloque de marca se centra en conjunto:

| Widget | Antes | Después |
|---|---|---|
| `marca` (escudo) | `AlignLeft \| AlignVCenter` | `AlignHCenter \| AlignVCenter` |
| `titulo` | alineación por defecto (izquierda) | `AlignHCenter` |
| `pie` | alineación por defecto (izquierda) | `AlignHCenter` |

## D5 · Icono de aplicación

Ya implementado fuera de la spec (petición directa del usuario):

- `resources/icono-app.svg` — rediseñado con la paleta (`#14448c`, `#d34223`).
- `resources/icono-app.png` — 256×256 generado con Pillow.
- `resources/icono-app.ico` — siete tamaños (16/24/32/48/64/128/256).
- `AutoML_MEC_Desktop.spec` — `icon='resources/icono-app.ico'`.

`icono_app()` sigue devolviendo el `QIcon` del SVG: Qt lo rasteriza a cualquier
tamaño y es el formato que `setWindowIcon` prefiere. El `.ico` es para el
ejecutable de Windows.

## D6 · Limpieza: línea duplicada

`ui/theme.py:413` y `:414` son byte a byte la misma regla
`QLabel[role="titulo"]`. Se elimina una. No cambia el comportamiento, pero la
hoja deja de arrastrar un defecto.

## Ficheros

| Fichero | Cambio |
|---|---|
| `ui/theme.py` | token `lateral`, dos `TONOS` aclarados, `escudo_pixmap(64)`, `QFrame#lateral` usa `lateral`, eliminar línea duplicada |
| `ui/sidebar.py` | quitar `organizacion`, centrar marca y pie, `pie` usa `ORGANIZACION` |
| `tests/test_theme.py` | contraste calculado, centrado del escudo, un solo «Ministerio», ICO de siete tamaños |
| `tests/test_navigation_ui.py` | el pie con el texto completo |

`core/` no se toca. `ui/main_window.py` no se toca.
