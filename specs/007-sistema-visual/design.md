# SDD-007 — Diseño

## 1. Arquitectura de la capa visual

```
main.py
  └─ apply_theme(app)            ← ui/theme.py: tokens + QSS + fuente
       └─ MainWindow(...)        ← ui/main_window.py
```

`ui/theme.py` es un módulo **de apoyo visual**, no un `catch-all`: contiene la
paleta, los espaciados y la construcción de la hoja de estilo. No se le permite
contener lógica de widgets ni de datos.

### Tokens

```python
COLORES = {
    "marino": "#12223b", "azul_osc": "#193867", "azul_med": "#14448c",
    "azul_clar": "#446dab", "acento": "#d34223", "oro": "#81660d",
    "verde": "#1c7c4d", "ambar": "#b8860b",
    "fondo": "#f3f4f7", "papel": "#ffffff", "texto": "#20242c",
    "suave": "#5b6168", "borde": "#d7dbe2",
}
ESPACIOS = {"xs": 4, "sm": 8, "md": 12, "lg": 16, "xl": 24}
RADIOS = {"sm": 4, "md": 6, "lg": 10}
```

`verde` y `ambar` se añaden a la paleta del sitio porque en esta aplicación
sí tienen significado: **operaciones con éxito** y **avisos no bloqueantes**.
En el sitio web eran colores de series; aquí son estados.

## 2. Por qué el QSS se genera desde Python

Qt implementa un subconjunto de CSS 2.1. **No soporta variables CSS**
(`var(--x)`), ni `flex`, ni `grid`, ni `box-shadow`, ni `gap`, ni
`text-overflow`. Consecuencias de diseño:

| Se quiere | Cómo se consigue |
|---|---|
| Tokens de color | `f`-string que interpola `COLORES[...]` en el texto del QSS |
| Rejilla y flexbox | `QGridLayout` / `QHBoxLayout` / `QFormLayout`, **nunca** CSS |
| Espaciado | `ESPACIOS` en los `setContentsMargins`/`setSpacing` de Python |
| Bordes y radios | QSS (`border`, `border-radius`) |
| Jerarquía de botones | Propiedad dinámica `clase` + `unpolish/polish` |

Se descarta deliberadamente el `qproperty` de Qt y el `Theme` de
`QtCharts`/`QtQuick`: no existen en PySide6 tal cual.

### Idempotencia (R-003)

`build_stylesheet()` es una función pura: no acumula texto, siempre devuelve
lo mismo. `apply_theme()` hace `app.setStyleSheet(build_stylesheet())` una
única vez y `app.setFont(...)`. Como `setStyleSheet` **reemplaza** en vez de
añadir, llamarlo dos veces no duplica reglas: por eso la idempotencia es
trivial y se prueba solo por si alguien lo rompe después.

## 3. Tipografía

```python
FAMILIAS = ["Ubuntu", "Noto Sans", "DejaVu Sans", "Segoe UI"]
```

Se usa `QFont.setFamilies()`, que acepta la lista y deja que fontconfig elija.
Ubuntu es la primera por coherencia con el sitio del proyecto ERCE, pero
**no está instalada** en el sistema (`fc-list` no la encuentra), así que en la
práctica se usará Noto Sans. La lista es deliberadamente una *preferencia*,
no una dependencia: R-006 prohíbe empaquetar fuentes.

Tamaño base 10 pt. Los pesos se declaran explícitamente en QSS (`font-weight:
600`) porque Qt no admite pesos variables en el QSS.

## 4. Marca

| Recurso | Origen | Uso |
|---|---|---|
| `resources/escudo-mec.png` | `analisis_3er_grado/web/public/img/mined-logo.png` | 104×117, cabecera de la barra lateral a 32 px lógicos |
| `resources/icono-app.svg` | `analisis_3er_grado/web/public/favicon.svg` | Icono de ventana y de la app |

`favicon.svg` es un SVG con la misma paleta (`#12223b`, `#446dab`, `#d34223`),
lo que confirma que el azul-rojo Pedido es el institucional y no una
interpretación.

### `resource_path`

```python
def resource_path(nombre):
    base = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    ruta = base / "resources" / nombre
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el recurso «{nombre}» en {ruta}")
    return str(ruta)
```

Falla ruidosamente: devolver una ruta inexistente produciría un `QPixmap`
vacío y un espacio en blanco sin explicación.

## 5. Jerarquía de botones

`marcar(w, clase)`:

```python
w.setProperty("clase", clase)
w.style().unpolish(w)
w.style().polish(w)
```

El `unpolish/polish` es obligatorio: Qt guarda la hoja de estilo ya resuelta en
el widget y un cambio de propiedad dinámica no la invalida. Este es el fallo
clásico al añadir temas en Qt, y por eso R-014 tiene su propio criterio.

El QSS selecciona por propiedad con `[clase="primario"]`.

## 6. Gráficos

`ui/plotting.py::aplicar_paleta_matplotlib()` fija `rcParams` con la misma
paleta: ejes y rejilla en `borde`/`suave`, fondo `papel`, texto `texto`, y el
ciclo de colores en `azul_med`, `acento`, `azul_clar`, `oro`.

Vive en `ui/` y no en `core/` por dos razones: `core/` no puede importar Qt
(AGENTS §3.1) y los gráficos son ua responsabilidad de la presentación. Se
llama una vez desde `apply_theme()` para que **todas** las figuras nascan ya
con la paleta, sin tocar los 8 sitios donde se crean figuras.

`plt.style.use("default")` antes de fijar, para que el estilo global del
entorno no se cuele.

## 7. Empaquetado

`AutoML_MEC_Desktop.spec`:

```python
datas=[("resources", "resources")],
icon="resources/icono-app.svg",
```

`.gitignore` tenía `*.spec`, que dejaba el fichero fuera del control de
versiones; ya se corrigió en la rama para que la compilación sea reproducible.

## 8. Contraste

| Combinación | Ratio | Decisión |
|---|---|---|
| `texto` sobre `papel` | 14.6:1 | Correcto |
| blanco sobre `azul_med` | 8.6:1 | Correcto |
| `suave` sobre `fondo` | 4.9:1 | Correcto para texto secundario |
| blanco sobre `acento` | 4.6:1 | Correcto para botones de peligro |

## 9. Alternativas descartadas

| Alternativa | Por qué no |
|---|---|
| QSS con `var(--token)` | Qt no lo soporta; el QSS se quedaría literal |
| Tema oscuro | Decidido que no |
| Empaquetar Ubuntu | Dependencia nueva y 500 KB por una fuente |
| `QGraphicsDropShadowEffect` | Repinta el canvas de matplotlib en cada cambio |
| Hoja de estilo en `resources/` | Reintroduce el fichero vacío que ya se elimina; Python es la fuente única |