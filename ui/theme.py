# ui/theme.py
"""Tokens de color, tipografía y hoja de estilo de la aplicación.

Este módulo es la **fuente única de verdad** de la identidad visual. Ningún
otro fichero de `ui/` debe escribir un literal de color: se importa de aquí.

La paleta es la del sitio institucional del proyecto ERCE
(`analisis_3er_grado/web/src/styles/global.css`), para que ambas piezas
compartan imagen.

Por qué el QSS se construye aquí y no en un `.qss`: **Qt no soporta variables
CSS**. No existen `var(--token)` ni `@token`, así que una hoja de estilo con
tokens literales no se puede escribir en un fichero plano sin duplicar la
paleta a mano. Interpolando desde Python, el color se declara una vez.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QIcon, QPixmap

#: Paleta corporativa. Los colores de estado (`verde`, `ambar`) tienen aquí un
#: significado que en el sitio web no tenían: resultado con éxito y aviso.
COLORES: dict[str, str] = {
    "marino": "#12223b",
    "azul_osc": "#193867",
    "azul_med": "#14448c",
    "azul_clar": "#446dab",
    "acento": "#d34223",
    "oro": "#81660d",
    "verde": "#1c7c4d",
    "ambar": "#b8860b",
    "fondo": "#f3f4f7",
    "papel": "#ffffff",
    "texto": "#20242c",
    "suave": "#5b6168",
    "borde": "#d7dbe2",
    "blanco": "#ffffff",
}

#: Tintes derivados de la paleta, con nombre del estado que representan.
#: Viven aquí en vez de escribirse en la hoja de estilo para que la comprobación
#: «ningún color fuera de los tokens» siga teniendo sentido: sin este diccionario
#: habría quetslistas sueltos como `_c('hover')` en el texto del QSS.
TONOS: dict[str, str] = {
    "hover": "#e8eef7",
    "seleccion": "#d8e3f4",
    "inactivo": "#eceef2",
    "inactivo_borde": "#e6eaf2",
    "inactivo_primario": "#b9c4d6",
    "inactivo_texto": "#f0f2f6",
    "sidebar_texto": "#a9bbd4",
    "sidebar_hover": "#1c3054",
    "tabla_alterna": "#f7f9fc",
    "rejilla": "#e6e9ef",
    "tab_hover": "#dfe6f2",
}

#: Escala de espaciado en píxeles lógicos.
ESPACIOS: dict[str, int] = {"xs": 4, "sm": 8, "md": 12, "lg": 16, "xl": 24}

#: Radios de esquina en píxeles.
RADIOS: dict[str, int] = {"sm": 4, "md": 6, "lg": 10}

#: Familias por orden de preferencia. Ubuntu es la primera por coherencia con
#: el sitio institucional, pero no está instalada en el sistema, así que en la
#: práctica se usará Noto Sans. No se empaqueta: sería una dependencia nueva.
FAMILIAS: list[str] = ["Ubuntu", "Noto Sans", "DejaVu Sans", "Segoe UI"]

#: Tamaño base de la interfaz.
TAMANO_BASE = 10

NOMBRE_APP = "AutoML Desktop"
ORGANIZACION = "Ministerio de Educación Superior — Cuba"


def resource_path(nombre: str) -> str:
    """Ruta absoluta de un recurso, en desarrollo o en la app compilada.

    PyInstaller extrae `resources/` en un directorio temporal y publica su
    ruta en `sys._MEIPASS`; en desarrollo vale la carpeta hermana de `ui/`.

    Args:
        nombre: nombre del fichero dentro de `resources/`.

    Returns:
        La ruta absoluta del recurso.

    Raises:
        FileNotFoundError: si el recurso no está. Falla a propósito: devolver
            una ruta inexistente produciría un `QPixmap` vacío y un hueco en
            blanco sin explicación posible.
    """
    base = getattr(sys, "_MEIPASS", None)
    raiz = Path(base) if base else Path(__file__).resolve().parent.parent
    ruta = raiz / "resources" / nombre
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el recurso «{nombre}» en {ruta}. "
            "En la aplicación compilada, ¿se empaquetó la carpeta resources/?"
        )
    return str(ruta)


def icono_app() -> QIcon:
    """Icono de la ventana y de la aplicación."""
    return QIcon(resource_path("icono-app.svg"))


def escudo_pixmap(px: int = 32) -> QPixmap:
    """Escudo del ministerio escalado a `px` píxeles lógicos de alto.

    El PNG original es de 104×117 y se muestra pequeño en la cabecera, así que
    se escala aquí en lugar de confiar en el tamaño del fichero.
    """
    pixmap = QPixmap(resource_path("escudo-mec.png"))
    if pixmap.isNull():
        return pixmap
    return pixmap.scaled(
        px,
        px,
        Qt.KeepAspectRatio,
        Qt.SmoothTransformation,
    )


def aplicar_fuente(app) -> None:
    """Fija la familia tipográfica y el tamaño base de la aplicación."""
    fuente = QFont()
    fuente.setFamilies(FAMILIAS)
    fuente.setPointSize(TAMANO_BASE)
    app.setFont(fuente)


def marcar(w, clase: str) -> None:
    """Asigna una clase visual a un widget y fuerza el repintado.

    Las clases son `primario`, `secundario` y `peligro`; la hoja de estilo las
    selecciona con `[clase="primario"]`.

    El `unpolish/polish` no es opcional: Qt guarda en el widget la hoja de
    estilo ya resuelta y cambiar una propiedad dinámica no la invalida, así
    que sin esta llamada el botón conserva el aspecto anterior.
    """
    w.setProperty("clase", clase)
    estilo = w.style()
    estilo.unpolish(w)
    estilo.polish(w)


def propiedad(w, nombre: str, valor) -> None:
    """Asigna una propiedad dinámica cualquiera y fuerza el repintado."""
    w.setProperty(nombre, valor)
    estilo = w.style()
    estilo.unpolish(w)
    estilo.polish(w)


def _c(nombre: str) -> str:
    """Valor de un token: de `COLORES` o de los tintes `TONOS`."""
    if nombre in COLORES:
        return COLORES[nombre]
    return TONOS[nombre]


def build_stylesheet() -> str:
    """Construye la hoja de estilo completa de la aplicación.

    Es una función pura: siempre devuelve lo mismo y no acumula texto, así que
    `apply_theme` puede llamarla las veces que haga falta.
    """
    c = _c
    r = RADIOS
    e = ESPACIOS
    return f"""
/* ---------- base ---------- */
QWidget {{
    background-color: {_c('fondo')};
    color: {_c('texto')};
    font-size: {TAMANO_BASE}pt;
}}
QToolTip {{
    background-color: {_c('marino')};
    color: {_c('blanco')};
    border: 1px solid {_c('azul_osc')};
    border-radius: {r['sm']}px;
    padding: {e['xs']}px {e['sm']}px;
}}

/* ---------- botones ---------- */
QPushButton {{
    background-color: {_c('papel')};
    color: {_c('azul_med')};
    border: 1px solid {_c('borde')};
    border-radius: {r['md']}px;
    padding: {e['xs']}px {e['lg']}px;
    min-height: 22px;
}}
QPushButton:hover  {{ background-color: {_c('hover')}; border-color: {_c('azul_clar')}; }}
QPushButton:pressed{{ background-color: {_c('seleccion')}; }}
QPushButton:disabled {{
    color: {_c('suave')};
    background-color: {_c('inactivo')};
    border-color: {_c('borde')};
}}
QPushButton[clase="secundario"] {{
    background-color: {_c('papel')};
    color: {_c('azul_med')};
    border: 1px solid {_c('borde')};
}}
QPushButton[clase="secundario"]:hover {{
    background-color: {_c('hover')};
    border-color: {_c('azul_clar')};
}}
QPushButton[clase="primario"] {{
    background-color: {_c('azul_med')};
    color: {_c('blanco')};
    border: 1px solid {_c('azul_med')};
    font-weight: 600;
}}
QPushButton[clase="primario"]:hover   {{ background-color: {_c('azul_osc')}; }}
QPushButton[clase="primario"]:pressed {{ background-color: {_c('marino')}; }}
QPushButton[clase="primario"]:disabled {{
    background-color: {_c('inactivo_primario')};
    border-color: {_c('inactivo_primario')};
    color: {_c('inactivo_texto')};
}}
QPushButton[clase="peligro"] {{
    background-color: {_c('papel')};
    color: {_c('acento')};
    border: 1px solid {_c('acento')};
}}
QPushButton[clase="peligro"]:hover {{ background-color: {_c('acento')}; color: {_c('blanco')}; }}

/* ---------- grupos ---------- */
QGroupBox {{
    background-color: {_c('papel')};
    border: 1px solid {_c('borde')};
    border-radius: {r['md']}px;
    margin-top: {e['md']}px;
    padding: {e['md']}px {e['md']}px {e['sm']}px {e['sm']}px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: {e['md']}px;
    padding: 0 {e['sm']}px;
    color: {_c('azul_med')};
}}

/* ---------- entradas ---------- */
QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {_c('papel')};
    color: {_c('texto')};
    border: 1px solid {_c('borde')};
    border-radius: {r['sm']}px;
    padding: 3px {e['sm']}px;
    selection-background-color: {_c('azul_med')};
    selection-color: {_c('blanco')};
}}
QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus,
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border: 1px solid {_c('azul_clar')};
}}
QLineEdit:disabled, QTextEdit:disabled, QPlainTextEdit:disabled,
QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
    background-color: {_c('inactivo')};
    color: {_c('suave')};
}}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
    background-color: {_c('papel')};
    color: {_c('texto')};
    border: 1px solid {_c('borde')};
    selection-background-color: {_c('azul_med')};
    selection-color: {_c('blanco')};
    outline: none;
}}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{ width: 14px; }}

/* ---------- casillas ---------- */
QCheckBox, QRadioButton {{ spacing: {e['sm']}px; background: transparent; }}
QCheckBox::indicator, QRadioButton::indicator {{ width: 15px; height: 15px; }}
QCheckBox::indicator {{
    border: 1px solid {_c('suave')};
    border-radius: {r['sm']}px;
    background-color: {_c('papel')};
}}
QCheckBox::indicator:checked {{
    background-color: {_c('azul_med')};
    border-color: {_c('azul_med')};
}}
QCheckBox::indicator:disabled {{ border-color: {_c('borde')}; background-color: {_c('inactivo')}; }}
QRadioButton::indicator {{
    border: 1px solid {_c('suave')};
    border-radius: 8px;
    background-color: {_c('papel')};
}}
QRadioButton::indicator:checked {{
    border: 5px solid {_c('azul_med')};
}}
QCheckBox:disabled, QRadioButton:disabled {{ color: {_c('suave')}; }}

/* ---------- tablas y listas ---------- */
QTableWidget, QTableView, QListWidget, QTreeWidget {{
    background-color: {_c('papel')};
    alternate-background-color: {_c('tabla_alterna')};
    border: 1px solid {_c('borde')};
    border-radius: {r['sm']}px;
    gridline-color: {_c('rejilla')};
    selection-background-color: {_c('seleccion')};
    selection-color: {_c('texto')};
}}
QTableWidget::item:selected, QListWidget::item:selected {{
    background-color: {_c('seleccion')};
    color: {_c('texto')};
}}
QHeaderView::section {{
    background-color: {_c('marino')};
    color: {_c('blanco')};
    border: none;
    border-right: 1px solid {_c('azul_osc')};
    padding: {e['xs']}px {e['sm']}px;
    font-weight: 600;
}}
QTableCornerButton::section {{ background-color: {_c('marino')}; border: none; }}

/* ---------- pestañas ---------- */
QTabWidget::pane {{
    border: 1px solid {_c('borde')};
    border-radius: {r['md']}px;
    background-color: {_c('papel')};
    top: -1px;
}}
QTabBar::tab {{
    background-color: {_c('inactivo_borde')};
    color: {_c('suave')};
    border: 1px solid {_c('borde')};
    border-bottom: none;
    border-top-left-radius: {r['sm']}px;
    border-top-right-radius: {r['sm']}px;
    padding: {e['sm']}px {e['lg']}px;
    margin-right: 2px;
}}
QTabBar::tab:selected {{
    background-color: {_c('papel')};
    color: {_c('azul_med')};
    font-weight: 600;
}}
QTabBar::tab:hover:!selected {{ background-color: {_c('tab_hover')}; }}

/* ---------- acordeón ---------- */
QToolBox::pane {{
    border: 1px solid {_c('borde')};
    border-radius: {r['md']}px;
    background-color: {_c('papel')};
    top: -1px;
}}
QToolBox QToolButton {{
    background-color: {_c('papel')};
    color: {_c('texto')};
    border: 1px solid {_c('borde')};
    border-radius: {r['sm']}px;
    padding: {e['sm']}px;
    font-weight: 600;
    text-align: left;
}}
QToolBox QToolButton:hover {{ background-color: {_c('hover')}; }}
QToolBox QToolButton:checked {{ background-color: {_c('seleccion')}; color: {_c('azul_med')}; }}

/* ---------- barras ---------- */
QScrollBar:vertical {{
    background: transparent;
    width: 12px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {_c('borde')};
    border-radius: 6px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: {_c('suave')}; }}
QScrollBar:horizontal {{ background: transparent; height: 12px; }}
QScrollBar::handle:horizontal {{
    background: {_c('borde')};
    border-radius: 6px;
    min-width: 30px;
}}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QProgressBar {{
    background-color: {_c('inactivo_borde')};
    border: 1px solid {_c('borde')};
    border-radius: {r['sm']}px;
    text-align: center;
    color: {_c('texto')};
    min-height: 16px;
}}
QProgressBar::chunk {{
    background-color: {_c('azul_clar')};
    border-radius: {r['sm'] - 1}px;
}}

/* ---------- etiquetas ---------- */
/* Sin esto cada etiqueta pinta su propio rectángulo y sobre un grupo blanco
   aparecen franjas grises donde solo hay texto. */
QLabel {{ background: transparent; }}
QLabel[role="titulo"]   {{ font-size: {TAMANO_BASE + 3}pt; font-weight: 600; color: {_c('marino')}; }}
QLabel[role="titulo"]   {{ font-size: {TAMANO_BASE + 3}pt; font-weight: 600; color: {_c('marino')}; }}
QLabel[role="seccion"]  {{ font-weight: 600; color: {_c('marino')}; }}
QLabel[role="aviso"]    {{ color: {_c('ambar')}; }}
QLabel[role="error"]    {{ color: {_c('acento')}; }}
QLabel[role="exito"]    {{ color: {_c('verde')}; }}
QLabel[role="suave"]    {{ color: {_c('suave')}; }}

/* ---------- barra lateral de pasos ---------- */
QFrame#lateral {{ background-color: {_c('marino')}; border: none; }}
QLabel#marca_titulo {{
    color: {_c('blanco')};
    font-size: {TAMANO_BASE + 2}pt;
    font-weight: 600;
    background-color: transparent;
}}
QLabel#marca_organizacion {{
    color: {_c('sidebar_texto')};
    font-size: {TAMANO_BASE - 1}pt;
    background-color: transparent;
}}
QLabel#paso_estado {{ color: {_c('sidebar_texto')}; background-color: transparent; }}
QPushButton#paso {{
    background-color: transparent;
    border: none;
    border-left: 3px solid transparent;
    border-radius: 0;
    color: {_c('sidebar_texto')};
    padding: {e['sm']}px {e['md']}px;
    text-align: left;
    font-weight: 600;
}}
QPushButton#paso:hover {{ background-color: {_c('sidebar_hover')}; color: {_c('blanco')}; }}
QPushButton#paso:checked {{
    background-color: {_c('azul_med')};
    border-left: 3px solid {_c('acento')};
    color: {_c('blanco')};
}}
/* Un paso completado se distingue por el texto («✓») y por el borde, no solo
   por el color: en escala de grises el usuario tiene que saber dónde está. */
QPushButton#paso[estado="completado"] {{ color: {_c('blanco')}; }}
QPushButton#paso[estado="completado"]:checked {{
    background-color: {_c('azul_osc')};
    border-left: 3px solid {_c('oro')};
}}

/* ---------- cabecera ---------- */
QFrame#cabecera {{
    background-color: {_c('papel')};
    border-bottom: 1px solid {_c('borde')};
}}
QLabel#cabecera_titulo {{ font-size: {TAMANO_BASE + 3}pt; font-weight: 600; color: {_c('marino')}; }}

/* ---------- estado y diálogos ---------- */
QStatusBar {{
    background-color: {_c('papel')};
    border-top: 1px solid {_c('borde')};
    color: {_c('suave')};
}}
QStatusBar::item {{ border: none; }}
QMessageBox, QDialog {{ background-color: {_c('fondo')}; }}
QMessageBox QLabel {{ color: {_c('texto')}; }}
QMessageBox QPushButton {{ min-width: 78px; }}

QScrollArea {{ border: none; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
"""


def aplicar_paleta_matplotlib() -> None:
    """Fija la paleta de matplotlib con los colores de la interfaz.

    Se llama una sola vez al arrancar para que **todas** las figuras nazcan ya
    con la paleta, sin tocar los sitios donde se crean.

    Vive en `ui/` porque `core/` no puede importar Qt ni depender de la capa
    visual (AGENTS §3.1): los gráficos son una responsabilidad de presentación.
    """
    import matplotlib as mpl
    import matplotlib.style as mpl_style

    mpl_style.use("default")  # que el estilo global del entorno no se cuele
    mpl.rcParams.update(
        {
            "figure.facecolor": _c("papel"),
            "figure.edgecolor": _c("borde"),
            "axes.facecolor": _c("papel"),
            "axes.edgecolor": _c("borde"),
            "axes.labelcolor": _c("texto"),
            "axes.titlecolor": _c("marino"),
            "axes.titlesize": TAMANO_BASE + 1,
            "axes.labelsize": TAMANO_BASE,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": _c("borde"),
            "grid.linewidth": 0.8,
            "grid.alpha": 0.7,
            "text.color": _c("texto"),
            "xtick.color": _c("suave"),
            "ytick.color": _c("suave"),
            "xtick.labelsize": TAMANO_BASE - 1,
            "ytick.labelsize": TAMANO_BASE - 1,
            "font.size": TAMANO_BASE,
            "font.family": "sans-serif",
            "font.sans-serif": FAMILIAS,
            "legend.frameon": False,
            "legend.fontsize": TAMANO_BASE - 1,
            "lines.linewidth": 1.8,
            "lines.markersize": 6,
            "figure.dpi": 100,
            "savefig.facecolor": _c("papel"),
        }
    )
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
        color=[_c("azul_med"), _c("acento"), _c("azul_clar"), _c("oro")]
    )


def apply_theme(app) -> None:
    """Aplica hoja de estilo, tipografía y paleta de gráficos.

    Idempotente: `setStyleSheet` reemplaza en vez de añadir, así que llamarla
    dos veces no duplica reglas. `main.py` la invoca antes de construir la
    ventana para que los widgets nazcan ya con estilo.
    """
    aplicar_fuente(app)
    app.setStyleSheet(build_stylesheet())
    aplicar_paleta_matplotlib()
