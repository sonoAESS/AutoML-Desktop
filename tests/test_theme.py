"""Pruebas del sistema visual y del empaquetado de recursos (SDD-007)."""

import os
import re
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QPushButton  # noqa: E402

from ui import theme  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


# ----------------------------------------------------------------------
# C-001, C-002 · tokens y hoja de estilo
# ----------------------------------------------------------------------
def test_la_hoja_de_estilo_no_esta_vacia():
    qss = theme.build_stylesheet()
    assert qss.strip()
    assert len(qss) > 3000


def test_la_hoja_no_deja_tokens_sin_sustituir():
    """Qt no soporta `var(--x)`: si alguien lo escribe, quedaría literal."""
    qss = theme.build_stylesheet()
    assert "var(--" not in qss
    assert "@@TOKEN@@" not in qss


def test_ningun_color_de_la_hoja_esta_fuera_de_los_tokens():
    """C-002: la paleta se declara una sola vez, en COLORES y TONOS."""
    qss = theme.build_stylesheet()
    # Los selectores de Qt también llevan `#` (`QFrame#cabecera`), y no son
    # colores: se quitan antes de buscar colores.
    sin_selectores = re.sub(r"#\w+\s*[{,]", "{", qss)
    encontrados = {h.lower() for h in re.findall(r"#[0-9a-fA-F]{6}", sin_selectores)}
    permitidos = {
        v.lower() for v in list(theme.COLORES.values()) + list(theme.TONOS.values())
    }
    assert encontrados, "la hoja debería tener colores"
    assert encontrados <= permitidos, encontrados - permitidos


def test_los_tokens_de_la_paleta_estan_declarados():
    for clave in ("marino", "azul_med", "acento", "fondo", "papel", "texto", "blanco"):
        assert clave in theme.COLORES
    assert theme.COLORES["acento"] == "#d34223"  # rojo institucional
    assert theme.COLORES["azul_med"] == "#14448c"
    assert theme.COLORES["fondo"] == "#f3f4f7"


def test_la_paleta_no_mezcla_valores_invalidos():
    for nombre, valor in {**theme.COLORES, **theme.TONOS}.items():
        assert re.fullmatch(r"#[0-9a-fA-F]{6}", valor), f"{nombre}={valor}"


# ----------------------------------------------------------------------
# C-003 · idempotencia
# ----------------------------------------------------------------------
def test_apply_theme_es_idempotente(app):
    theme.apply_theme(app)
    primera = app.styleSheet()
    theme.apply_theme(app)
    assert app.styleSheet() == primera


def test_apply_theme_deja_la_aplicacion_con_estilo(app):
    theme.apply_theme(app)
    assert app.styleSheet().strip()
    assert app.font().pointSize() == theme.TAMANO_BASE


def test_la_fuente_pregunta_por_ubuntu_pero_acepta_cualquiera(app):
    theme.aplicar_fuente(app)
    familias = app.font().families()
    assert familias, "debe informarse al menos una familia"
    assert isinstance(familias, list)


# ----------------------------------------------------------------------
# C-004, C-005, C-006 · recursos
# ----------------------------------------------------------------------
def test_el_escudo_es_un_png_valido():
    from PySide6.QtGui import QImage

    ruta = theme.resource_path("escudo-mec.png")
    assert ruta.endswith(".png")
    imagen = QImage(ruta)
    assert not imagen.isNull()
    assert imagen.width() > 0 and imagen.height() > 0


def test_el_icono_es_un_svg_valido():
    ruta = theme.resource_path("icono-app.svg")
    contenido = Path(ruta).read_text(encoding="utf-8")
    assert contenido.lstrip().startswith("<svg")
    assert "</svg>" in contenido
    # Hereda la paleta institucional, que es lo que lo hace coherente.
    for color in ("#12223b", "#446dab", "#d34223"):
        assert color in contenido


def test_resource_path_funciona_en_modo_compilado(monkeypatch, tmp_path):
    """Con PyInstaller los recursos se extraen en `_MEIPASS`."""
    (tmp_path / "resources").mkdir()
    (tmp_path / "resources" / "icono-app.svg").write_text("<svg/>", encoding="utf-8")
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)

    assert theme.resource_path("icono-app.svg") == str(
        tmp_path / "resources" / "icono-app.svg"
    )


def test_resource_path_avisa_de_un_recurso_que_no_existe():
    with pytest.raises(FileNotFoundError) as exc:
        theme.resource_path("no-existe-este-fichero.png")
    # El mensaje tiene que decir qué falta, si no el error no ayuda.
    assert "no-existe-este-fichero.png" in str(exc.value)


def test_la_carpeta_de_recursos_solo_tiene_binarios():
    permitidos = {".png", ".svg", ".ico", ".jpg"}
    ficheros = [p for p in (RAIZ / "resources").iterdir() if p.is_file()]
    assert ficheros, "resources/ no debería estar vacía"
    for f in ficheros:
        assert f.suffix.lower() in permitidos, f"{f.name} no es un recurso gráfico"


def test_la_hoja_de_estilo_antigua_se_borro():
    """`resources/style.qss` estaba vacío y sin cargar: no debe volver."""
    assert not (RAIZ / "resources" / "style.qss").exists()


# ----------------------------------------------------------------------
# C-007, C-008 · icono y clases de botón
# ----------------------------------------------------------------------
def test_la_aplicacion_tiene_icono(app):
    theme.apply_theme(app)
    assert not theme.icono_app().isNull()


def test_la_ventana_tiene_icono(app):
    theme.apply_theme(app)
    from ui.main_window import MainWindow

    ventana = MainWindow()
    assert not ventana.windowIcon().isNull()
    assert ventana.windowTitle() == theme.NOMBRE_APP


def test_marcar_asigna_la_clase(app):
    theme.apply_theme(app)
    boton = QPushButton("Prueba")
    theme.marcar(boton, "primario")
    assert boton.property("clase") == "primario"


def test_marcar_cambia_el_aspecto_real(app):
    """Sin `unpolish/polish` la propiedad cambia pero el estilo no."""
    theme.apply_theme(app)
    boton = QPushButton("Prueba")
    boton.ensurePolished()
    pixmap_normal = boton.grab().toImage()
    theme.marcar(boton, "primario")
    pixmap_primario = boton.grab().toImage()

    assert pixmap_normal != pixmap_primario, "el botón debería verse distinto"


def test_los_botones_principales_estan_marcados(app):
    theme.apply_theme(app)
    from ui.main_window import MainWindow

    ventana = MainWindow()
    marcados = [
        b
        for b in ventana.findChildren(QPushButton)
        if b.property("clase") in {"primario", "peligro"}
    ]
    assert len(marcados) >= 4, "se esperaban los botones de acción principal"
    assert all(b.property("clase") for b in marcados)


def test_la_hoja_estiliza_las_tres_clases():
    qss = theme.build_stylesheet()
    for clase in ("primario", "secundario", "peligro"):
        assert f'QPushButton[clase="{clase}"]' in qss


# ----------------------------------------------------------------------
# C-009 · paleta de matplotlib
# ----------------------------------------------------------------------
def test_la_paleta_de_matplotlib_usa_los_tokens():
    import matplotlib as mpl

    theme.aplicar_paleta_matplotlib()
    assert mpl.rcParams["axes.prop_cycle"].by_key()["color"][0] == (
        theme.COLORES["azul_med"]
    )
    assert mpl.rcParams["axes.facecolor"].lower() == theme.COLORES["papel"].lower()
    assert mpl.rcParams["axes.grid"] is True


def test_las_figuras_nacen_con_la_paleta(app):
    """Una figura creada después del tema ya sale con los colores nuevos."""
    theme.apply_theme(app)
    from matplotlib.figure import Figure

    figura = Figure()
    eje = figura.add_subplot(111)
    eje.plot([0, 1], [0, 1])
    color = eje.lines[0].get_color()
    assert color.lower() == theme.COLORES["azul_med"].lower()


def test_la_paleta_no_toca_core():
    """`core/` no puede depender de la capa visual (AGENTS §3.1)."""
    for modulo in Path(RAIZ / "core").glob("*.py"):
        contenido = modulo.read_text(encoding="utf-8")
        assert "from ui" not in contenido, modulo.name
        assert "import ui" not in contenido, modulo.name


# ----------------------------------------------------------------------
# C-010 · empaquetado
# ----------------------------------------------------------------------
def test_el_spec_declara_los_recursos():
    spec = (RAIZ / "AutoML_MEC_Desktop.spec").read_text(encoding="utf-8")
    assert "datas=" in spec and "resources" in spec
    assert "icon=" in spec


def test_el_spec_esta_versionado():
    """El patrón `*.spec` de .gitignore lo dejaba fuera del repositorio."""
    import subprocess

    salida = subprocess.run(
        ["git", "ls-files", "AutoML_MEC_Desktop.spec"],
        cwd=RAIZ,
        capture_output=True,
        text=True,
    )
    assert salida.stdout.strip(), "el .spec debe estar versionado"


def test_las_rutas_del_spec_existen():
    spec = (RAIZ / "AutoML_MEC_Desktop.spec").read_text(encoding="utf-8")
    for ruta in re.findall(r"\(\s*[\"']([^\"']+)[\"']\s*,", spec):
        assert (RAIZ / ruta).exists(), ruta
