"""Prueba de humo del ejecutable construido.

Se salta sola si no hay binario, para que `pytest -q` siga valiendo en un
clon recién descargado. En el CI se ejecuta después de PyInstaller y es lo
único que comprueba el paquete real: el resto de tests miran el código.

Comprueba dos cosas:

1. **Estructura**: que dentro de `_internal/` estén los plugins que la
   aplicación necesita (`platforms`, `imageformats`) y que no se hayan colado
   los que el `.spec` elimina. Un recorte de más aquí no da error hasta que
   alguien abre la aplicación.
2. **Arranque**: que el proceso siga vivo diez segundos después de lanzarlo
   con la plataforma `offscreen` y que no haya escrito una traza.

La ruta se puede forzar con `AUTOML_BINARY`; si no, se busca el paquete en
`dist/`.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent


def _ruta_binario() -> Path:
    forzada = os.environ.get("AUTOML_BINARY")
    if forzada:
        return Path(forzada)
    nombre = (
        "AutoML_MEC_Desktop.exe" if sys.platform == "win32" else "AutoML_MEC_Desktop"
    )
    return RAIZ / "dist" / "AutoML_MEC_Desktop" / nombre


BINARIO = _ruta_binario()

pytestmark = pytest.mark.skipif(
    not BINARIO.is_file(), reason="no hay binario compilado en dist/"
)


def _interno() -> Path:
    return BINARIO.parent / "_internal"


def test_el_paquete_trae_los_plugins_necesarios():
    plugins = _interno() / "PySide6" / "Qt" / "plugins"
    assert plugins.is_dir(), f"no existe {plugins}"

    presentes = {p.name for p in plugins.iterdir()}
    for necesario in ("platforms", "imageformats", "iconengines"):
        assert necesario in presentes, f"falta el plugin «{necesario}»"

    # El icono de la app es un SVG: sin estos dos, la ventana sale sin icono.
    formatos = {p.name for p in (plugins / "imageformats").iterdir()}
    assert any("qsvg" in n for n in formatos), "falta imageformats/qsvg"

    # Recortes del .spec: si vuelven, es que el filtro dejó de aplicarse.
    for sobrante in ("qmltooling", "virtualkeyboard", "tls", "sqldrivers"):
        assert sobrante not in presentes, f"no se recortó el plugin «{sobrante}»"

    assert not list(BINARIO.parent.rglob("*qgtk3*")), "no se recortó el tema GTK"


def test_el_paquete_solo_trae_las_traducciones_en_espanol():
    traducciones = _interno() / "PySide6" / "Qt" / "translations"
    if not traducciones.is_dir():
        return
    ajenas = [p.name for p in traducciones.iterdir() if not p.name.endswith("_es.qm")]
    assert not ajenas, f"quedaron traducciones sin usar: {ajenas[:5]}"


def test_el_paquete_trae_los_dos_recursos():
    recursos = _interno() / "resources"
    for nombre in ("escudo-mec.png", "icono-app.svg"):
        assert (recursos / nombre).is_file(), f"falta resources/{nombre}"


def test_el_binario_arranca_y_no_muere():
    entorno = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    proceso = subprocess.Popen(
        [str(BINARIO)],
        env=entorno,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        # 15 s y no 10: en un runner de Windows en frío, con el antivirus
        # revisando 1500 ficheros la primera vez, el arranque se nota.
        time.sleep(15)
        vivo = proceso.poll() is None
    finally:
        if proceso.poll() is None:
            proceso.terminate()
            try:
                proceso.wait(timeout=15)
            except subprocess.TimeoutExpired:
                proceso.kill()
                proceso.wait(timeout=15)

    salida = ""
    try:
        restante_out, restante_err = proceso.communicate(timeout=5)
        salida = f"{restante_out or ''}{restante_err or ''}"
    except (subprocess.TimeoutExpired, ValueError):
        pass

    assert vivo, "el binario murió al arrancar. Salida:\n" + salida[-3000:]
    for sintoma in ("Traceback", "ModuleNotFoundError", "cannot open shared object"):
        assert (
            sintoma not in salida
        ), f"el binario arrancó con «{sintoma}»:\n{salida[-3000:]}"
