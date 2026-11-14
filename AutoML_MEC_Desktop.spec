# -*- mode: python ; coding: utf-8 -*-
"""Empaquetado de AutoML Desktop en modo carpeta (`onedir`).

Se usa `onedir` y no `onefile` a propósito. Un único archivo
autodescomprimible obligaba a extraer ~477 MB a un directorio temporal en cada
arranque, y eso se notaba: varios segundos antes de ver la ventana. En modo
carpeta no hay nada que extraer; el precio es que se distribuye una carpeta en
lugar de un fichero, y que `sys._MEIPASS` apunta a `_internal/` (lo que
`ui.theme.resource_path` ya contempla).

Además se recorta lo que la aplicación no usa. El paquete llegó a llevar dos
ICU completos: Qt trae el 73 (38 MB) y el plugin de tema nativo de GTK
arrastraba el 78 del sistema (35 MB) a través de `libxml2`. Ese plugin solo
cambia el aspecto de los diálogos, así que se va con toda su cadena.
"""

import os
import subprocess

# --- Qué sobra -----------------------------------------------------------
# Se compara contra el nombre de fichero, así que sirve tanto para
# `libQt6Qml.so.6` (Linux) como para `Qt6Qml.dll` (Windows).
LIBRERIAS_QUE_SOBRAN = tuple("""
    libQt6Qml libQt6Quick libQt6Pdf libQt6Network libQt6VirtualKeyboard
    libQt6OpenGL libQt6EglFS libQt6EglFsKmsSupport
    Qt6Qml Qt6Quick Qt6Pdf Qt6Network Qt6VirtualKeyboard
    Qt6OpenGL Qt6EglFS Qt6EglFsKmsSupport
    """.split())

#: Carpetas de plugins que nadie carga. Van enteras porque cada una arrastra
#: sus propias dependencias.
PLUGINS_QUE_SOBRAN = tuple("""
    qmltooling virtualkeyboard tls sqldrivers egldeviceintegrations
    xcbglintegrations sceneparsers renderers assetimporters geometryloaders
    """.split())

#: Plugin de tema GTK: 0,2 MB que arrastran 80 librerías y 80 MB.
PLUGIN_GTK = "libqgtk3"

#: Se conservan las traducciones en español (176 KB) por si algún día se
#: instala un `QTranslator`; el resto son 6,5 MB que nadie lee, porque hoy no
#: se carga ningún traductor.
SUFIJO_TRADUCCION = "_es.qm"

#: PyInstaller usa `/` en las rutas de destino, pero en Windows pueden venir
#: con barra invertida. Se normalizan antes de comparar.
BARRA_INVERTIDA = chr(92)


def _ruta(destino):
    """Ruta de destino con separadores `/`, venga de donde venga."""
    return destino.replace(BARRA_INVERTIDA, "/")


def _es_binario(ruta):
    """¿Es un ejecutable o una biblioteca, o solo un fichero de datos?"""
    try:
        with open(ruta, "rb") as fh:
            cabecera = fh.read(4)
    except OSError:
        return False
    return cabecera[:4] == b"\x7fELF" or cabecera[:2] == b"MZ"


def _dependencias(ruta):
    """Nombres de las bibliotecas que importa un binario."""
    if not _es_binario(ruta):
        return set()
    try:
        salida = subprocess.run(
            ["objdump", "-p", ruta],
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout
        deps = {
            linea.split()[1]
            for linea in salida.splitlines()
            if linea.strip().startswith("NEEDED")
        }
        if deps:
            return deps
    except Exception:
        pass
    # Windows no trae objdump; PyInstaller sí sabe leer las tablas PE.
    try:
        from PyInstaller.depend import bindepend

        return {os.path.basename(nombre) for nombre, _ in bindepend.get_imports(ruta)}
    except Exception:
        return set()


def _podar_binarios(entradas, sobra):
    """Quita `sobra` y lo que se quede sin nadie que lo use.

    Primero se marcan las entradas que sobran por nombre. Después se recorren
    sus dependencias y solo se elimina una biblioteca si *todas* las que la
    usaban han desaparecido. Así la cadena de GTK se va sola, sin mantener a
    mano una lista de cincuenta nombres que cambia en cada distribución.

    Una biblioteca con cero consumidores no se toca: puede que la cargue algo
    en tiempo de ejecución mediante `dlopen`, y esos enlaces no aparecen en la
    tabla `NEEDED`.
    """

    def base(destino):
        return os.path.basename(_ruta(destino))

    por_nombre = {base(e[0]): e for e in entradas}
    quitados = {n for n, e in por_nombre.items() if sobra(_ruta(e[0]))}
    if not quitados:
        return list(entradas)

    grafo = {
        n: {d for d in _dependencias(e[1]) if d in por_nombre}
        for n, e in por_nombre.items()
    }
    consumidores = {n: set() for n in por_nombre}
    for n, deps in grafo.items():
        for d in deps:
            consumidores[d].add(n)

    # Solo puede caer lo que colgaba de algo ya quitado.
    candidatos, pila = set(), list(quitados)
    while pila:
        for d in grafo.get(pila.pop(), ()):
            if d not in candidatos:
                candidatos.add(d)
                pila.append(d)

    cambio = True
    while cambio:
        cambio = False
        for n in candidatos - quitados:
            if consumidores[n] and consumidores[n] <= quitados:
                quitados.add(n)
                cambio = True

    return [e for e in entradas if base(e[0]) not in quitados]


def _sobra_binario(destino):
    nombre = os.path.basename(destino)
    if PLUGIN_GTK in nombre:
        return True
    if "/plugins/" in destino:
        return any(f"/{d}/" in destino for d in PLUGINS_QUE_SOBRAN)
    return nombre.startswith(LIBRERIAS_QUE_SOBRAN)


def _sobra_dato(destino):
    destino = _ruta(destino)
    if "/translations/" in destino:
        return not os.path.basename(destino).endswith(SUFIJO_TRADUCCION)
    return False


a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    # Solo los dos recursos que `ui.theme.resource_path` usa de verdad. El
    # `.ico` es para el icono del ejecutable y el `.png` su fuente: ninguno de
    # los dos hace falta dentro del paquete.
    datas=[
        ("resources/escudo-mec.png", "resources"),
        ("resources/icono-app.svg", "resources"),
    ],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

a.binaries = _podar_binarios(a.binaries, _sobra_binario)
a.datas = [e for e in a.datas if not _sobra_dato(e[0])]

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AutoML_MEC_Desktop",
    icon="resources/icono-app.ico",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    # UPX no está instalado en el CI, y cuando lo está, comprimir el
    # ejecutable provoca falsos positivos de antivirus.
    upx=False,
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="AutoML_MEC_Desktop",
)
