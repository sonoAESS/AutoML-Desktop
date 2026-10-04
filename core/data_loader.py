# core/data_loader.py
"""Carga de datos tabulares desde CSV y libros de Excel.

Este módulo solo hace entrada/salida: no conoce `scikit-learn` ni Qt.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd

from core.preprocessor import categorical_columns, numeric_columns

SEPARADORES = (",", ";", "\t", "|")
ENCODINGS = ("utf-8-sig", "cp1252", "latin-1")

# Extensión -> (motor de pandas, paquete que lo proporciona).
MOTORES_EXCEL = {
    ".xlsx": ("openpyxl", "openpyxl"),
    ".xlsm": ("openpyxl", "openpyxl"),
    ".xls": ("xlrd", "xlrd"),
    ".ods": ("odf", "odfpy"),
}

EXTENSIONES_TABULARES = tuple(MOTORES_EXCEL) + (".csv", ".txt", ".tsv")


class DataLoadError(ValueError):
    """Fichero que no se puede leer como tabla.

    Hereda de `ValueError` para que los `except ValueError` y
    `pytest.raises(ValueError)` existentes sigan siendo válidos.
    """


def extension_de(path: str | Path) -> str:
    """Extensión en minúsculas, con punto inicial (`""` si no la tiene)."""
    return Path(path).suffix.lower()


def _es_csv(ext: str) -> bool:
    return ext in (".csv", ".txt", ".tsv")


def detect_separator(header_line: str, preference: tuple = SEPARADORES) -> str | None:
    """Separador más frecuente en la cabecera, o `None` si no hay ninguno.

    Args:
        header_line: la cabecera del fichero.
        preference: orden de preferencia para desempatar. Por defecto es
            `SEPARADORES`; al exportar se usa `(";", ",", "\\t", "|")` porque
            Excel en español espera el punto y coma.

    Returns:
        El separador elegido, o `None` si la cabecera no contiene ninguno.
    """
    conteos = [(sep, header_line.count(sep)) for sep in preference]
    sep, n = max(conteos, key=lambda par: par[1])
    return sep if n > 0 else None


def _primera_linea(path: str | Path) -> str:
    """Primera línea física del fichero, decodificada como `latin-1`.

    `latin-1` no falla nunca y los bytes de los delimitadores son los mismos
    en todas las codificaciones, así que la detección es segura.
    """
    with open(path, "rb") as manejador:
        return manejador.readline().decode("latin-1")


def list_sheets(path: str) -> list[str]:
    """Nombres de las hojas del libro; `[]` para CSV.

    Args:
        path: ruta del fichero.

    Returns:
        Lista de nombres de hojas, vacía si el fichero es CSV.

    Raises:
        DataLoadError: si la extensión no es tabular o falta el motor.
    """
    ext = extension_de(path)
    if _es_csv(ext):
        return []
    motor, _paquete = _motor_para(ext)
    try:
        with pd.ExcelFile(path, engine=motor) as libro:
            return list(libro.sheet_names)
    except DataLoadError:
        raise
    except Exception as exc:
        raise DataLoadError(
            f"No se pudo abrir el libro «{path}» con {motor}: {exc}"
        ) from exc


def _motor_para(ext: str) -> tuple[str, str]:
    """Devuelve (motor, paquete) para una extensión de Excel o lanza error."""
    if ext not in MOTORES_EXCEL:
        raise DataLoadError(
            f"Formato no admitido: «{ext or 'sin extensión'}». "
            f"Se admiten: {', '.join(EXTENSIONES_TABULARES)}."
        )
    motor, paquete = MOTORES_EXCEL[ext]
    if importlib.util.find_spec(motor) is None:
        raise DataLoadError(
            f"Falta el paquete «{paquete}» para leer {ext}. "
            f"Instálalo con:  pip install {paquete}"
        )
    return motor, paquete


def _valida(df: pd.DataFrame) -> bool:
    """True si el DataFrame sirve como tabla de datos.

    Se acepta cualquier CSV con más de una columna. Para un CSV de una sola
    columna se exige que su nombre no contenga ningún separador candidato, para
    no confundirlo con uno mal detectado.
    """
    if df is None or df.shape[1] == 0:
        return False
    if df.shape[0] == 0:
        return False
    if df.shape[1] > 1:
        return True
    return not any(otro in str(df.columns[0]) for otro in SEPARADORES)


def load_csv_table(path: str | Path, sep: str | None = None) -> pd.DataFrame:
    """Lee un CSV con autodetección de separador y de codificación."""
    separadores: list[str] = []
    if sep is not None:
        separadores.append(sep)
    else:
        detectado = detect_separator(_primera_linea(path))
        if detectado is not None:
            separadores.append(detectado)
        separadores.extend(s for s in SEPARADORES if s not in separadores)

    intentos: list[str] = []
    for separador in separadores:
        for codificacion in ENCODINGS:
            intentos.append(f"sep={separador!r} encoding={codificacion}")
            try:
                df = pd.read_csv(path, sep=separador, encoding=codificacion)
            except Exception:
                continue
            if _valida(df):
                return df
    raise DataLoadError(
        f"No se pudo leer el archivo «{Path(path).name}» como CSV. "
        f"Se probaron {len(intentos)} combinaciones de separador y "
        f"codificación ({', '.join(ENCODINGS)} con separadores "
        f"{', '.join(repr(s) for s in separadores)}). "
        "Comprueba que el fichero no esté vacío y tenga una fila de cabecera."
    )


def load_excel(path: str | Path, sheet: str | int = 0) -> pd.DataFrame:
    """Lee una hoja de un libro de Excel u OpenDocument."""
    ext = extension_de(path)
    motor, paquete = _motor_para(ext)

    try:
        with pd.ExcelFile(path, engine=motor) as libro:
            hojas = list(libro.sheet_names)
    except Exception as exc:
        raise DataLoadError(
            f"No se pudo abrir «{Path(path).name}» con el paquete {paquete}: " f"{exc}"
        ) from exc

    clave = sheet
    if isinstance(clave, int):
        indice = clave if -len(hojas) <= clave < len(hojas) else 0
        clave = hojas[indice]
    if clave not in hojas:
        raise DataLoadError(
            f"La hoja «{sheet}» no existe en «{Path(path).name}». "
            f"Hojas disponibles: {', '.join(hojas)}."
        )

    try:
        df = pd.read_excel(path, sheet_name=clave, engine=motor, header=0)
    except Exception as exc:
        raise DataLoadError(
            f"No se pudo leer la hoja «{clave}» de «{Path(path).name}»: {exc}"
        ) from exc

    if df is None or df.shape[1] == 0:
        raise DataLoadError(
            f"La hoja «{clave}» de «{Path(path).name}» está vacía o no tiene "
            "cabecera."
        )
    if df.shape[0] == 0:
        raise DataLoadError(
            f"La hoja «{clave}» de «{Path(path).name}» no contiene filas de " "datos."
        )
    return df


def load_table(
    path: str | Path, sheet: str | int = 0, sep: str | None = None
) -> pd.DataFrame:
    """Lee una tabla de CSV, XLSX, XLS u ODS.

    Args:
        path: ruta del fichero. La extensión decide el formato.
        sheet: nombre o índice (empezando en 0) de la hoja. Se ignora en CSV.
        sep: separador explícito; `None` autodetecta.

    Returns:
        El `DataFrame` con el contenido literal del fichero.

    Raises:
        DataLoadError: si el formato no es tabular, falta el motor, el fichero
            está vacío o no se puede interpretar.
    """
    ruta = Path(path)
    ext = extension_de(ruta)
    if _es_csv(ext):
        return load_csv_table(ruta, sep=sep)
    if ext in MOTORES_EXCEL:
        return load_excel(ruta, sheet=sheet)
    if ext == ".xlsb":
        raise DataLoadError(
            "El formato .xlsb no es compatible. Ábrelo en Excel y vuelve a "
            "guardarlo como .xlsx."
        )
    raise DataLoadError(
        f"Formato no admitido: «{ext or 'sin extensión'}» "
        f"({ruta.name}). Se admiten: {', '.join(EXTENSIONES_TABULARES)}."
    )


def load_csv(path: str) -> pd.DataFrame:
    """Carga un CSV detectando automáticamente separador y encoding.

    Se acepta un archivo de una sola columna siempre que su nombre no
    contenga ningún separador candidato; así no se confunde un CSV de una
    columna con uno mal detectado.

    Wrapper retrocompatible de `load_table`.
    """
    return load_table(path)


def summarize(df: pd.DataFrame) -> dict:
    """Devuelve un resumen útil para mostrar en la UI."""
    return {
        "n_rows": len(df),
        "n_cols": df.shape[1],
        "missing": df.isna().sum().sum(),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "numeric_cols": numeric_columns(df),
        "categorical_cols": categorical_columns(df),
    }
