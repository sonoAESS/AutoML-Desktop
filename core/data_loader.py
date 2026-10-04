# core/data_loader.py
import pandas as pd

from core.preprocessor import categorical_columns, numeric_columns

SEPARADORES = (",", ";", "\t", "|")


def load_csv(path: str) -> pd.DataFrame:
    """Carga un CSV detectando automáticamente separador y encoding.

    Se acepta un archivo de una sola columna siempre que su nombre no
    contenga ningún separador 후보; así no se confunde un CSV de una
    columna con uno mal detectado.
    """
    for sep in SEPARADORES:
        try:
            df = pd.read_csv(path, sep=sep, encoding="utf-8")
        except Exception:
            continue
        if df.shape[1] > 1:
            return df
        if not any(otro in str(df.columns[0]) for otro in SEPARADORES):
            return df
    raise ValueError("No se pudo leer el archivo CSV correctamente.")

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