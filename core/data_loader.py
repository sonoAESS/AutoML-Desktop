# core/data_loader.py
import pandas as pd

from core.preprocessor import categorical_columns, numeric_columns

def load_csv(path: str) -> pd.DataFrame:
    """Carga un CSV detectando automáticamente separador y encoding."""
    for sep in [",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(path, sep=sep, encoding="utf-8")
            if df.shape[1] > 1:
                return df
        except Exception:
            continue
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