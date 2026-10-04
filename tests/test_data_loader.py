import pandas as pd
import pytest

from core import data_loader


def test_load_csv_detecta_separador_coma(tmp_path):
    ruta = tmp_path / "coma.csv"
    ruta.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    df = data_loader.load_csv(str(ruta))
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2


def test_load_csv_detecta_separador_punto_y_coma(tmp_path):
    ruta = tmp_path / "punto_y_coma.csv"
    ruta.write_text("a;b\n1;2\n3;4\n", encoding="utf-8")
    df = data_loader.load_csv(str(ruta))
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2


def test_load_csv_acepta_una_sola_columna(tmp_path):
    ruta = tmp_path / "una_columna.csv"
    ruta.write_text("edad\n20\n31\n45\n", encoding="utf-8")
    df = data_loader.load_csv(str(ruta))
    assert list(df.columns) == ["edad"]
    assert len(df) == 3


def test_load_csv_no_confunde_una_columna_con_separador(tmp_path):
    """Un CSV de una columna con ';' en el nombre se relee con ';'."""
    ruta = tmp_path / "raro.csv"
    ruta.write_text("a;b;c\n1;2;3\n", encoding="utf-8")
    df = data_loader.load_csv(str(ruta))
    assert list(df.columns) == ["a", "b", "c"]
    assert df.shape == (1, 3)


def test_load_csv_error_si_no_se_puede_leer(tmp_path):
    ruta = tmp_path / "vacio.csv"
    ruta.write_text("", encoding="utf-8")
    with pytest.raises(ValueError):
        data_loader.load_csv(str(ruta))


def test_summarize_incluye_columnas_numericas_y_categoricas():
    df = pd.DataFrame({
        "num": [1.0, 2.0, None],
        "cat": ["a", "b", "a"],
    })
    resumen = data_loader.summarize(df)
    assert resumen["n_rows"] == 3
    assert resumen["n_cols"] == 2
    assert resumen["missing"] == 1
    assert resumen["numeric_cols"] == ["num"]
    assert resumen["categorical_cols"] == ["cat"]