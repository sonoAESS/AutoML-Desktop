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
    df = pd.DataFrame(
        {
            "num": [1.0, 2.0, None],
            "cat": ["a", "b", "a"],
        }
    )
    resumen = data_loader.summarize(df)
    assert resumen["n_rows"] == 3
    assert resumen["n_cols"] == 2
    assert resumen["missing"] == 1
    assert resumen["numeric_cols"] == ["num"]
    assert resumen["categorical_cols"] == ["cat"]


# ---------------------------------------------------------------------------
# SDD-001: carga multiformato (CSV + Excel)
# ---------------------------------------------------------------------------


def _escribir_excel(ruta, df, motor):
    df.to_excel(ruta, engine=motor, index=False)


def test_detect_separator_devuelve_el_mas_frecuente():
    assert data_loader.detect_separator("a;b;c") == ";"
    assert data_loader.detect_separator("a,b,c") == ","
    assert data_loader.detect_separator("a\tb\tc\t") == "\t"
    assert data_loader.detect_separator("a|b") == "|"


def test_detect_separator_sin_candidato_devuelve_none():
    assert data_loader.detect_separator("edad") is None


def test_detect_separator_empate_gana_el_primero_de_separadores():
    assert data_loader.detect_separator("a,b;c") == ","


def test_load_table_lee_xlsx(tmp_path):
    pytest.importorskip("openpyxl")
    ruta = tmp_path / "libro.xlsx"
    df = pd.DataFrame({"edad": [20, 31], "ciudad": ["A Coruña", "Vigo"]})
    _escribir_excel(ruta, df, "openpyxl")

    out = data_loader.load_table(str(ruta))

    assert list(out.columns) == ["edad", "ciudad"]
    assert out["edad"].tolist() == [20, 31]
    assert out["ciudad"].tolist() == ["A Coruña", "Vigo"]


def test_load_table_lee_xlsm(tmp_path):
    pytest.importorskip("openpyxl")
    ruta = tmp_path / "macro.xlsm"
    pd.DataFrame({"a": [1, 2]}).to_excel(ruta, engine="openpyxl", index=False)

    assert data_loader.load_table(str(ruta)).shape == (2, 1)


def test_load_table_lee_ods(tmp_path):
    pytest.importorskip("odf")
    ruta = tmp_path / "libro.ods"
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    _escribir_excel(ruta, df, "odf")

    out = data_loader.load_table(str(ruta))

    assert list(out.columns) == ["a", "b"]
    assert out["b"].tolist() == ["x", "y"]


def test_load_table_lee_xls(tmp_path):
    pytest.importorskip("xlwt")
    pytest.importorskip("xlrd")
    import xlwt

    ruta = tmp_path / "antiguo.xls"
    libro = xlwt.Workbook()
    hoja = libro.add_sheet("datos")
    for fila, valores in enumerate([["a", "b"], [1, 2], [3, 4]]):
        for columna, valor in enumerate(valores):
            hoja.write(fila, columna, valor)
    libro.save(str(ruta))

    out = data_loader.load_table(str(ruta))

    assert list(out.columns) == ["a", "b"]
    assert out.shape == (2, 2)


def test_load_table_elige_hoja_por_nombre(tmp_path):
    pytest.importorskip("openpyxl")
    ruta = tmp_path / "tres_hojas.xlsx"
    with pd.ExcelWriter(ruta, engine="openpyxl") as escritor:
        pd.DataFrame({"a": [1]}).to_excel(escritor, sheet_name="primera", index=False)
        pd.DataFrame({"b": [2, 3]}).to_excel(
            escritor, sheet_name="segunda", index=False
        )
        pd.DataFrame({"c": [4]}).to_excel(escritor, sheet_name="tercera", index=False)

    assert data_loader.load_table(str(ruta), sheet="segunda")["b"].tolist() == [2, 3]
    assert data_loader.load_table(str(ruta), sheet=0)["a"].tolist() == [1]


def test_load_table_hoja_inexistente_lista_las_disponibles(tmp_path):
    pytest.importorskip("openpyxl")
    ruta = tmp_path / "una_hoja.xlsx"
    pd.DataFrame({"a": [1]}).to_excel(ruta, engine="openpyxl", index=False)

    with pytest.raises(data_loader.DataLoadError, match="no existe"):
        data_loader.load_table(str(ruta), sheet="hoja_fantasma")


def test_load_table_hoja_sin_filas_da_error(tmp_path):
    pytest.importorskip("openpyxl")
    ruta = tmp_path / "solo_cabecera.xlsx"
    with pd.ExcelWriter(ruta, engine="openpyxl") as escritor:
        pd.DataFrame({"a": []}).to_excel(escritor, sheet_name="vacia", index=False)

    with pytest.raises(data_loader.DataLoadError, match="vacía|filas"):
        data_loader.load_table(str(ruta))


def test_list_sheets_devuelve_los_nombres(tmp_path):
    pytest.importorskip("openpyxl")
    ruta = tmp_path / "tres_hojas.xlsx"
    with pd.ExcelWriter(ruta, engine="openpyxl") as escritor:
        for nombre in ("primera", "segunda", "tercera"):
            pd.DataFrame({"a": [1]}).to_excel(escritor, sheet_name=nombre, index=False)

    assert data_loader.list_sheets(str(ruta)) == ["primera", "segunda", "tercera"]


def test_list_sheets_de_csv_devuelve_lista_vacia(tmp_path):
    ruta = tmp_path / "datos.csv"
    ruta.write_text("a,b\n1,2\n", encoding="utf-8")

    assert data_loader.list_sheets(str(ruta)) == []


def test_list_sheets_de_extension_no_tabular_falla(tmp_path):
    ruta = tmp_path / "informe.pdf"
    ruta.write_bytes(b"%PDF-1.4")

    with pytest.raises(data_loader.DataLoadError, match="Formato no admitido"):
        data_loader.list_sheets(str(ruta))


def test_load_table_csv_cp1252_con_acentos_y_punto_y_coma(tmp_path):
    ruta = tmp_path / "excel_es.csv"
    ruta.write_bytes("edad;ciudad\n30;Añedo\n40;Córdoba\n".encode("cp1252"))

    df = data_loader.load_table(str(ruta))

    assert list(df.columns) == ["edad", "ciudad"]
    assert df["ciudad"].tolist() == ["Añedo", "Córdoba"]


def test_load_table_csv_utf8_con_bom_no_deja_el_bom(tmp_path):
    ruta = tmp_path / "bom.csv"
    ruta.write_bytes("a,b\n1,2\n".encode("utf-8-sig"))

    df = data_loader.load_table(str(ruta))

    assert list(df.columns) == ["a", "b"]


def test_load_table_csv_tabulado(tmp_path):
    ruta = tmp_path / "tabulado.csv"
    ruta.write_text("a\tb\n1\t2\n3\t4\n", encoding="utf-8")

    assert list(data_loader.load_table(str(ruta)).columns) == ["a", "b"]


def test_load_table_csv_barra_vertical(tmp_path):
    ruta = tmp_path / "pipe.csv"
    ruta.write_text("a|b\n1|2\n3|4\n", encoding="utf-8")

    assert list(data_loader.load_table(str(ruta)).columns) == ["a", "b"]


def test_load_table_una_sola_columna_con_espacios_en_el_nombre(tmp_path):
    ruta = tmp_path / "una_columna.txt"
    ruta.write_text("edad media\n20\n31\n", encoding="utf-8")

    df = data_loader.load_table(str(ruta))

    assert list(df.columns) == ["edad media"]
    assert len(df) == 2


def test_load_table_txt_se_trata_como_csv(tmp_path):
    ruta = tmp_path / "exportacion.txt"
    ruta.write_text("a;b\n1;2\n", encoding="utf-8")

    assert list(data_loader.load_table(str(ruta)).columns) == ["a", "b"]


def test_load_table_acepta_separador_explicito(tmp_path):
    ruta = tmp_path / "raro.csv"
    ruta.write_text("a@b\n1@2\n", encoding="utf-8")

    df = data_loader.load_table(str(ruta), sep="@")

    assert list(df.columns) == ["a", "b"]
    assert df["b"].tolist() == [2]


def test_load_table_extension_no_tabular_es_valueerror(tmp_path):
    ruta = tmp_path / "modelo.bin"
    ruta.write_bytes(b"\x00\x01")

    with pytest.raises(ValueError, match="Formato no admitido"):
        data_loader.load_table(str(ruta))


def test_load_table_rechaza_xlsb(tmp_path):
    ruta = tmp_path / "binario.xlsb"
    ruta.write_bytes(b"PK\x03\x04")

    with pytest.raises(data_loader.DataLoadError, match="xlsb"):
        data_loader.load_table(str(ruta))


def test_load_table_fichero_vacio_da_error(tmp_path):
    ruta = tmp_path / "vacio.csv"
    ruta.write_text("", encoding="utf-8")

    with pytest.raises(data_loader.DataLoadError, match="No se pudo leer"):
        data_loader.load_table(str(ruta))


def test_load_table_solo_cabecera_da_error(tmp_path):
    ruta = tmp_path / "cabecera.csv"
    ruta.write_text("a,b\n", encoding="utf-8")

    with pytest.raises(data_loader.DataLoadError, match="No se pudo leer"):
        data_loader.load_table(str(ruta))


def test_load_table_extincion_en_mayusculas_funciona(tmp_path):
    pytest.importorskip("openpyxl")
    ruta = tmp_path / "LIBRO.XLSX"
    pd.DataFrame({"a": [1]}).to_excel(ruta, engine="openpyxl", index=False)

    assert data_loader.load_table(str(ruta)).shape == (1, 1)


def test_load_table_explica_el_paquete_que_falta(monkeypatch, tmp_path):
    ruta = tmp_path / "libro.xlsx"
    pd.DataFrame({"a": [1]}).to_excel(ruta, engine="openpyxl", index=False)
    monkeypatch.setattr(data_loader.importlib.util, "find_spec", lambda nombre: None)

    with pytest.raises(data_loader.DataLoadError, match="openpyxl"):
        data_loader.load_table(str(ruta))


def test_detect_separator_respeta_la_preferencia():
    cabecera = "a;b,c"
    assert data_loader.detect_separator(cabecera, preference=(";", ",")) == ";"
    assert data_loader.detect_separator(cabecera, preference=(",", ";")) == ","


def test_detect_separator_preferencia_sin_candidatos_devuelve_none():
    assert data_loader.detect_separator("edad", preference=(";", ",")) is None
