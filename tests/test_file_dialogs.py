"""Pruebas headless de la capa de diálogos y de la carga en las pestañas.

Requieren PySide6 y se ejecutan con `QT_QPA_PLATFORM=offscreen`; si PySide6 no
está instalado, el módulo se omite entero.
"""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pd = pytest.importorskip("pandas")

from core import data_loader  # noqa: E402
from core.state import AppState  # noqa: E402

PySide6 = pytest.importorskip("PySide6")

from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QFileDialog,
    QInputDialog,
    QMessageBox,
)

from ui import file_dialogs  # noqa: E402
from ui.data_tab import DataTab  # noqa: E402


@pytest.fixture(scope="module")
def app():
    instancia = QApplication.instance() or QApplication([])
    yield instancia


@pytest.fixture
def mensajes(app, monkeypatch):
    """Captura los diálogos modales para que ningún test se bloquee."""
    registros = []

    def registrar(categoria):
        def _falso(parent, titulo, texto, *args, **kwargs):
            registros.append((categoria, titulo, texto))
            return QMessageBox.Yes

        return _falso

    monkeypatch.setattr(QMessageBox, "critical", registrar("critical"))
    monkeypatch.setattr(QMessageBox, "warning", registrar("warning"))
    monkeypatch.setattr(QMessageBox, "information", registrar("information"))
    monkeypatch.setattr(QMessageBox, "question", registrar("question"))
    return registros


# ---------------------------------------------------------------------------
# file_dialogs
# ---------------------------------------------------------------------------


def test_table_filter_incluye_todos_los_formatos():
    for patron in ("*.csv", "*.tsv", "*.xlsx", "*.xlsm", "*.xls", "*.ods"):
        assert patron in file_dialogs.TABLE_FILTER


def test_choose_table_file_usa_el_filtro_compartido(monkeypatch):
    capturado = {}

    def falso(parent, caption, directorio, filtro):
        capturado.update(caption=caption, directorio=directorio, filtro=filtro)
        return "/tmp/datos.xlsx", ""

    monkeypatch.setattr(QFileDialog, "getOpenFileName", falso)
    ruta = file_dialogs.choose_table_file(None, "Elige algo")

    assert ruta == "/tmp/datos.xlsx"
    assert capturado["filtro"] == file_dialogs.TABLE_FILTER
    assert capturado["caption"] == "Elige algo"


def test_choose_table_file_cancelado_devuelve_none(monkeypatch):
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: ("", ""))
    assert file_dialogs.choose_table_file(None) is None


def test_choose_sheet_con_una_sola_hoja_no_pregunta(monkeypatch):
    def prohibido(*args, **kwargs):
        raise AssertionError("no debe preguntar con una sola hoja")

    monkeypatch.setattr(QInputDialog, "getItem", prohibido)
    assert file_dialogs.choose_sheet(None, ["datos"]) == "datos"


def test_choose_sheet_sin_hojas_devuelve_none():
    assert file_dialogs.choose_sheet(None, []) is None


def test_choose_sheet_ofrece_la_primera_por_defecto(monkeypatch):
    llamadas = {}

    def falso(parent, titulo, mensaje, hojas, indice, editable):
        llamadas.update(hojas=hojas, indice=indice)
        return "segunda", True

    monkeypatch.setattr(QInputDialog, "getItem", falso)
    elegido = file_dialogs.choose_sheet(None, ["primera", "segunda", "tercera"])

    assert elegido == "segunda"
    assert llamadas["hojas"] == ["primera", "segunda", "tercera"]
    assert llamadas["indice"] == 0


def test_choose_sheet_cancelado_devuelve_none(monkeypatch):
    monkeypatch.setattr(QInputDialog, "getItem", lambda *a, **k: ("primera", False))
    assert file_dialogs.choose_sheet(None, ["primera", "segunda"]) is None


def test_choose_save_path_normaliza_la_extension(monkeypatch):
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *a, **k: ("/tmp/salida", "Excel (*.xlsx)"),
    )
    ruta = file_dialogs.choose_save_path(
        None, "Exportar", "salida", "CSV (*.csv);;Excel (*.xlsx)"
    )
    assert ruta == "/tmp/salida.xlsx"


def test_choose_save_path_respeta_la_extension_escrita(monkeypatch):
    monkeypatch.setattr(
        QFileDialog,
        "getSaveFileName",
        lambda *a, **k: ("/tmp/salida.csv", "Excel (*.xlsx)"),
    )
    ruta = file_dialogs.choose_save_path(
        None, "Exportar", "salida", "CSV (*.csv);;Excel (*.xlsx)"
    )
    assert ruta == "/tmp/salida.csv"


def test_choose_save_path_cancelado_devuelve_none(monkeypatch):
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: ("", ""))
    assert file_dialogs.choose_save_path(None, "Exportar", "x", "CSV (*.csv)") is None


# ---------------------------------------------------------------------------
# DataTab
# ---------------------------------------------------------------------------


def _preparar(monkeypatch, ruta):
    monkeypatch.setattr(file_dialogs, "choose_table_file", lambda *a, **k: str(ruta))
    monkeypatch.setattr(
        file_dialogs,
        "choose_sheet",
        lambda parent, hojas, **k: hojas[0] if hojas else None,
    )


def test_data_tab_carga_un_xlsx(app, mensajes, monkeypatch, tmp_path):
    pytest.importorskip("openpyxl")
    ruta = tmp_path / "libro.xlsx"
    pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}).to_excel(
        ruta, engine="openpyxl", index=False
    )
    state = AppState()
    tab = DataTab(state)
    _preparar(monkeypatch, ruta)

    tab.load_data()

    assert state.raw_df is not None
    assert list(state.raw_df.columns) == ["a", "b"]
    assert state.source_path == str(ruta)
    assert "libro.xlsx" in tab.lbl_info.text()


def test_data_tab_carga_un_csv_cp1252(app, mensajes, monkeypatch, tmp_path):
    ruta = tmp_path / "excel_es.csv"
    ruta.write_bytes("a;b\n1;Añedo\n".encode("cp1252"))
    state = AppState()
    tab = DataTab(state)
    _preparar(monkeypatch, ruta)

    tab.load_data()

    assert state.raw_df["b"].tolist() == ["Añedo"]


def test_data_tab_cancela_sin_tocar_el_estado(app, mensajes, monkeypatch, tmp_path):
    pytest.importorskip("openpyxl")
    ruta = tmp_path / "libro.xlsx"
    with pd.ExcelWriter(ruta, engine="openpyxl") as escritor:
        pd.DataFrame({"a": [1]}).to_excel(escritor, sheet_name="primera", index=False)
        pd.DataFrame({"b": [2]}).to_excel(escritor, sheet_name="segunda", index=False)
    state = AppState()
    state.raw_df = pd.DataFrame({"previo": [1]})
    tab = DataTab(state)
    monkeypatch.setattr(file_dialogs, "choose_table_file", lambda *a, **k: str(ruta))
    monkeypatch.setattr(file_dialogs, "choose_sheet", lambda parent, hojas, **k: None)

    tab.load_data()

    assert list(state.raw_df.columns) == ["previo"]
    assert not [m for m in mensajes if m[0] == "critical"]


def test_data_tab_no_deja_nada_si_falla_la_carga(app, mensajes, monkeypatch, tmp_path):
    ruta = tmp_path / "basura.xyz"
    ruta.write_bytes(b"\x00\x01")
    state = AppState()
    tab = DataTab(state)
    _preparar(monkeypatch, ruta)

    state.raw_df = pd.DataFrame({"previo": [1]})

    tab.load_data()

    criticos = [texto for categoria, _, texto in mensajes if categoria == "critical"]
    assert criticos, "debe mostrar un QMessageBox"
    assert "Formato no admitido" in criticos[0]
    assert list(state.raw_df.columns) == ["previo"]


# ---------------------------------------------------------------------------
# SDD-006: exportar los datos de entrada junto con la predicción
# ---------------------------------------------------------------------------


@pytest.fixture
def bundle(tmp_path):
    from core import model_trainer, persistence

    dataset = pd.DataFrame(
        {
            "edad": [20, 30, 40, 50, 60, 70, 80, 90],
            "ciudad": ["a", "b"] * 4,
            "objetivo": [0, 1] * 4,
        }
    )
    pipe, metrics, _ = model_trainer.train_model(
        dataset,
        target="objetivo",
        model_name="Random Forest",
        task_type="classification",
    )
    export = model_trainer.transform_with_pipeline(dataset, pipe)
    metadata = persistence.build_metadata(
        pipe,
        target_column="objetivo",
        task_type="classification",
        model_name="Random Forest",
        metrics=metrics,
        df=export,
    )
    return persistence.load_bundle(
        persistence.save_bundle(tmp_path / "m.automl", pipe, metadata)
    )


@pytest.fixture
def predict_tab(app, bundle):
    from ui.predict_tab import PredictTab

    state = AppState()
    state.bundle = bundle
    tab = PredictTab(state)
    tab._input_df = pd.DataFrame(
        {"edad": [25, 45], "ciudad": ["a", "c"], "extra": ["n1", "n2"]}
    )
    return tab


def test_predict_calcula_las_columnas_nuevas(predict_tab):
    predict_tab.predict()

    assert predict_tab._predictions is not None
    assert list(predict_tab._preview.columns) == [
        persistence_column(predict_tab, 0),
        persistence_column(predict_tab, 1),
        persistence_column(predict_tab, 2),
    ]
    assert "extra" not in predict_tab._preview.columns
    assert "extra" in predict_tab._predictions.columns
    assert predict_tab.btn_export.isEnabled()


def persistence_column(tab, indice):
    return tab._preview.columns[indice]


def test_export_csv_incluye_entrada_y_prediccion(predict_tab, monkeypatch, tmp_path):
    from ui import predict_tab as modulo

    predict_tab.predict()
    destino = tmp_path / "salida.csv"
    monkeypatch.setattr(
        modulo.file_dialogs,
        "choose_save_path",
        lambda *a, **k: str(destino),
    )

    predict_tab.export_predictions()

    assert destino.exists()
    escrito = data_loader.load_table(str(destino))
    assert list(escrito.columns)[:3] == ["edad", "ciudad", "extra"]
    assert "prediccion" in escrito.columns
    assert len(escrito) == 2


def test_export_csv_usa_punto_y_coma_para_excel_espanol(
    predict_tab, monkeypatch, tmp_path
):
    from ui import predict_tab as modulo

    predict_tab.predict()
    destino = tmp_path / "salida.csv"
    monkeypatch.setattr(
        modulo.file_dialogs,
        "choose_save_path",
        lambda *a, **k: str(destino),
    )

    predict_tab.export_predictions()

    cabecera = destino.read_text(encoding="utf-8-sig").splitlines()[0]
    assert cabecera.count(";") == len(predict_tab._predictions.columns) - 1


def test_export_csv_conserva_los_acentos(predict_tab, monkeypatch, tmp_path):
    from ui import predict_tab as modulo

    entrada = pd.DataFrame({"edad": [25], "ciudad": ["Añedo"], "extra": ["ñ"]})
    predict_tab._input_df = entrada
    predict_tab.predict()
    destino = tmp_path / "acentos.csv"
    monkeypatch.setattr(
        modulo.file_dialogs,
        "choose_save_path",
        lambda *a, **k: str(destino),
    )

    predict_tab.export_predictions()

    escrito = data_loader.load_table(str(destino))
    assert escrito["ciudad"].tolist() == ["Añedo"]
    assert escrito["extra"].tolist() == ["ñ"]


def test_export_xlsx_escribe_una_hoja(predict_tab, monkeypatch, tmp_path):
    pytest.importorskip("openpyxl")
    from ui import predict_tab as modulo

    predict_tab.predict()
    destino = tmp_path / "salida.xlsx"
    monkeypatch.setattr(
        modulo.file_dialogs,
        "choose_save_path",
        lambda *a, **k: str(destino),
    )

    predict_tab.export_predictions()

    escrito = data_loader.load_table(str(destino))
    assert "prediccion" in escrito.columns
    assert len(escrito) == 2


def test_export_xlsx_avisa_si_falta_openpyxl(
    predict_tab, monkeypatch, tmp_path, mensajes
):
    from ui import predict_tab as modulo

    predict_tab.predict()
    destino = tmp_path / "salida.xlsx"
    monkeypatch.setattr(
        modulo.file_dialogs,
        "choose_save_path",
        lambda *a, **k: str(destino),
    )
    monkeypatch.setattr(modulo.importlib.util, "find_spec", lambda nombre: None)

    predict_tab.export_predictions()

    criticos = [texto for categoria, _, texto in mensajes if categoria == "critical"]
    assert criticos and "openpyxl" in criticos[0]
    assert not destino.exists()


def test_export_a_ruta_sin_extension_no_falla(
    predict_tab, monkeypatch, tmp_path, mensajes
):
    """Una ruta sin extensión se escribe igual, sin romper la app."""
    from ui import predict_tab as modulo

    predict_tab.predict()
    destino = tmp_path / "sin_extension"
    monkeypatch.setattr(
        modulo.file_dialogs,
        "choose_save_path",
        lambda *a, **k: str(destino),
    )

    predict_tab.export_predictions()

    assert destino.exists()
    assert not [m for m in mensajes if m[0] == "critical"]


def test_predict_muestra_las_columnas_nuevas_en_el_aviso(predict_tab):
    predict_tab.predict()

    texto = predict_tab.lbl_resultado.text()
    assert "prediccion" in texto
    assert "exportables juntas" in texto
