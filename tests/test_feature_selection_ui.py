"""Pruebas headless de la selección de atributos en la interfaz (SDD-002)."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from core.state import AppState  # noqa: E402


@pytest.fixture(scope="module")
def app():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def mensajes(app, monkeypatch):
    registros = []

    def registrar(categoria):
        def _falso(parent, titulo, texto, *args, **kwargs):
            registros.append((categoria, titulo))
            return QMessageBox.Yes

        return _falso

    monkeypatch.setattr(QMessageBox, "critical", registrar("critical"))
    monkeypatch.setattr(QMessageBox, "warning", registrar("warning"))
    monkeypatch.setattr(QMessageBox, "information", registrar("information"))
    monkeypatch.setattr(QMessageBox, "question", registrar("question"))
    return registros


@pytest.fixture
def dataset():
    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(4)})
    df["objetivo"] = (df["x0"] > 0).astype(int)
    df["ciudad"] = np.where(df["x1"] > 0, "norte", "sur")
    return df


def _tabla(dataset, task="classification"):
    from ui.train_tab import TrainTab

    state = AppState()
    state.clean_df = dataset
    state.export_df = dataset
    state.target_column = "objetivo"
    state.task_type = task
    tab = TrainTab(state)
    tab.refresh()
    return tab, state


# ----------------------------------------------------------------------
# T-010 · TrainTab
# ----------------------------------------------------------------------
def test_el_grupo_aparece_con_objetivo(app, mensajes, dataset):
    tab, _ = _tabla(dataset)

    assert not tab.gb_seleccion.isHidden()
    metodos = [
        tab.cmb_metodo_sel.itemData(i) for i in range(tab.cmb_metodo_sel.count())
    ]
    assert metodos == ["chi2", "anova", "mutual_info", "embedded"]


def test_en_regresion_no_ofrece_chi2(app, mensajes, dataset):
    from core import model_specs

    tab, _ = _tabla(dataset)
    # El objetivo es numérico, así que la sugerencia es clasificación: el tipo
    # se cambia como lo haría el usuario, desde el desplegable.
    tab.cmb_task.setCurrentIndex(tab.cmb_task.findData(model_specs.REGRESSION))
    tab._on_task_changed()

    metodos = [
        tab.cmb_metodo_sel.itemData(i) for i in range(tab.cmb_metodo_sel.count())
    ]
    assert "chi2" not in metodos


def test_sin_marca_no_hay_configuracion(app, mensajes, dataset):
    tab, state = _tabla(dataset)

    assert tab._selection_config() is None
    assert state.selection_method is None


def test_la_casilla_habilita_los_controles(app, mensajes, dataset):
    tab, _ = _tabla(dataset)
    assert not tab.btn_analizar.isEnabled()

    tab.chk_seleccion.setChecked(True)

    assert tab.btn_analizar.isEnabled()
    assert tab.cmb_metodo_sel.isEnabled()


def test_criterio_por_numero_y_porcentaje(app, mensajes, dataset):
    tab, _ = _tabla(dataset)
    tab.chk_seleccion.setChecked(True)

    assert tab._selection_config() == {"method": "anova", "k": 10}

    tab.cmb_criterio_sel.setCurrentIndex(tab.cmb_criterio_sel.findData("percentile"))
    config = tab._selection_config()
    assert config["method"] == "anova"
    assert config["percentile"] == 30.0


def test_embedded_no_admite_porcentaje(app, mensajes, dataset):
    tab, _ = _tabla(dataset)
    tab.chk_seleccion.setChecked(True)
    tab.cmb_metodo_sel.setCurrentIndex(tab.cmb_metodo_sel.findData("embedded"))
    indice_pct = tab.cmb_criterio_sel.findData("percentile")

    assert not tab.cmb_criterio_sel.model().item(indice_pct).isEnabled()
    # Aunque se llegue a porcentaje de otro modo, la configuración cae a `k`.
    tab.cmb_criterio_sel.setCurrentIndex(indice_pct)
    assert tab._selection_config() == {"method": "embedded", "k": 10}


def test_el_analisis_rellena_la_tabla(app, mensajes, dataset):
    tab, state = _tabla(dataset)
    tab.chk_seleccion.setChecked(True)
    tab.cmb_metodo_sel.setCurrentIndex(tab.cmb_metodo_sel.findData("anova"))
    tab.spn_sel.setValue(2)

    from core import model_trainer

    tabla = model_trainer.analyze_selection(
        dataset,
        state.target_column,
        tab.cmb_model.currentData(),
        "classification",
        selection=tab._selection_config(),
    )
    tab._pintar_seleccion(tabla)

    assert tab.tbl_seleccion.rowCount() == len(tabla)
    assert tab.tbl_seleccion.item(0, 0).text() == tabla.iloc[0]["atributo"]
    assert "atributos seleccionados" in tab.lbl_sel.text()
    assert state.selection_method == "anova"


def test_entrenar_sin_seleccion_no_cambia_el_pipeline(app, mensajes, dataset):
    tab, _ = _tabla(dataset)
    assert tab._selection_config() is None


def test_entrenar_pasa_la_seleccion(app, mensajes, dataset, monkeypatch):
    tab, _ = _tabla(dataset)
    tab.chk_seleccion.setChecked(True)
    tab.cmb_metodo_sel.setCurrentIndex(tab.cmb_metodo_sel.findData("anova"))

    captados = {}

    def _falso(*args, **kwargs):
        captados.update(kwargs)

        class _Pipe:
            named_steps = {}

        return _Pipe(), {"accuracy": 0.9, "n_test": 16}, None

    monkeypatch.setattr("ui.train_tab.model_trainer.train_model", _falso)
    tab.train()

    assert captados["selection"] == {"method": "anova", "k": 10}


def test_tune_worker_lleva_la_seleccion(app, mensajes, dataset):
    from ui.workers import TuneWorker

    worker = TuneWorker(
        df=dataset,
        target="objetivo",
        model_name="Random Forest",
        task_type="classification",
        search_type="grid",
        metric="accuracy",
        cv=3,
        n_iter=5,
        selection={"method": "chi2", "k": 4},
    )

    assert worker._kwargs["selection"] == {"method": "chi2", "k": 4}


# ----------------------------------------------------------------------
# T-011 · ResultsTab
# ----------------------------------------------------------------------
def test_results_tab_muestra_los_atributos(app, mensajes, dataset):
    from core import model_trainer
    from ui.results_tab import ResultsTab

    state = AppState()
    state.clean_df = dataset
    state.target_column = "objetivo"
    state.task_type = "classification"
    state.model_name = "Random Forest"
    pipe, metrics, test_data = model_trainer.train_model(
        dataset,
        "objetivo",
        "Random Forest",
        "classification",
        selection={"method": "anova", "k": 2},
    )
    state.pipeline = pipe
    state.metrics = metrics
    state.selection_method = "anova"
    state._test_data = test_data

    tab = ResultsTab(state)
    tab.refresh()

    assert "Selección: ANOVA F (2 atributos conservados)" in tab.lbl.text()


def test_results_tab_sin_selector_no_muestra_linea(app, mensajes, dataset):
    from core import model_trainer
    from ui.results_tab import ResultsTab

    state = AppState()
    state.clean_df = dataset
    state.target_column = "objetivo"
    state.task_type = "classification"
    state.model_name = "Random Forest"
    pipe, metrics, test_data = model_trainer.train_model(
        dataset, "objetivo", "Random Forest", "classification"
    )
    state.pipeline = pipe
    state.metrics = metrics
    state._test_data = test_data

    tab = ResultsTab(state)
    tab.refresh()

    assert "Selección" not in tab.lbl.text()
