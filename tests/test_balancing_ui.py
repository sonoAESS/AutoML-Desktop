"""Pruebas headless de la interfaz de balanceo de clases (SDD-004)."""

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pd = pytest.importorskip("pandas")

from core.state import AppState  # noqa: E402

PySide6 = pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402


@pytest.fixture(scope="module")
def app():
    instancia = QApplication.instance() or QApplication([])
    yield instancia


@pytest.fixture
def mensajes(app, monkeypatch):
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


@pytest.fixture
def dataset():
    return pd.DataFrame(
        {
            "edad": [20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
            "objetivo": ["no"] * 8 + ["sí"] * 2,
        }
    )


def test_preprocess_tab_muestra_la_distribucion(app, mensajes, dataset):
    from ui.preprocess_tab import PreprocessTab

    state = AppState()
    state.clean_df = dataset
    state.target_column = "objetivo"
    state.task_type = "classification"
    tab = PreprocessTab(state)

    tab.refresh()

    assert not tab.gb_distribucion.isHidden()
    assert "2 clases" in tab.lbl_distribucion.text()
    assert "severo" in tab.lbl_distribucion.text()
    assert tab.tbl_distribucion.item(0, 0).text() == "no"
    assert tab.tbl_distribucion.item(0, 1).text() == "8"
    assert state.class_distribution["total"] == 10


def test_preprocess_tab_oculta_la_distribucion_en_regresion(app, mensajes, dataset):
    from ui.preprocess_tab import PreprocessTab

    state = AppState()
    state.clean_df = dataset
    state.target_column = "objetivo"
    state.task_type = "regression"
    tab = PreprocessTab(state)

    tab.refresh()

    assert tab.gb_distribucion.isHidden()


def test_preprocess_tab_sin_objetivo_oculta_la_distribucion(app, mensajes, dataset):
    from ui.preprocess_tab import PreprocessTab

    state = AppState()
    state.clean_df = dataset
    state.target_column = None
    state.task_type = "classification"
    tab = PreprocessTab(state)

    tab.refresh()

    assert tab.gb_distribucion.isHidden()


def test_train_tab_ofrece_las_estrategias(app, mensajes, dataset):
    from ui.train_tab import TrainTab

    state = AppState()
    state.clean_df = dataset
    state.target_column = "objetivo"
    state.task_type = "classification"
    tab = TrainTab(state)
    tab.refresh()

    assert not tab.gb_balanceo.isHidden()
    estrategias = [
        tab.cmb_balancing.itemData(i) for i in range(tab.cmb_balancing.count())
    ]
    assert estrategias == [
        "none",
        "class_weight",
        "random_under",
        "random_over",
        "smote",
        "smoten",
    ]


def test_train_tab_solo_ofrece_estrategias_compatibles(app, mensajes, dataset):
    """El desplegable nunca debe ofrecer más de lo que permite `balancing`."""
    from core import balancing
    from ui.train_tab import TrainTab

    state = AppState()
    state.clean_df = dataset
    state.target_column = "objetivo"
    state.task_type = "classification"
    tab = TrainTab(state)
    tab.refresh()

    for indice in range(tab.cmb_model.count()):
        model_name = tab.cmb_model.itemData(indice)
        tab.cmb_model.setCurrentIndex(indice)
        strategies = [
            tab.cmb_balancing.itemData(i) for i in range(tab.cmb_balancing.count())
        ]
        assert strategies == balancing.available_strategies(
            "classification", model_name
        )


def test_train_tab_config_cambia_segun_la_estrategia(app, mensajes, dataset):
    from ui.train_tab import TrainTab

    state = AppState()
    state.clean_df = dataset
    state.target_column = "objetivo"
    state.task_type = "classification"
    tab = TrainTab(state)
    tab.refresh()

    assert tab._balancing_config() is None

    tab.cmb_balancing.setCurrentIndex(2)
    assert tab._balancing_config() == {"method": "random_under"}

    tab.cmb_balancing.setCurrentIndex(4)
    config = tab._balancing_config()
    assert config["method"] == "smote"
    assert config["k_neighbors"] == 5
    assert tab.spn_vecinos.isEnabled()


def test_train_tab_emite_config_changed(app, mensajes, dataset):
    from ui.train_tab import TrainTab

    state = AppState()
    state.clean_df = dataset
    state.target_column = "objetivo"
    state.task_type = "classification"
    tab = TrainTab(state)
    tab.refresh()

    llamadas = []
    tab.config_changed.connect(lambda: llamadas.append(1))
    tab.cmb_balancing.setCurrentIndex(3)

    assert llamadas


def test_train_tab_en_regresion_oculta_el_balanceo(app, mensajes, dataset):
    from ui.train_tab import TrainTab

    state = AppState()
    state.clean_df = dataset
    state.target_column = "objetivo"
    tab = TrainTab(state)
    tab.cmb_task.setCurrentIndex(tab.cmb_task.findData("regression"))
    tab._on_task_changed()

    assert tab.gb_balanceo.isHidden()
    assert tab._balancing_config() is None


def test_train_tab_omite_el_peso_de_clases_manual(app, mensajes, dataset):
    """Con balanceo `class_weight` el control manual sobraría."""
    from ui.train_tab import TrainTab

    state = AppState()
    state.clean_df = dataset
    state.target_column = "objetivo"
    state.task_type = "classification"
    tab = TrainTab(state)
    tab.refresh()

    assert "class_weight" in tab._param_widgets

    indice = tab.cmb_balancing.findData("class_weight")
    tab.cmb_balancing.setCurrentIndex(indice)
    assert "class_weight" not in tab._param_widgets

    indice = tab.cmb_balancing.findData("random_under")
    tab.cmb_balancing.setCurrentIndex(indice)
    assert "class_weight" in tab._param_widgets


def test_results_tab_muestra_el_efecto_del_balanceo(app, mensajes):
    from core import model_trainer
    from ui.results_tab import ResultsTab

    dataset = pd.DataFrame(
        {
            "edad": [20, 30, 40, 50, 60, 70, 80, 90, 100, 110] * 3,
            "objetivo": ["no"] * 26 + ["sí"] * 4,
        }
    )
    state = AppState()
    state.clean_df = dataset
    state.target_column = "objetivo"
    state.task_type = "classification"
    pipe, metrics, _ = model_trainer.train_model(
        dataset,
        "objetivo",
        "Regresión Logística",
        "classification",
        balancing={"method": "random_under"},
    )
    state.pipeline = pipe
    state.metrics = metrics
    state.model_name = "Regresión Logística"
    state.balancing = "random_under"

    tab = ResultsTab(state)
    tab.refresh()

    assert "Balanceo: random_under" in tab.lbl.text()
    assert "instancias" in tab.lbl.text()
