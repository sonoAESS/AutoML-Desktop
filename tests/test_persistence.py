# tests/test_persistence.py
import pandas as pd
import pytest

from core import model_trainer, persistence


@pytest.fixture
def dataset():
    return pd.DataFrame({
        "edad": [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 22, 33],
        "ciudad": ["a", "b"] * 6,
        "alta": ["sí", "no"] * 6,
        "objetivo": [0, 1] * 6,
    })


@pytest.fixture
def casts():
    return {"edad": "numerico", "alta": "booleano"}


@pytest.fixture
def normalizations():
    return {"edad": "minmax"}


@pytest.fixture
def trained(dataset, casts, normalizations):
    pipe, metrics, _ = model_trainer.train_model(
        dataset, target="objetivo", model_name="Random Forest",
        task_type="classification", casts=casts, normalizations=normalizations,
    )
    export = model_trainer.transform_with_pipeline(dataset, pipe)
    metadata = persistence.build_metadata(
        pipe, target_column="objetivo", task_type="classification",
        model_name="Random Forest", metrics=metrics, df=export,
        casts=casts, normalizations=normalizations,
    )
    return pipe, metadata, export


def test_build_metadata_recoge_el_esquema(trained):
    _, metadata, _ = trained
    assert set(metadata.feature_columns) == {"edad", "ciudad", "alta"}
    assert metadata.target_column == "objetivo"
    assert metadata.task_type == "classification"
    assert metadata.model_name == "Random Forest"
    assert metadata.family == "ensemble"
    assert metadata.classes == (0, 1)
    assert metadata.casts == {"edad": "numerico", "alta": "booleano"}
    assert "accuracy" in metadata.metrics
    assert metadata.best_params == {}


def test_build_metadata_con_hiperparametros_afinados(dataset, casts):
    pipe, metrics, _, _ = model_trainer.tune_model(
        dataset, target="objetivo", model_name="Random Forest",
        task_type="classification", cv=2,
        selections={"n_estimators": 20},
    )
    metadata = persistence.build_metadata(
        pipe, "objetivo", "classification", "Random Forest", metrics
    )
    assert metadata.best_params["model__n_estimators"] == 20
    assert "best_params" not in metadata.metrics


def test_guardar_y_cargar_el_bundle(trained, tmp_path):
    pipe, metadata, export = trained
    ruta = persistence.save_bundle(tmp_path / "modelo", pipe, metadata, dataset=export)
    assert ruta.endswith(".automl")

    bundle = persistence.load_bundle(ruta)
    assert bundle.metadata.dataset_included is True
    assert bundle.dataset.shape == export.shape
    assert bundle.metadata.feature_columns == metadata.feature_columns


def test_guardar_sin_dataset(tmp_path, trained):
    pipe, metadata, _ = trained
    bundle = persistence.load_bundle(
        persistence.save_bundle(tmp_path / "m.automl", pipe, metadata)
    )
    assert bundle.dataset is None
    assert bundle.metadata.dataset_included is False


def test_cargar_archivo_que_no_es_bundle(tmp_path):
    ruta = tmp_path / "basura.automl"
    ruta.write_bytes(b"esto no es un bundle")
    with pytest.raises(Exception):
        persistence.load_bundle(ruta)


def test_cargar_archivo_inexistente(tmp_path):
    with pytest.raises(ValueError):
        persistence.load_bundle(tmp_path / "no_existe.automl")


def test_check_schema_detecta_faltantes_y_sobrantes(trained, dataset):
    _, metadata, _ = trained
    nuevo = dataset.drop(columns=["alta"]).assign(extra=1)
    report = persistence.check_schema(nuevo, metadata)
    assert report.ok is False
    assert "alta" in report.missing
    assert "extra" in report.unexpected


def test_align_features_convierte_tipos(trained, dataset):
    _, metadata, _ = trained
    nuevo = dataset.copy()
    nuevo["edad"] = ["55"] * len(nuevo)
    X, y, report = persistence.align_features(nuevo, metadata)
    assert report.ok is True
    assert X["edad"].dtype.kind == "f"
    assert list(X.columns) == list(metadata.feature_columns)


def test_align_features_falla_si_falta_una_columna(trained, dataset):
    _, metadata, _ = trained
    with pytest.raises(ValueError, match="alta"):
        persistence.align_features(dataset.drop(columns=["alta"]), metadata)


def test_align_features_no_estricto_sigue_adelante(trained, dataset):
    _, metadata, _ = trained
    X, _, report = persistence.align_features(
        dataset.drop(columns=["alta"]), metadata, strict=False
    )
    assert report.ok is False
    assert "alta" in X.columns and X["alta"].isna().all()


def test_predict_devuelve_clase_y_probabilidades(trained, dataset, tmp_path):
    pipe, metadata, export = trained
    bundle = persistence.load_bundle(
        persistence.save_bundle(tmp_path / "m.automl", pipe, metadata, dataset=export)
    )
    salida, report, X, _ = persistence.predict(bundle, dataset)
    assert report.ok is True
    assert persistence.PREDICTION_COLUMN in salida.columns
    assert {"prob_0", "prob_1"} <= set(salida.columns)
    assert len(salida) == len(dataset)


def test_predict_reutiliza_la_normalizacion_del_entrenamiento(trained, tmp_path):
    pipe, metadata, export = trained
    persistence.load_bundle(
        persistence.save_bundle(tmp_path / "m.automl", pipe, metadata)
    )
    paso = pipe.named_steps["normalizer"]
    assert paso.stats_["edad"]["center"] == 20.0
    assert paso.stats_["edad"]["scale"] == pytest.approx(70.0)

    nuevo = paso.transform(pd.DataFrame({"edad": [130.0]}))
    assert nuevo["edad"].tolist() == pytest.approx([(130.0 - 20.0) / 70.0])

    assert export["edad"].min() == pytest.approx(0.0)


def test_metadata_describe_en_castellano(trained):
    _, metadata, _ = trained
    texto = metadata.describe()
    assert "Random Forest" in texto
    assert "Objetivo: objetivo" in texto


def test_profile_input_solo_devuelve_las_columnas_del_modelo(trained, dataset):
    _, metadata, _ = trained
    nuevo = dataset.assign(extra=1)
    nombres = {p.name for p in persistence.profile_input(nuevo, metadata)}
    assert nombres == {"edad", "ciudad", "alta", "objetivo"}

def test_predict_no_normaliza_dos_veces(trained, tmp_path):
    """La normalización la aplica solo el pipeline, no `align_features`."""
    pipe, metadata, export = trained
    ruta = persistence.save_bundle(tmp_path / "m.automl", pipe, metadata)
    bundle = persistence.load_bundle(ruta)

    # una fila muy por encima del rango de entrenamiento
    nuevo = export.head(1).assign(edad=[9999.0])
    salida, _, _, _ = persistence.predict(bundle, nuevo)

    directa = pipe.predict(
        export.drop(columns=["objetivo"]).head(1).assign(edad=[9999.0])
    )
    assert salida["prediccion"].tolist() == pytest.approx(list(directa))


def test_align_features_no_aplica_normalizacion(trained, dataset):
    """`align_features` solo alinea columnas; deja transformar al pipeline."""
    _, metadata, _ = trained
    X, y, _ = persistence.align_features(dataset, metadata)
    assert X["edad"].tolist() == dataset["edad"].tolist()
    assert X["edad"].max() > 1.5  # sin normalizar min-max


def test_predict_sin_columnas_opcionales(trained, tmp_path):
    pipe, metadata, export = trained
    bundle = persistence.load_bundle(
        persistence.save_bundle(tmp_path / "m.automl", pipe, metadata)
    )
    salida, report, _, _ = persistence.predict(
        bundle, export[["edad", "objetivo"]], strict=False
    )
    assert len(salida) == len(export)
    assert "ciudad" in report.missing and "alta" in report.missing
