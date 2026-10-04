# tests/test_persistence.py
import numpy as np
import pandas as pd
import pytest

from core import model_trainer, persistence


@pytest.fixture
def dataset():
    return pd.DataFrame(
        {
            "edad": [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 22, 33],
            "ciudad": ["a", "b"] * 6,
            "alta": ["sí", "no"] * 6,
            "objetivo": [0, 1] * 6,
        }
    )


@pytest.fixture
def casts():
    return {"edad": "numerico", "alta": "booleano"}


@pytest.fixture
def normalizations():
    return {"edad": "minmax"}


@pytest.fixture
def trained(dataset, casts, normalizations):
    pipe, metrics, _ = model_trainer.train_model(
        dataset,
        target="objetivo",
        model_name="Random Forest",
        task_type="classification",
        casts=casts,
        normalizations=normalizations,
    )
    export = model_trainer.transform_with_pipeline(dataset, pipe)
    metadata = persistence.build_metadata(
        pipe,
        target_column="objetivo",
        task_type="classification",
        model_name="Random Forest",
        metrics=metrics,
        df=export,
        casts=casts,
        normalizations=normalizations,
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
        dataset,
        target="objetivo",
        model_name="Random Forest",
        task_type="classification",
        cv=2,
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


# ---------------------------------------------------------------------------
# SDD-006: exportar los datos de entrada junto con la predicción
# ---------------------------------------------------------------------------


def test_predict_frame_añade_la_prediccion_al_input(trained, dataset, tmp_path):
    pipe, metadata, export = trained
    bundle = persistence.load_bundle(
        persistence.save_bundle(tmp_path / "m.automl", pipe, metadata, dataset=export)
    )

    frame, report = persistence.predict_frame(bundle, dataset)

    entradas = list(dataset.columns)
    assert list(frame.columns)[: len(entradas)] == entradas
    assert persistence.PREDICTION_COLUMN in frame.columns
    assert {"prob_0", "prob_1"} <= set(frame.columns)
    assert report.ok is True
    assert report.renamed == ()
    assert len(frame) == len(dataset)


def test_predict_frame_conserva_el_indice_del_input(trained, dataset, tmp_path):
    pipe, metadata, export = trained
    bundle = persistence.load_bundle(
        persistence.save_bundle(tmp_path / "m.automl", pipe, metadata, dataset=export)
    )
    entrada = dataset.sample(frac=1.0, random_state=0)

    frame, _ = persistence.predict_frame(bundle, entrada)

    assert list(frame.index) == list(entrada.index)
    esperado, _, _, _ = persistence.predict(bundle, entrada)
    assert (
        frame[persistence.PREDICTION_COLUMN].tolist()
        == esperado[persistence.PREDICTION_COLUMN].tolist()
    )


def test_predict_frame_no_modifica_el_dataframe_recibido(trained, dataset, tmp_path):
    pipe, metadata, export = trained
    bundle = persistence.load_bundle(
        persistence.save_bundle(tmp_path / "m.automl", pipe, metadata, dataset=export)
    )
    entrada = dataset.copy()
    columnas_antes = list(entrada.columns)

    persistence.predict_frame(bundle, entrada)

    assert list(entrada.columns) == columnas_antes


def test_predict_frame_conserva_la_columna_objetivo(trained, dataset, tmp_path):
    pipe, metadata, export = trained
    bundle = persistence.load_bundle(
        persistence.save_bundle(tmp_path / "m.automl", pipe, metadata, dataset=export)
    )

    frame, _ = persistence.predict_frame(bundle, dataset)

    assert frame["objetivo"].tolist() == dataset["objetivo"].tolist()


def test_predict_frame_renombra_si_ya_existe_prediccion(trained, dataset, tmp_path):
    pipe, metadata, export = trained
    bundle = persistence.load_bundle(
        persistence.save_bundle(tmp_path / "m.automl", pipe, metadata, dataset=export)
    )
    entrada = dataset.assign(prediccion="mía")

    frame, report = persistence.predict_frame(bundle, entrada)

    assert frame["prediccion"].tolist() == ["mía"] * len(entrada)
    assert "prediccion_2" in frame.columns
    assert set(report.renamed) == {("prediccion", "prediccion_2")}
    assert "prediccion → prediccion_2" in report.describe()


def test_predict_frame_renombra_en_cascada(trained, dataset, tmp_path):
    pipe, metadata, export = trained
    bundle = persistence.load_bundle(
        persistence.save_bundle(tmp_path / "m.automl", pipe, metadata, dataset=export)
    )
    entrada = dataset.assign(
        **{
            persistence.PREDICTION_COLUMN: "a",
            f"{persistence.PREDICTION_COLUMN}_2": "b",
        }
    )

    frame, _ = persistence.predict_frame(bundle, entrada)

    assert f"{persistence.PREDICTION_COLUMN}_3" in frame.columns


def test_predict_sigue_devolviendo_solo_las_columnas_nuevas(trained, dataset, tmp_path):
    pipe, metadata, export = trained
    bundle = persistence.load_bundle(
        persistence.save_bundle(tmp_path / "m.automl", pipe, metadata, dataset=export)
    )

    salida, _, _, _ = persistence.predict(bundle, dataset)

    assert set(dataset.columns) & set(salida.columns) == set()
    assert persistence.PREDICTION_COLUMN in salida.columns


def test_predict_frame_es_idempotente(trained, dataset, tmp_path):
    """Aplicar el modelo a su propia salida no debe romper nada."""
    pipe, metadata, export = trained
    bundle = persistence.load_bundle(
        persistence.save_bundle(tmp_path / "m.automl", pipe, metadata, dataset=export)
    )

    una_vez, _ = persistence.predict_frame(bundle, dataset)
    otra_vez, report = persistence.predict_frame(bundle, una_vez)

    assert report.ok is True
    assert (
        otra_vez[f"{persistence.PREDICTION_COLUMN}_2"].tolist()
        == una_vez[persistence.PREDICTION_COLUMN].tolist()
    )


def test_predict_frame_falla_si_el_numero_de_filas_no_cuadra(
    trained, dataset, tmp_path, monkeypatch
):
    pipe, metadata, export = trained
    bundle = persistence.load_bundle(
        persistence.save_bundle(tmp_path / "m.automl", pipe, metadata, dataset=export)
    )
    monkeypatch.setattr(
        persistence,
        "align_features",
        lambda *a, **k: (dataset.head(3), None, persistence.SchemaReport()),
    )

    with pytest.raises(ValueError, match="predicciones"):
        persistence.predict_frame(bundle, dataset)


def test_free_names_genera_sufijos_libres():
    libres = persistence._free_names(
        ["a", "prediccion", "prediccion_2"],
        {"prediccion": [1], "prob_0": [0.5]},
    )
    assert libres == {"prediccion": "prediccion_3", "prob_0": "prob_0"}


def test_free_names_no_toca_los_demas_nombres():
    libres = persistence._free_names(["edad"], {"prediccion": [1]})
    assert libres["prediccion"] == "prediccion"


# ---------------------------------------------------------------------------
# SDD-004: balanceo en los metadatos y migración de bundles antiguos
# ---------------------------------------------------------------------------


def test_build_metadata_guarda_las_estadisticas_del_balanceo():
    rng = np.random.default_rng(5)
    dataset = pd.DataFrame(
        {
            "edad": rng.integers(20, 80, size=100),
            "objetivo": ["no"] * 80 + ["sí"] * 20,
        }
    )

    pipe, metrics, _ = model_trainer.train_model(
        dataset,
        target="objetivo",
        model_name="Regresión Logística",
        task_type="classification",
        balancing={"method": "random_under"},
    )
    export = model_trainer.transform_with_pipeline(dataset, pipe)
    metadata = persistence.build_metadata(
        pipe,
        target_column="objetivo",
        task_type="classification",
        model_name="Regresión Logística",
        metrics=metrics,
        df=export,
        balancing={"method": "random_under"},
    )

    assert metadata.balancing["method"] == "random_under"
    # El balancer solo ve el 80 % de entrenamiento, no el dataset completo.
    assert metadata.balancing["n_before"] < len(dataset)
    assert metadata.balancing["n_after"] < metadata.balancing["n_before"]
    assert "Balanceo: random_under" in metadata.describe()


def test_build_metadata_con_class_weight_no_remuestrea(dataset, casts):
    pipe, metrics, _ = model_trainer.train_model(
        dataset,
        target="objetivo",
        model_name="Regresión Logística",
        task_type="classification",
        casts=casts,
        balancing={"method": "class_weight"},
    )
    export = model_trainer.transform_with_pipeline(dataset, pipe)
    metadata = persistence.build_metadata(
        pipe,
        target_column="objetivo",
        task_type="classification",
        model_name="Regresión Logística",
        metrics=metrics,
        df=export,
        casts=casts,
        balancing={"method": "class_weight"},
    )

    assert metadata.balancing == {"method": "class_weight"}
    assert "pesos de clase" in metadata.describe()


def test_build_metadata_sin_balanceo_no_añade_nada(dataset, casts):
    pipe, metrics, _ = model_trainer.train_model(
        dataset,
        target="objetivo",
        model_name="Regresión Logística",
        task_type="classification",
        casts=casts,
    )
    export = model_trainer.transform_with_pipeline(dataset, pipe)
    metadata = persistence.build_metadata(
        pipe,
        target_column="objetivo",
        task_type="classification",
        model_name="Regresión Logística",
        metrics=metrics,
        df=export,
        casts=casts,
    )

    assert metadata.balancing == {}
    assert "Balanceo" not in metadata.describe()


def test_bundle_antiguamente_guardado_sin_balanceo_carga(trained, tmp_path):
    """Un `.automl` sin el campo `balancing` sigue cargando sin error."""
    pipe, metadata, export = trained
    viejo = persistence.BundleMetadata.__new__(persistence.BundleMetadata)
    viejo.__dict__.update(
        {k: v for k, v in metadata.__dict__.items() if k != "balancing"}
    )
    assert not hasattr(viejo, "balancing")

    ruta = persistence.save_bundle(tmp_path / "viejo.automl", pipe, viejo)
    cargado = persistence.load_bundle(ruta)

    assert cargado.metadata.balancing == {}
    assert "Balanceo" not in cargado.metadata.describe()


def test_balanceo_sobrevive_al_guardado_y_carga(dataset, casts, tmp_path):
    pipe, metrics, _ = model_trainer.train_model(
        dataset,
        target="objetivo",
        model_name="Regresión Logística",
        task_type="classification",
        casts=casts,
        balancing={"method": "random_under"},
    )
    export = model_trainer.transform_with_pipeline(dataset, pipe)
    metadata = persistence.build_metadata(
        pipe,
        target_column="objetivo",
        task_type="classification",
        model_name="Regresión Logística",
        metrics=metrics,
        df=export,
        casts=casts,
        balancing={"method": "random_under"},
    )
    ruta = persistence.save_bundle(tmp_path / "m.automl", pipe, metadata)
    cargado = persistence.load_bundle(ruta)

    assert cargado.metadata.balancing["method"] == "random_under"
    assert (
        cargado.metadata.balancing["n_after"] < cargado.metadata.balancing["n_before"]
    )
