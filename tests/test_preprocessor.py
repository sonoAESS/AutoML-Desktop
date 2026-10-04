# tests/test_preprocessor.py
import numpy as np
import pandas as pd
import pytest

from core import preprocessor


def test_fill_missing_median():
    df = pd.DataFrame({"a": [1.0, None, 3.0]})
    out = preprocessor.fill_missing(df, numeric_strategy="median")
    assert out["a"].isna().sum() == 0
    assert out["a"].iloc[1] == 2.0


def test_cast_column_numerico_desde_texto():
    serie = pd.Series(["1", "2", "no", "4"])
    out = preprocessor.cast_column(serie, "numerico")
    assert out.iloc[:2].tolist() == [1.0, 2.0]
    assert pd.isna(out.iloc[2])


def test_cast_column_booleano_interpreta_texto():
    serie = pd.Series(["sí", "no", "1", None])
    out = preprocessor.cast_column(serie, "booleano")
    assert out.iloc[0] and not out.iloc[1] and out.iloc[2]
    assert pd.isna(out.iloc[3])


def test_cast_column_fecha():
    serie = pd.Series(["2024-01-15", "no-es-fecha"])
    out = preprocessor.cast_column(serie, "fecha")
    assert out.iloc[0] == pd.Timestamp("2024-01-15")
    assert pd.isna(out.iloc[1])


def test_cast_column_tipo_desconocido():
    with pytest.raises(ValueError):
        preprocessor.cast_column(pd.Series([1]), "inventado")


def test_cast_columns_no_toca_el_original():
    df = pd.DataFrame({"a": ["1", "2"]})
    out = preprocessor.cast_columns(df, {"a": "numerico"})
    assert out["a"].dtype.kind in "if"
    assert df["a"].tolist() == ["1", "2"]


def test_date_features():
    df = pd.DataFrame({"fecha": ["2024-03-04", "2023-12-31"]})
    out = preprocessor.date_features(df, ["fecha"])
    assert "fecha" not in out.columns
    assert out["fecha_year"].tolist() == [2024, 2023]
    assert out["fecha_month"].tolist() == [3, 12]


@pytest.mark.parametrize(
    "method, expected",
    [
        ("minmax", [0.0, 0.5, 1.0]),
        ("standard", [-1.224744871, 0.0, 1.224744871]),
        ("robust", [-1.0, 0.0, 1.0]),
    ],
)
def test_normalize_column(method, expected):
    serie = pd.Series([1.0, 2.0, 3.0])
    out = preprocessor.normalize_column(serie, method)
    assert out.tolist() == pytest.approx(expected)


def test_normalize_column_log():
    out = preprocessor.normalize_column(pd.Series([0.0, np.e - 1.0]), "log")
    assert out.iloc[0] == 0.0
    assert out.iloc[1] == pytest.approx(1.0)


def test_normalize_column_none_no_cambia():
    serie = pd.Series([1.0, 5.0])
    assert preprocessor.normalize_column(serie, "none").tolist() == [1.0, 5.0]


def test_normalize_column_valor_constante_no_divide_por_cero():
    out = preprocessor.normalize_column(pd.Series([7.0, 7.0]), "standard")
    assert out.tolist() == [0.0, 0.0]


def test_normalization_plan_agrupa_y_descarta_none():
    plan = preprocessor.normalization_plan(
        {"a": "standard", "b": "none", "c": None, "d": "standard"}
    )
    assert plan == {"standard": ["a", "d"]}


def test_normalize_columns_solo_numericas():
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]})
    out = preprocessor.normalize_columns(df, ["a", "b"], "minmax")
    assert out["a"].tolist() == [0.0, 0.5, 1.0]
    assert out["b"].tolist() == ["x", "y", "z"]


def test_column_typer_es_un_transformador():
    typer = preprocessor.ColumnTyper({"a": "numerico"})
    X = pd.DataFrame({"a": ["1", "2"]})
    out = typer.fit(X).transform(X)
    assert out["a"].tolist() == [1.0, 2.0]


def test_column_normalizer_es_un_transformador():
    normalizer = preprocessor.ColumnNormalizer({"a": "minmax"})
    X = pd.DataFrame({"a": [0.0, 5.0, 10.0]})
    out = normalizer.fit(X).transform(X)
    assert out["a"].tolist() == [0.0, 0.5, 1.0]


def test_column_normalizer_reutiliza_las_constantes_de_entrenamiento():
    normalizer = preprocessor.ColumnNormalizer({"a": "minmax"}).fit(
        pd.DataFrame({"a": [0.0, 10.0]})
    )
    nuevo = normalizer.transform(pd.DataFrame({"a": [20.0]}))
    assert nuevo["a"].tolist() == [2.0]


def test_column_normalizer_sin_fit_normaliza_el_lote():
    normalizer = preprocessor.ColumnNormalizer({"a": "standard"})
    X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    out = normalizer.transform(X)
    assert out["a"].mean() == pytest.approx(0.0, abs=1e-9)


def test_transformers_son_picklables():
    import pickle
    datos = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
    for obj in (
        preprocessor.ColumnTyper({"a": "numerico"}),
        preprocessor.ColumnNormalizer({"a": "standard"}),
    ):
        copia = pickle.loads(pickle.dumps(obj))
        assert copia.fit(datos).transform(datos) is not None