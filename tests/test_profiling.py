# tests/test_profiling.py
import numpy as np
import pandas as pd
import pytest

from core import profiling


def test_infer_tipos_basicos():
    assert profiling.infer_semantic_type(pd.Series([1.5, 2.5, 3.5])) == "numerico"
    assert profiling.infer_semantic_type(pd.Series([True, False])) == "booleano"
    assert profiling.infer_semantic_type(pd.Series(["a", "b", "a"])) == "categorico"
    fechas = pd.Series(pd.to_datetime(["2024-01-01", "2024-02-01"]))
    assert profiling.infer_semantic_type(fechas) == "fecha"


def test_identificador_por_valores_unicos():
    serie = pd.Series([f"id-{i}" for i in range(30)])
    assert profiling.infer_semantic_type(serie) == "identificador"

    enteros = pd.Series(range(30))
    assert profiling.infer_semantic_type(enteros) == "identificador"


def test_texto_libre_se_detecta():
    serie = pd.Series([f"frase número {i} con palabras" for i in range(80)])
    assert profiling.infer_semantic_type(serie) == "texto"


def test_suggest_task_type():
    assert profiling.suggest_task_type(pd.Series([0, 1, 0, 1])) == "classification"
    assert profiling.suggest_task_type(pd.Series(["si", "no", "si"])) == "classification"
    assert profiling.suggest_task_type(pd.Series([0.5, 1.5, 2.5, 9.75])) == "regression"
    assert profiling.suggest_task_type(pd.Series([f"id-{i}" for i in range(30)])) is None


def test_columna_convertible_a_numerico():
    valores = ["1", "2", "3", "no-es-un-número"] + [str(i) for i in range(4, 21)]
    valores.append("1")
    serie = pd.Series(valores)
    profile = profiling.profile_column(serie)
    assert profile.semantic_type == "categorico"
    assert profile.convertible_to_numeric is True
    assert any("números como texto" in w for w in profile.warnings)


def test_profile_column_calcula_nulos_y_cardinalidad():
    serie = pd.Series([1.0, 1.0, np.nan, 5.0])
    profile = profiling.profile_column(serie)
    assert profile.n_rows == 4
    assert profile.n_missing == 1
    assert profile.n_unique == 2
    assert profile.missing_pct == pytest.approx(0.25)
    assert profile.label == "Numérico"


def test_suggest_target_elige_menor_cardinalidad():
    df = pd.DataFrame({
        "edad": [20, 31, 44, 55, 61, 72, 18, 29, 40, 51],
        "ciudad": ["a", "b", "a", "c", "b", "a", "c", "b", "a", "c"],
        "id": [f"x{i}" for i in range(10)],
    })
    profiles = profiling.profile_dataframe(df)
    assert profiling.suggest_target(profiles).name == "ciudad"


def test_dataset_warnings_avisa_de_bloqueos():
    df = pd.DataFrame({
        "texto": [f"nota larga {i}" for i in range(80)],
        "valor": np.arange(80.0),
    })
    messages = profiling.dataset_warnings(profiling.profile_dataframe(df))
    assert any("no utilizables" in m for m in messages)


def test_columna_numerica_no_es_identificador_si_repite_valores():
    serie = pd.Series(np.arange(60.0) % 5)
    assert profiling.infer_semantic_type(serie) == "numerico"
    assert profiling.profile_column(serie).discrete is True

def test_profile_column_calcula_el_rango_de_valores():
    serie = pd.Series([3.0, 1.0, 7.0, None])
    profile = profiling.profile_column(serie, nombre="edad")
    assert profile.name == "edad"
    assert profile.value_min == 1.0
    assert profile.value_max == 7.0
    assert profile.value_range == "1 … 7"
    assert profile.n_missing == 1


def test_value_range_de_una_columna_sin_numeros():
    profile = profiling.profile_column(pd.Series(["a", "b", None]))
    assert profile.value_min is None
    assert profile.value_max is None
    assert profile.value_range == "—"


def test_dataset_warnings_distingue_texto_de_identificador():
    df = pd.DataFrame({
        "nota": [f"comentario largo de la fila {i}" for i in range(60)],
        "codigo": [f"c{i}" for i in range(60)],
    })
    profiles = profiling.profile_dataframe(df)
    texto = " ".join(profiling.dataset_warnings(profiles))
    assert "nota" in texto and "conviértela o elimínala" in texto
    assert "codigo" in texto and "conviene eliminarla" in texto
