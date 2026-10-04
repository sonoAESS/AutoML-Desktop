"""Comparativa de varios modelos con el test de Friedman.

Solo lógica pura, sin Qt: devuelve los datos del análisis (ranking, post-hoc y
diferencia crítica) y el dibujo del diagrama vive en `ui/`.

Dos ideas sostienen el módulo:

1. Los bloques deben estar **pareados**: los mismos folds se usan para todos
   los modelos. Si cada modelo tuviera folds propios, el ranking por bloques no
   sería comparable y el test carecería de validez.
2. "Mejor" es siempre "rango más bajo": las métricas de error se invierten antes
   de ordenar, así que el resto del código no tiene que recordar qué métrica es
   mejor que cuál.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold

from core import model_specs
from core.model_specs import CLASSIFICATION
from core.model_trainer import build_pipeline, roc_auc_value

#: Signo con el que se orienta cada métrica: `-1` porque menor es mejor.
ORIENTACION = {"rmse": -1, "mae": -1}

#: Mínimo de algoritmos para que el test de Friedman tenga sentido.
MIN_MODELOS = 3


@dataclass
class ModelComparison:
    """Resultado de comparar varios modelos sobre bloques pareados.

    Attributes:
        metric: Métrica con la que se puntuó cada bloque.
        task_type: `"classification"` o `"regression"`.
        scores: Puntuación de cada modelo en cada bloque (`modelo`,
            `repeticion`, `fold`, `puntuacion`).
        ranking: Un modelo por fila, ordenado de mejor a peor, con
            `puntuacion_media`, `desviacion`, `rango_medio` y `equivalente`.
        friedman_statistic: Estadístico χ² del test de Friedman.
        friedman_p: p-valor del test de Friedman.
        n_observaciones: Número de bloques (`b`) usados.
        posthoc: Comparación por pares con p-valor ajustado por Holm.
        critical_difference: Diferencia crítica de Nemenyi.
        alpha: Nivel de significación con el que se marcó la significación.
        mejor_modelo: Modelo con menor rango medio.
        excluidos: Modelos que no se pudieron puntuar y su motivo, en español.
        n_splits: Folds por repetición realmente usados.
        n_repeats: Repeticiones realmente usadas.
    """

    metric: str
    task_type: str
    scores: pd.DataFrame
    ranking: pd.DataFrame
    friedman_statistic: float
    friedman_p: float
    n_observaciones: int
    posthoc: pd.DataFrame
    critical_difference: float
    alpha: float
    mejor_modelo: str
    excluidos: Dict[str, str] = field(default_factory=dict)
    n_splits: int = 0
    n_repeats: int = 0

    def describe(self) -> str:
        """Resumen en español con el estadístico y la lectura del p-valor."""
        if self.friedman_p < self.alpha:
            lectura = (
                "las diferencias entre modelos son estadísticamente " "significativas"
            )
        else:
            lectura = (
                "no hay diferencias estadísticamente significativas entre "
                "los modelos"
            )
        lineas = [
            f"Friedman χ²={self.friedman_statistic:.3f}, "
            f"p={self.friedman_p:.4f} sobre {self.n_observaciones} bloques: "
            f"{lectura} (α={self.alpha}).",
            f"Diferencia crítica: {self.critical_difference:.3f}. "
            f"Mejor modelo: {self.mejor_modelo}.",
        ]
        if self.friedman_p >= self.alpha and not self.posthoc.empty:
            lineas.append(
                "El post-hoc solo es concluyente si el test de Friedman "
                "resulta significativo: aquí no lo es."
            )
        for modelo, motivo in self.excluidos.items():
            lineas.append(f"Excluido: {modelo} ({motivo}).")
        return "\n".join(lineas)


def metric_label(metric: str) -> str:
    """Nombre de la métrica para la interfaz."""
    from core import model_specs

    return model_specs.metric_label(metric)


def orientation(metric: str) -> int:
    """Signo con el que se orienta la métrica: `-1` si menor es mejor."""
    return ORIENTACION.get(metric, 1)


def _holms_pvalues(pvalues: Sequence[float]) -> np.ndarray:
    """Corrección de Holm para controlar la tasa de error familiar.

    Implementada aquí porque `statsmodels` no es una dependencia permitida: el
    ajuste es *step-down monotónico*, de forma que un p-valor mayor que el
    anterior ya ajustado no puede bajar.

    Args:
        pvalues: p-valores sin ajustar.

    Returns:
        Los mismos p-valores en el orden de entrada, ajustados y acotados a 1.
    """
    p = np.asarray(list(pvalues), dtype=float)
    n = len(p)
    if n == 0:
        return p

    orden = np.argsort(p, kind="stable")
    ajustados = np.empty(n, dtype=float)
    acumulado = 0.0
    for posicion, indice in enumerate(orden):
        candidato = (n - posicion) * p[indice]
        acumulado = max(acumulado, candidato)
        ajustados[indice] = min(acumulado, 1.0)
    return ajustados


def critical_difference(k: int, b: int, alpha: float = 0.05) -> float:
    """Diferencia crítica de Nemenyi para `k` modelos y `b` bloques.

    Usa la aproximación de Demšar: `CD = q / sqrt(k)` con
    `q = sqrt(2) * t.ppf(1 - alpha/2, df=b - 1)`.

    Args:
        k: Número de modelos comparados.
        b: Número de bloques.
        alpha: Nivel de significación.

    Returns:
        La diferencia crítica en la escala de rangos medios.
    """
    if k < 2 or b < 2:
        return 0.0
    q = np.sqrt(2) * stats.t.ppf(1 - alpha / 2, df=b - 1)
    return float(q / np.sqrt(k))


def _effective_n_splits(task_type: str, y, n_splits: int) -> int:
    """Reduce `n_splits` si algún fold se quedaría sin una clase."""
    if task_type != CLASSIFICATION:
        return max(2, min(n_splits, len(y)))

    conteo = pd.Series(y).value_counts()
    minoria = int(conteo.min()) if len(conteo) else 0
    if minoria < 2:
        raise ValueError(
            "La clase minoritaria tiene menos de 2 filas: no se puede validar "
            "con folds estratificados."
        )
    return max(2, min(n_splits, minoria, len(y)))


def _score_block(metric, pipe, X_test, y_test, task_type) -> float:
    """Puntuación de un modelo en un bloque, o `NaN` si no se puede."""
    try:
        y_pred = pipe.predict(X_test)
    except Exception:  # noqa: BLE001 - el modelo se excluye con motivo
        return float("nan")

    if metric == "accuracy":
        from sklearn.metrics import accuracy_score

        return float(accuracy_score(y_test, y_pred))
    if metric == "f1_macro":
        from sklearn.metrics import f1_score

        return float(f1_score(y_test, y_pred, average="macro"))
    if metric == "roc_auc":
        from core.model_trainer import _auc_scores

        try:
            scores, clases = _auc_scores(pipe, X_test)
            valor, _motivo = roc_auc_value(y_test, scores, clases)
        except Exception:  # noqa: BLE001 - p. ej. fold con una sola clase
            return float("nan")
        return float("nan") if valor is None else float(valor)

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    if metric == "rmse":
        return float(np.sqrt(mean_squared_error(y_test, y_pred)))
    if metric == "mae":
        return float(mean_absolute_error(y_test, y_pred))
    if metric == "r2":
        return float(r2_score(y_test, y_pred))
    raise ValueError(f"Métrica no soportada en la comparativa: {metric!r}.")


def _matriz_bloques(scores: pd.DataFrame, modelos: List[str]) -> np.ndarray:
    """Matriz `bloques × modelos` con los datos en el mismo orden."""
    pivot = scores.pivot_table(
        index=["repeticion", "fold"], columns="modelo", values="puntuacion"
    ).sort_index()
    return pivot[modelos].to_numpy(dtype=float)


def _ranking(
    matriz: np.ndarray,
    rangos: np.ndarray,
    modelos: List[str],
    cd: float,
) -> pd.DataFrame:
    """Ranking final: menor rango medio es mejor.

    Recibe la matriz **sin** orientar (para que la media y la desviación que ve
    el usuario sean las de su métrica) y los rangos ya calculados por el
    llamante, que son los mismos que usa el test de Friedman.
    """
    medias = matriz.mean(axis=0)
    filas = []
    for posicion, modelo in enumerate(modelos):
        columna = rangos[:, posicion]
        filas.append(
            {
                "modelo": modelo,
                "puntuacion_media": float(medias[posicion]),
                "desviacion": float(matriz[:, posicion].std(ddof=1)),
                "rango_medio": float(columna.mean()),
            }
        )
    tabla = pd.DataFrame(filas)
    tabla = tabla.sort_values(["rango_medio", "modelo"], kind="stable").reset_index(
        drop=True
    )

    mejor = float(tabla["rango_medio"].iloc[0])
    tabla["equivalente"] = (tabla["rango_medio"] - mejor) <= cd
    return tabla


def _posthoc(matriz: np.ndarray, modelos: List[str], alpha: float) -> pd.DataFrame:
    """Comparación por pares con Wilcoxon pareado y ajuste de Holm.

    Con menos de 6 bloques el test t pareado sustituye al de Wilcoxon, que
    necesita más observaciones, y queda anotado en la columna `test`.
    """
    n_bloques = matriz.shape[0]
    pares = []
    for i in range(len(modelos)):
        for j in range(i + 1, len(modelos)):
            diferencia = matriz[:, i] - matriz[:, j]
            if n_bloques < 6:
                test = "t pareado"
                pvalor = float(
                    stats.ttest_rel(
                        matriz[:, i], matriz[:, j], nan_policy="omit"
                    ).pvalue
                )
            else:
                test = "Wilcoxon"
                if np.allclose(diferencia, 0):
                    pvalor = 1.0
                else:
                    pvalor = float(stats.wilcoxon(diferencia).pvalue)
            pares.append(
                {
                    "modelo_a": modelos[i],
                    "modelo_b": modelos[j],
                    "diferencia_media": float(diferencia.mean()),
                    "p_valor": pvalor,
                    "test": test,
                    "n_bloques": n_bloques,
                }
            )

    tabla = pd.DataFrame(pares)
    if tabla.empty:
        tabla["p_ajustada"] = []
        tabla["significativo"] = []
        return tabla

    tabla["p_ajustada"] = _holms_pvalues(tabla["p_valor"])
    tabla["significativo"] = tabla["p_ajustada"] < alpha
    return tabla


def compare_models(
    df,
    target: str,
    model_names: Sequence[str],
    task_type: str,
    metric: str,
    casts=None,
    normalizations=None,
    selection=None,
    balancing=None,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = 42,
    progress: Optional[Callable[[int, int], None]] = None,
) -> ModelComparison:
    """Compara varios modelos con el test de Friedman sobre bloques pareados.

    Cada modelo se ajusta con su pipeline en el fold de entrenamiento y se
    puntúa en el de prueba; ningún fold de prueba se usa para ajustar.

    Args:
        df: Dataset transformado (`state.export_df`).
        target: Columna objetivo.
        model_names: Modelos a comparar; hacen falta al menos tres.
        task_type: `"classification"` o `"regression"`.
        metric: Métrica de `available_metrics`.
        casts: Tipos forzados por columna.
        normalizations: Normalización por columna.
        selection: Selección de atributos.
        balancing: Estrategia de balanceo.
        n_splits: Folds por repetición.
        n_repeats: Repeticiones de la validación cruzada.
        random_state: Semilla de la partición.
        progress: Callback `progress(hecho, total)` para la interfaz.

    Returns:
        `ModelComparison` con las tablas y el resumen del análisis.

    Raises:
        ValueError: Menos de tres modelos, métrica no soportada o tan pocos
            datos que no se pueden construir folds estratificados.
    """
    metricas = set(model_specs.available_metrics(task_type))
    if metric not in metricas:
        raise ValueError(
            f"Métrica no soportada en la comparativa: {metric!r}. "
            f"Disponibles: {', '.join(sorted(metricas))}."
        )

    modelos = list(dict.fromkeys(model_names))
    if len(modelos) < MIN_MODELOS:
        raise ValueError(
            f"Hacen falta al menos {MIN_MODELOS} modelos para el test de "
            f"Friedman; se han pedido {len(modelos)}."
        )

    X = df.drop(columns=[target])
    y = df[target]
    n_splits = _effective_n_splits(task_type, y, n_splits)

    if task_type == CLASSIFICATION:
        cv = RepeatedStratifiedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )
    else:
        cv = RepeatedKFold(
            n_splits=n_splits, n_repeats=n_repeats, random_state=random_state
        )

    # Los bloques se generan una sola vez y se reutilizan para todos los
    # modelos: es la condición de validez del test (R-002).
    bloques = list(cv.split(X, y))
    total = len(modelos) * len(bloques)
    avisar = progress or (lambda hecho, _total: None)
    hechos = 0

    pipelines = {
        nombre: build_pipeline(
            df,
            nombre,
            task_type,
            target,
            casts=casts,
            normalizations=normalizations,
            selection=selection,
            balancing=balancing,
        )
        for nombre in modelos
    }

    filas = []
    for nombre in modelos:
        for indice, (i_train, i_test) in enumerate(bloques):
            repeticion = indice // n_splits + 1
            fold = indice % n_splits + 1
            pipe = clone(pipelines[nombre])
            X_train, y_train = X.iloc[i_train], y.iloc[i_train]
            X_test, y_test = X.iloc[i_test], y.iloc[i_test]
            puntuacion = float("nan")
            try:
                pipe.fit(X_train, y_train)
                puntuacion = _score_block(metric, pipe, X_test, y_test, task_type)
            except Exception:  # noqa: BLE001 - el modelo se excluye
                # Un fold puede fallar (p. ej. una clase ausente en el
                # entrenamiento). No se aborta: la puntuación queda como NaN
                # y el modelo entero se excluye al final, con su motivo.
                puntuacion = float("nan")

            filas.append(
                {
                    "modelo": nombre,
                    "repeticion": repeticion,
                    "fold": fold,
                    "puntuacion": puntuacion,
                }
            )
            hechos += 1
            avisar(hechos, total)

    scores = pd.DataFrame(filas)

    # Un NaN en cualquier bloque invalida al modelo: la matriz de rangos tiene
    # que ser rectangular.
    excluido = scores.groupby("modelo")["puntuacion"].apply(
        lambda s: bool(s.isna().any())
    )
    excluidos: Dict[str, str] = {}
    supervivientes = []
    for nombre in modelos:
        if bool(excluido.get(nombre, False)):
            excluidos[nombre] = (
                "algún bloque dio una puntuación no calculable; el ranking "
                "exige los mismos bloques para todos los modelos"
            )
        else:
            supervivientes.append(nombre)

    if len(supervivientes) < MIN_MODELOS:
        raise ValueError(
            "Solo han sobrevivido "
            f"{len(supervivientes)} modelos de los {len(modelos)} pedidos "
            f"({len(excluidos)} excluidos). Hacen falta al menos "
            f"{MIN_MODELOS} para el test de Friedman."
        )

    matriz_real = _matriz_bloques(scores, supervivientes)
    # Orientación: que "mayor es mejor" sea universal al ordenar (R-011). Solo
    # se invierte para los rangos; las medias que se muestran son las reales.
    matriz = matriz_real * orientation(metric)

    rangos = stats.rankdata(-matriz, axis=0)
    if np.all(rangos == rangos[0]):
        # Todos los modelos empatan: el estadístico es 0/0 y SciPy devuelve
        # NaN. La respuesta honesta es "no hay diferencia entre ellos".
        friedman_statistic, friedman_p = 0.0, 1.0
    else:
        friedman_statistic, friedman_p = stats.friedmanchisquare(*rangos)

    cd = critical_difference(len(supervivientes), len(bloques))
    ranking = _ranking(matriz_real, rangos, supervivientes, cd)
    posthoc = _posthoc(matriz, supervivientes, alpha=0.05)

    mejor = str(ranking.iloc[0]["modelo"])
    return ModelComparison(
        metric=metric,
        task_type=task_type,
        scores=scores,
        ranking=ranking,
        friedman_statistic=float(friedman_statistic),
        friedman_p=float(friedman_p),
        n_observaciones=len(bloques),
        posthoc=posthoc,
        critical_difference=cd,
        alpha=0.05,
        mejor_modelo=mejor,
        excluidos=excluidos,
        n_splits=n_splits,
        n_repeats=n_repeats,
    )
