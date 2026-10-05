# 012 — Catálogo ampliado · Tareas

**Spec:** [spec.md](spec.md) · **Diseño:** [design.md](design.md)

---

## Fase 1 · Familias

- [x] T-001 · `FAMILIES`: añadir `naive_bayes` y `lda` con sus etiquetas en
      español, al final del diccionario.
- [x] T-002 · `MIN_ROWS`: añadir `naive_bayes: 10` y `lda: 20`.

## Fase 2 · Limpieza

- [x] T-003 · Eliminar `ModelSpec.supports_probability` y sus cinco
      asignaciones.
- [x] T-004 · Eliminar `RANKING_METRIC` y `metrics_for_ranking()`.
- [x] T-005 · Borrar el test de `metrics_for_ranking` en
      `tests/test_model_trainer.py`.

## Fase 3 · Param groups nuevos

- [x] T-006 · `_BAGGING_PARAMS`: `n_estimators` (10, 20, 50), `max_samples`
      (0.5, 0.8, 1.0), `max_features` (0.5, 0.8, 1.0).
- [x] T-007 · `_BOOST_PARAMS`: `n_estimators` (100, 200, 400),
      `learning_rate` (0.01, 0.05, 0.1, 0.3), `max_depth` (3, 5, 10),
      `subsample` (0.8, 1.0).
- [x] T-008 · `_ADABOOST_PARAMS`: `n_estimators` (50, 100, 200),
      `learning_rate` (0.01, 0.1, 0.5, 1.0).
- [x] T-009 · `_HIST_PARAMS`: `learning_rate` (0.05, 0.1, 0.3),
      `max_iter` (100, 200, 500), `max_depth` (None, 3, 5, 10),
      `min_samples_leaf` (20, 40), `l2_regularization` (0.0, 0.1, 1.0). Con los
      valores del diseño pasaban a 864 combinaciones al añadir `class_weight`.
- [x] T-010 · `_EXTRA_TREES_PARAMS`: reutiliza `_FOREST_PARAMS` y añade
      `class_weight` (None, "balanced") solo en clasificación.
- [x] T-011 · `_LINEAR_REG_PARAMS`: `alpha` (0.001, 0.01, 0.1, 1.0),
      `fit_intercept` (True, False).

## Fase 4 · Clasificadores (12 nuevos)

- [x] T-012 · `SGD` (`lineal`, `class_weight_path`).
- [x] T-013 · `ExtraTrees` (`ensemble`, `class_weight_path`).
- [x] T-014 · `Bagging` (`ensemble`, sin `class_weight_path`).
- [x] T-015 · `AdaBoost` (`ensemble`, sin `class_weight_path`).
- [x] T-016 · `GradientBoosting` (`ensemble`, sin `class_weight_path`).
- [x] T-017 · `HistGradientBoosting` (`ensemble`, `class_weight_path`).
- [x] T-018 · `LinearSVC` (`svm`, `class_weight_path`, `max_iter=3000`).
- [x] T-019 · `RadiusNeighbors` (`vecinos`).
- [x] T-020 · `GaussianNB` (`naive_bayes`).
- [x] T-021 · `BernoulliNB` (`naive_bayes`).
- [x] T-022 · `LinearDiscriminant` (`lda`).
- [x] T-023 · `QuadraticDiscriminant` (`lda`).

## Fase 5 · Regresores (12 nuevos)

- [x] T-024 · `Lasso` (`lineal`).
- [x] T-025 · `ElasticNet` (`lineal`).
- [x] T-026 · `Huber` (`lineal`).
- [x] T-027 · `TheilSen` (`lineal`).
- [x] T-028 · `ExtraTrees` (`ensemble`).
- [x] T-029 · `Bagging` (`ensemble`).
- [x] T-030 · `AdaBoost` (`ensemble`), con `loss` (linear, square, exponential).
- [x] T-031 · `GradientBoosting` (`ensemble`).
- [x] T-032 · `HistGradientBoosting` (`ensemble`).
- [x] T-033 · `KernelRidge` (`svm`).
- [x] T-034 · `RadiusNeighbors` (`vecinos`).

## Fase 6 · Tests

- [x] T-035 · `test_model_specs.py`: recuentos 17 / 18 y 7 familias (C-001,
      C-002).
- [x] T-036 · `test_model_specs.py`: `class_weight_path` devuelve ruta para
      `HistGradientBoosting`, `ExtraTrees`, `SGD`, `LinearSVC` y `None` para
      el resto (C-003).
- [x] T-037 · `test_model_specs.py`: `MIN_ROWS` contiene las cuatro claves
      (C-004).
- [x] T-038 · `test_model_specs.py`: todo grid nuevo usa claves `model__*` y
      el producto no pasa de 500 (C-005).
- [x] T-039 · `test_model_specs.py`: `ModelSpec` sin `supports_probability` y
      sin `RANKING_METRIC` / `metrics_for_ranking` (C-006).
- [x] T-040 · `test_model_specs.py`: `build_estimator` devuelve instancia
      válida con `random_state=42` para cada modelo nuevo (C-007).
- [x] T-041 · `test_model_specs.py`: con 15 filas, `lda` es incompatible con su
      motivo (C-008).
- [x] T-042 · `test_model_trainer.py`: entrenar los 24 modelos nuevos de verdad
      sobre un dataset pequeño y comprobar que devuelve métricas (C-009).
- [x] T-047 · `test_model_specs.py`: barrer las 1620 combinaciones de todos los
      grids y comprobar que ninguna falla (marcado `slow`). Salió de la
      implementación: encontró los grids rotos de `LinearDiscriminant`,
      `KernelRidge` y `LinearSVC`.

## Fase 7 · Documentación

- [x] T-043 · `README.md`: tabla de modelos por familia y de familias nuevas.

## Fase 8 · Verificación

- [x] T-044 · `pytest -q -m "not slow"` en verde.
- [x] T-045 · `grep -R "PySide6\|PyQt\|from ui\." core/` vacío.
- [x] T-046 · `black --check` e `isort --check-only` en los ficheros tocados.