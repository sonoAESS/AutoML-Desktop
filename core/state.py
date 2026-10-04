# core/state.py
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from sklearn.pipeline import Pipeline


@dataclass
class AppState:
    """Datos compartidos entre las pestañas."""

    # --- Datos ---
    raw_df: Optional[pd.DataFrame] = None
    clean_df: Optional[pd.DataFrame] = None
    export_df: Optional[pd.DataFrame] = None
    source_path: Optional[str] = None
    profiles: list = field(default_factory=list)

    # --- Decisiones del usuario sobre los datos ---
    column_types: dict = field(default_factory=dict)
    normalizations: dict = field(default_factory=dict)

    # --- Modelo ---
    target_column: Optional[str] = None
    task_type: Optional[str] = None  # "classification" | "regression"
    model_name: Optional[str] = None
    family: Optional[str] = None
    pipeline: Optional[Pipeline] = None
    trained_model = None
    metrics: dict = field(default_factory=dict)
    _cv_results: dict = field(default_factory=dict)
    _test_data: Optional[tuple] = None

    # --- Balanceo de clases ---
    balancing: Optional[str] = None
    class_distribution: Optional[dict] = None

    # --- Persistencia ---
    bundle_path: Optional[str] = None
    bundle = None

    def reset_data(self):
        """Deja el estado como si no hubiera datos cargados."""
        self.raw_df = None
        self.clean_df = None
        self.export_df = None
        self.source_path = None
        self.profiles = []
        self.column_types = {}
        self.normalizations = {}
        self.target_column = None
        self.task_type = None
        self.model_name = None
        self.family = None
        self.pipeline = None
        self.trained_model = None
        self.metrics = {}
        self._cv_results = {}
        self._test_data = None
        self.balancing = None
        self.class_distribution = None
        self.bundle_path = None

    def reset_model(self):
        """Olvida el modelo entrenado, conservando los datos."""
        self.model_name = None
        self.pipeline = None
        self.trained_model = None
        self.metrics = {}
        self._cv_results = {}
        self._test_data = None
        self.bundle_path = None
