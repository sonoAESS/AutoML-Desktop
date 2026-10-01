# core/state.py
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
from sklearn.pipeline import Pipeline

@dataclass
class AppState:
    raw_df: Optional[pd.DataFrame] = None
    clean_df: Optional[pd.DataFrame] = None
    target_column: Optional[str] = None
    task_type: Optional[str] = None       # "classification" | "regression"
    pipeline: Optional[Pipeline] = None
    trained_model = None
    metrics: dict = field(default_factory=dict)