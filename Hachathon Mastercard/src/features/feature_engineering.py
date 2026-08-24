"""FraudForge AI: Feature Engineering Pipeline.

Provides robust preprocessing, scaling, encoding, and transformer pipelines
for raw payment transactions.
"""

from typing import Tuple, List, Optional
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.utils.config import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMNS,
    PROCESSED_DATA_DIR,
)


class FraudFeaturePipeline(BaseEstimator, TransformerMixin):
    """Encapsulates feature extraction, scaling, and encoding for fraud detection."""

    def __init__(
        self,
        numerical_features: Optional[List[str]] = None,
        categorical_features: Optional[List[str]] = None,
    ):
        self.numerical_features = numerical_features or NUMERICAL_FEATURES
        self.categorical_features = categorical_features or CATEGORICAL_FEATURES
        self.preprocessor: Optional[ColumnTransformer] = None
        self._is_fitted: bool = False
        self._feature_names: List[str] = []

    def _build_preprocessor(self) -> ColumnTransformer:
        """Construct scikit-learn ColumnTransformer."""
        num_transformer = StandardScaler()
        cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

        return ColumnTransformer(
            transformers=[
                ("num", num_transformer, self.numerical_features),
                ("cat", cat_transformer, self.categorical_features),
            ],
            remainder="drop",
        )

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FraudFeaturePipeline":
        """Fit scaler on numerical features and encoder on categorical features."""
        self.preprocessor = self._build_preprocessor()
        self.preprocessor.fit(X[self.numerical_features + self.categorical_features])
        self._is_fitted = True

        # Extract output feature names
        num_names = self.numerical_features
        cat_encoder: OneHotEncoder = self.preprocessor.named_transformers_["cat"]
        cat_names = list(cat_encoder.get_feature_names_out(self.categorical_features))
        self._feature_names = list(num_names) + list(cat_names)

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform input DataFrame into a normalized feature matrix."""
        if not self._is_fitted or self.preprocessor is None:
            raise RuntimeError("Pipeline must be fitted before calling transform.")
        return self.preprocessor.transform(X[self.numerical_features + self.categorical_features])

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> np.ndarray:
        """Fit and transform in a single pass."""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self) -> List[str]:
        """Return generated feature names after one-hot encoding."""
        if not self._is_fitted:
            raise RuntimeError("Pipeline must be fitted before querying feature names.")
        return self._feature_names

    def save(self, filepath: Path) -> None:
        """Serialize fitted pipeline to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: Path) -> "FraudFeaturePipeline":
        """Load serialized pipeline from disk."""
        return joblib.load(filepath)


def extract_features_and_targets(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Separate input features (X), fraud label (y), and attack type metadata."""
    feature_cols = [c for c in df.columns if c not in TARGET_COLUMNS]
    X = df[feature_cols].copy()
    y = df["fraud_label"].copy()
    attack_types = df["attack_type"].copy()
    return X, y, attack_types
