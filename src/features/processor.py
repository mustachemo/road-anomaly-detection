"""Feature processing module for road anomaly detection."""

from typing import Optional

import numpy as np
from rich.console import Console
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

console = Console()


class FeatureProcessor:
    """A class to handle feature processing for anomaly detection.

    This class provides methods for preprocessing and dimensionality reduction
    of input features using PCA and standardization.

    Attributes:
        n_components: Number of PCA components to use. If None, no PCA is applied.
        scaler: StandardScaler instance for feature normalization.
        pca: PCA instance for dimensionality reduction.
    """

    def __init__(self, n_components: Optional[int] = None) -> None:
        """Initialize the feature processor.

        Args:
            n_components: Number of PCA components to use. If None, no PCA is applied.
        """
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components) if n_components is not None else None

    def _reshape_for_processing(self, X: np.ndarray) -> np.ndarray:
        """Reshape input data for processing.

        Args:
            X: Input data array of shape (n_samples, height, width, channels).

        Returns:
            Reshaped array of shape (n_samples, height * width * channels).
        """
        n_samples = X.shape[0]
        return X.reshape(n_samples, -1)

    def fit(self, X: np.ndarray) -> "FeatureProcessor":
        """Fit the feature processor to the data.

        Args:
            X: Input data array of shape (n_samples, n_features).

        Returns:
            self: The fitted FeatureProcessor instance.

        Raises:
            ValueError: If the input data is invalid.
        """
        if not isinstance(X, np.ndarray):
            msg = f"Input must be numpy array, got {type(X)}"
            console.print(f"[red]Error: {msg}[/red]")
            raise ValueError(msg)

        if X.ndim != 2:
            msg = f"Input must be 2D array, got shape {X.shape}"
            console.print(f"[red]Error: {msg}[/red]")
            raise ValueError(msg)

        # Fit the scaler
        self.scaler.fit(X)

        # If PCA is enabled, fit it on the scaled data
        if self.pca is not None:
            X_scaled = self.scaler.transform(X)
            self.pca.fit(X_scaled)

        return self

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the processor to the data and transform it.

        Args:
            X: Input data array of shape (n_samples, height, width, channels).

        Returns:
            Transformed data array.

        Raises:
            ValueError: If input data is not 4D.
        """
        if len(X.shape) != 4:
            raise ValueError(f"Input must be 4D array, got shape {X.shape}")

        # Reshape to 2D
        X_reshaped = self._reshape_for_processing(X)

        # Standardize features
        X_scaled = self.scaler.fit_transform(X_reshaped)

        # Apply PCA if configured
        if self.pca is not None:
            return self.pca.fit_transform(X_scaled)
        return X_scaled

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the data using the fitted processor.

        Args:
            X: Input data array of shape (n_samples, height, width, channels).

        Returns:
            Transformed data array.

        Raises:
            ValueError: If input data is not 4D.
        """
        if len(X.shape) != 4:
            raise ValueError(f"Input must be 4D array, got shape {X.shape}")

        # Reshape to 2D
        X_reshaped = self._reshape_for_processing(X)

        # Standardize features
        X_scaled = self.scaler.transform(X_reshaped)

        # Apply PCA if configured
        if self.pca is not None:
            return self.pca.transform(X_scaled)
        return X_scaled

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance scores from PCA.

        Returns:
            Array of feature importance scores if PCA is enabled, None otherwise.
        """
        if self.pca is None:
            return None

        if not hasattr(self.pca, "components_"):
            msg = "PCA has not been fitted. Call fit() first."
            console.print(f"[yellow]Warning: {msg}[/yellow]")
            return None

        # Calculate feature importance as the sum of absolute values of PCA components
        return np.sum(np.abs(self.pca.components_), axis=0)

    def get_explained_variance_ratio(self) -> Optional[np.ndarray]:
        """Get the explained variance ratio from PCA.

        Returns:
            Array of explained variance ratios if PCA is enabled, None otherwise.
        """
        if self.pca is None:
            return None

        if not hasattr(self.pca, "explained_variance_ratio_"):
            msg = "PCA has not been fitted. Call fit() first."
            console.print(f"[yellow]Warning: {msg}[/yellow]")
            return None

        return self.pca.explained_variance_ratio_
