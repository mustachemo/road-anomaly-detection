"""Feature processing module for the ROAD dataset."""

from typing import Optional, Tuple

import numpy as np
from rich.console import Console
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

console = Console()


class FeatureProcessor:
    """A class to handle feature processing and engineering for the ROAD dataset.

    This class provides methods for preprocessing data, including normalization,
    dimensionality reduction, and feature engineering.

    Attributes:
        n_components: Number of components for PCA.
        scaler: StandardScaler instance for feature normalization.
        pca: PCA instance for dimensionality reduction.
    """

    def __init__(self, n_components: Optional[int] = None) -> None:
        """Initialize the feature processor.

        Args:
            n_components: Number of components for PCA. If None, no PCA is applied.
        """
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components) if n_components is not None else None

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

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform the input data using the fitted processors.

        Args:
            X: Input data array of shape (n_samples, n_features).

        Returns:
            Transformed data array.

        Raises:
            ValueError: If the input data is invalid or processor is not fitted.
            RuntimeError: If the processor has not been fitted.
        """
        if not isinstance(X, np.ndarray):
            msg = f"Input must be numpy array, got {type(X)}"
            console.print(f"[red]Error: {msg}[/red]")
            raise ValueError(msg)

        if X.ndim != 2:
            msg = f"Input must be 2D array, got shape {X.shape}"
            console.print(f"[red]Error: {msg}[/red]")
            raise ValueError(msg)

        # Check if the processor has been fitted
        if not hasattr(self.scaler, "mean_"):
            msg = "FeatureProcessor has not been fitted. Call fit() first."
            console.print(f"[red]Error: {msg}[/red]")
            raise RuntimeError(msg)

        # Apply scaling
        X_scaled = self.scaler.transform(X)

        # Apply PCA if enabled
        if self.pca is not None:
            if not hasattr(self.pca, "components_"):
                msg = "PCA has not been fitted. Call fit() first."
                console.print(f"[red]Error: {msg}[/red]")
                raise RuntimeError(msg)
            X_transformed = self.pca.transform(X_scaled)
        else:
            X_transformed = X_scaled

        return X_transformed

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the processor and transform the input data.

        Args:
            X: Input data array of shape (n_samples, n_features).

        Returns:
            Transformed data array.

        Raises:
            ValueError: If the input data is invalid.
        """
        return self.fit(X).transform(X)

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
