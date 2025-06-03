"""Anomaly detection module using One-Class SVM."""

from pathlib import Path
from typing import Optional, Tuple

import joblib
import numpy as np
from rich.console import Console
from sklearn.svm import OneClassSVM

console = Console()


class AnomalyDetector:
    """A class for anomaly detection using One-Class SVM.

    This class implements an anomaly detector using scikit-learn's OneClassSVM,
    with methods for training, prediction, and model persistence.

    Attributes:
        kernel: The kernel type for the SVM.
        nu: An upper bound on the fraction of training errors and a lower bound
            of the fraction of support vectors.
        gamma: Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
        model: The trained OneClassSVM model.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        nu: float = 0.1,
        gamma: str = "scale",
    ) -> None:
        """Initialize the anomaly detector.

        Args:
            kernel: The kernel type ('linear', 'poly', 'rbf', 'sigmoid').
            nu: An upper bound on the fraction of training errors.
            gamma: Kernel coefficient ('scale', 'auto' or float).

        Raises:
            ValueError: If kernel or gamma parameters are invalid.
        """
        valid_kernels = ["linear", "poly", "rbf", "sigmoid"]
        if kernel not in valid_kernels:
            msg = f"Invalid kernel. Must be one of {valid_kernels}, got {kernel}"
            console.print(f"[red]Error: {msg}[/red]")
            raise ValueError(msg)

        if gamma not in ["scale", "auto"] and not isinstance(gamma, (int, float)):
            msg = f"Invalid gamma. Must be 'scale', 'auto' or float, got {gamma}"
            console.print(f"[red]Error: {msg}[/red]")
            raise ValueError(msg)

        if not 0 < nu <= 1:
            msg = f"nu must be in (0, 1], got {nu}"
            console.print(f"[red]Error: {msg}[/red]")
            raise ValueError(msg)

        self.kernel = kernel
        self.nu = nu
        self.gamma = gamma
        self.model: Optional[OneClassSVM] = None

    def fit(self, X: np.ndarray) -> "AnomalyDetector":
        """Fit the One-Class SVM model.

        Args:
            X: Training data of shape (n_samples, n_features).

        Returns:
            self: The fitted AnomalyDetector instance.

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

        self.model = OneClassSVM(
            kernel=self.kernel,
            nu=self.nu,
            gamma=self.gamma,
        )

        with console.status("[bold green]Training One-Class SVM..."):
            self.model.fit(X)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict anomalies in the input data.

        Args:
            X: Input data of shape (n_samples, n_features).

        Returns:
            Array of predictions (-1 for anomalies, 1 for normal).

        Raises:
            ValueError: If the input data is invalid.
            RuntimeError: If the model has not been fitted.
        """
        if not isinstance(X, np.ndarray):
            msg = f"Input must be numpy array, got {type(X)}"
            console.print(f"[red]Error: {msg}[/red]")
            raise ValueError(msg)

        if X.ndim != 2:
            msg = f"Input must be 2D array, got shape {X.shape}"
            console.print(f"[red]Error: {msg}[/red]")
            raise ValueError(msg)

        if self.model is None:
            msg = "Model has not been fitted. Call fit() first."
            console.print(f"[red]Error: {msg}[/red]")
            raise RuntimeError(msg)

        return self.model.predict(X)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute the decision function of the samples.

        Args:
            X: Input data of shape (n_samples, n_features).

        Returns:
            Array of decision function values.

        Raises:
            ValueError: If the input data is invalid.
            RuntimeError: If the model has not been fitted.
        """
        if not isinstance(X, np.ndarray):
            msg = f"Input must be numpy array, got {type(X)}"
            console.print(f"[red]Error: {msg}[/red]")
            raise ValueError(msg)

        if X.ndim != 2:
            msg = f"Input must be 2D array, got shape {X.shape}"
            console.print(f"[red]Error: {msg}[/red]")
            raise ValueError(msg)

        if self.model is None:
            msg = "Model has not been fitted. Call fit() first."
            console.print(f"[red]Error: {msg}[/red]")
            raise RuntimeError(msg)

        return self.model.decision_function(X)

    def save(self, path: Path) -> None:
        """Save the model to disk.

        Args:
            path: Path where the model will be saved.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        if self.model is None:
            msg = "Model has not been fitted. Call fit() first."
            console.print(f"[red]Error: {msg}[/red]")
            raise RuntimeError(msg)

        try:
            joblib.dump(self.model, path)
            console.print(f"[green]Model saved to {path}[/green]")
        except Exception as e:
            msg = f"Failed to save model to {path}: {e}"
            console.print(f"[red]Error: {msg}[/red]")
            raise

    @classmethod
    def load(cls, path: Path) -> "AnomalyDetector":
        """Load a model from disk.

        Args:
            path: Path to the saved model.

        Returns:
            An AnomalyDetector instance with the loaded model.

        Raises:
            FileNotFoundError: If the model file does not exist.
            ValueError: If the loaded model is invalid.
        """
        if not path.exists():
            msg = f"Model file not found: {path}"
            console.print(f"[red]Error: {msg}[/red]")
            raise FileNotFoundError(msg)

        try:
            model = joblib.load(path)
            if not isinstance(model, OneClassSVM):
                msg = f"Invalid model type. Expected OneClassSVM, got {type(model)}"
                console.print(f"[red]Error: {msg}[/red]")
                raise ValueError(msg)

            detector = cls()
            detector.model = model
            return detector
        except Exception as e:
            msg = f"Failed to load model from {path}: {e}"
            console.print(f"[red]Error: {msg}[/red]")
            raise
