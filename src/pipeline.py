"""Main pipeline module for road anomaly detection."""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import typer
from rich.console import Console
from rich.progress import Progress

from src.data.loader import ROADDataLoader
from src.features.processor import FeatureProcessor
from src.models.anomaly_detector import AnomalyDetector

app = typer.Typer()
console = Console()


def train_model(
    data_path: Path,
    model_path: Path,
    n_components: Optional[int] = None,
    kernel: str = "rbf",
    nu: float = 0.1,
    gamma: str = "scale",
    batch_size: int = 32,
) -> Tuple[AnomalyDetector, FeatureProcessor]:
    """Train the anomaly detection model.

    Args:
        data_path: Path to the ROAD dataset HDF5 file.
        model_path: Path where the trained model will be saved.
        n_components: Number of PCA components. If None, no PCA is applied.
        kernel: SVM kernel type.
        nu: SVM nu parameter.
        gamma: SVM gamma parameter.
        batch_size: Batch size for data loading.

    Returns:
        A tuple containing:
            - The trained AnomalyDetector instance
            - The fitted FeatureProcessor instance

    Raises:
        FileNotFoundError: If the data file does not exist.
        RuntimeError: If training fails.
    """
    try:
        # Initialize components
        feature_processor = FeatureProcessor(n_components=n_components)
        anomaly_detector = AnomalyDetector(kernel=kernel, nu=nu, gamma=gamma)

        # Load and process data
        with ROADDataLoader(data_path, batch_size=batch_size) as loader:
            # Get dataset info
            info = loader.get_dataset_info()
            console.print(f"[bold]Dataset Info:[/bold]\n{info}")

            # Process data in batches
            total_samples = loader.total_samples
            processed_data = []

            with Progress() as progress:
                task = progress.add_task(
                    "[cyan]Processing data...", total=total_samples
                )

                for start_idx in range(0, total_samples, batch_size):
                    # Load batch
                    X_batch, _ = loader.get_batch(start_idx)

                    # Process batch
                    if len(processed_data) == 0:
                        # First batch: fit and transform
                        X_processed = feature_processor.fit_transform(X_batch)
                    else:
                        # Subsequent batches: transform only
                        X_processed = feature_processor.transform(X_batch)

                    processed_data.append(X_processed)
                    progress.update(
                        task, completed=min(start_idx + batch_size, total_samples)
                    )

            # Combine processed batches
            X_processed = np.vstack(processed_data)

        # Train model
        console.print("[bold green]Training anomaly detector...[/bold green]")
        anomaly_detector.fit(X_processed)

        # Save model
        model_path.parent.mkdir(parents=True, exist_ok=True)
        anomaly_detector.save(model_path)

        return anomaly_detector, feature_processor

    except Exception as e:
        msg = f"Training failed: {e}"
        console.print(f"[bold red]Error: {msg}[/bold red]")
        raise RuntimeError(msg)


def predict_anomalies(
    data_path: Path,
    model_path: Path,
    feature_processor: FeatureProcessor,
    batch_size: int = 32,
) -> np.ndarray:
    """Predict anomalies in the input data.

    Args:
        data_path: Path to the ROAD dataset HDF5 file.
        model_path: Path to the trained model.
        feature_processor: Fitted FeatureProcessor instance.
        batch_size: Batch size for data loading.

    Returns:
        Array of predictions (-1 for anomalies, 1 for normal).

    Raises:
        FileNotFoundError: If the data or model file does not exist.
        RuntimeError: If prediction fails.
    """
    try:
        # Load model
        anomaly_detector = AnomalyDetector.load(model_path)

        # Process and predict data in batches
        with ROADDataLoader(data_path, batch_size=batch_size) as loader:
            total_samples = loader.total_samples
            all_predictions = []

            with Progress() as progress:
                task = progress.add_task(
                    "[cyan]Predicting anomalies...", total=total_samples
                )

                for start_idx in range(0, total_samples, batch_size):
                    # Load and process batch
                    X_batch, _ = loader.get_batch(start_idx)
                    X_processed = feature_processor.transform(X_batch)

                    # Predict
                    predictions = anomaly_detector.predict(X_processed)
                    all_predictions.append(predictions)

                    progress.update(
                        task, completed=min(start_idx + batch_size, total_samples)
                    )

            # Combine predictions
            return np.concatenate(all_predictions)

    except Exception as e:
        msg = f"Prediction failed: {e}"
        console.print(f"[bold red]Error: {msg}[/bold red]")
        raise RuntimeError(msg)


@app.command()
def train(
    data_path: Path = typer.Argument(..., help="Path to the ROAD dataset HDF5 file"),
    model_path: Path = typer.Argument(
        ..., help="Path where the trained model will be saved"
    ),
    n_components: Optional[int] = typer.Option(None, help="Number of PCA components"),
    kernel: str = typer.Option("rbf", help="SVM kernel type"),
    nu: float = typer.Option(0.1, help="SVM nu parameter"),
    gamma: str = typer.Option("scale", help="SVM gamma parameter"),
    batch_size: int = typer.Option(32, help="Batch size for data loading"),
) -> None:
    """Train the anomaly detection model."""
    try:
        train_model(
            data_path=data_path,
            model_path=model_path,
            n_components=n_components,
            kernel=kernel,
            nu=nu,
            gamma=gamma,
            batch_size=batch_size,
        )
        console.print("[bold green]Training completed successfully![/bold green]")
    except Exception as e:
        console.print(f"[bold red]Training failed: {e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def predict(
    data_path: Path = typer.Argument(..., help="Path to the ROAD dataset HDF5 file"),
    model_path: Path = typer.Argument(..., help="Path to the trained model"),
    output_path: Path = typer.Argument(
        ..., help="Path where predictions will be saved"
    ),
    n_components: Optional[int] = typer.Option(None, help="Number of PCA components"),
    batch_size: int = typer.Option(32, help="Batch size for data loading"),
) -> None:
    """Predict anomalies in the input data."""
    try:
        # Initialize feature processor
        feature_processor = FeatureProcessor(n_components=n_components)

        # Load a small batch to fit the processor
        with ROADDataLoader(data_path, batch_size=batch_size) as loader:
            X_batch, _ = loader.get_batch(0)
            feature_processor.fit(X_batch)

        # Predict anomalies
        predictions = predict_anomalies(
            data_path=data_path,
            model_path=model_path,
            feature_processor=feature_processor,
            batch_size=batch_size,
        )

        # Save predictions
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, predictions)
        console.print(f"[bold green]Predictions saved to {output_path}[/bold green]")

    except Exception as e:
        console.print(f"[bold red]Prediction failed: {e}[/bold red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
