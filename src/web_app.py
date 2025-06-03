"""Web interface for road anomaly detection using Gradio."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from rich.console import Console

from src.data.loader import ROADDataLoader
from src.features.processor import FeatureProcessor
from src.models.anomaly_detector import AnomalyDetector

console = Console()


class TrainingVisualizer:
    """A class to handle training visualization and Gradio interface."""

    def __init__(self) -> None:
        """Initialize the training visualizer."""
        self.feature_processor: Optional[FeatureProcessor] = None
        self.anomaly_detector: Optional[AnomalyDetector] = None
        self.training_history: List[Dict[str, float]] = []
        self.current_batch_predictions: List[np.ndarray] = []
        self.current_batch_labels: List[np.ndarray] = []

    def plot_training_progress(self) -> plt.Figure:
        """Plot the training progress.

        Returns:
            A matplotlib figure showing training metrics.
        """
        if not self.training_history:
            return plt.figure()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Plot anomaly ratio
        anomaly_ratios = [h["anomaly_ratio"] for h in self.training_history]
        ax1.plot(anomaly_ratios, label="Anomaly Ratio")
        ax1.set_xlabel("Batch")
        ax1.set_ylabel("Ratio of Anomalies")
        ax1.set_title("Anomaly Detection Progress")
        ax1.grid(True)
        ax1.legend()

        # Plot decision function distribution
        if self.current_batch_predictions:
            decision_values = np.concatenate(self.current_batch_predictions)
            ax2.hist(decision_values, bins=50, alpha=0.7)
            ax2.set_xlabel("Decision Function Value")
            ax2.set_ylabel("Count")
            ax2.set_title("Decision Function Distribution")
            ax2.grid(True)

        plt.tight_layout()
        return fig

    def process_batch(
        self,
        X_batch: np.ndarray,
        y_batch: np.ndarray,
        is_first_batch: bool = False,
    ) -> Dict[str, float]:
        """Process a batch of data and update training metrics.

        Args:
            X_batch: Input data batch.
            y_batch: Labels batch.
            is_first_batch: Whether this is the first batch.

        Returns:
            Dictionary of metrics for this batch.
        """
        # Process features
        if is_first_batch:
            X_processed = self.feature_processor.fit_transform(X_batch)
        else:
            X_processed = self.feature_processor.transform(X_batch)

        # Get predictions
        predictions = self.anomaly_detector.predict(X_processed)
        decision_values = self.anomaly_detector.decision_function(X_processed)

        # Store predictions for visualization
        self.current_batch_predictions.append(decision_values)
        self.current_batch_labels.append(y_batch)

        # Calculate metrics
        anomaly_ratio = np.mean(predictions == -1)
        metrics = {
            "anomaly_ratio": float(anomaly_ratio),
            "batch_size": len(X_batch),
        }

        self.training_history.append(metrics)
        return metrics

    def train_model(
        self,
        data_path: str,
        n_components: Optional[int],
        kernel: str,
        nu: float,
        gamma: str,
        batch_size: int,
        progress: gr.Progress,
    ) -> Tuple[plt.Figure, str]:
        """Train the model with visualization.

        Args:
            data_path: Path to the dataset.
            n_components: Number of PCA components.
            kernel: SVM kernel type.
            nu: SVM nu parameter.
            gamma: SVM gamma parameter.
            batch_size: Batch size for training.
            progress: Gradio progress tracker.

        Returns:
            Tuple of (training plot, status message).
        """
        try:
            # Initialize components
            self.feature_processor = FeatureProcessor(n_components=n_components)
            self.anomaly_detector = AnomalyDetector(kernel=kernel, nu=nu, gamma=gamma)
            self.training_history = []
            self.current_batch_predictions = []
            self.current_batch_labels = []

            # Load and process data
            with ROADDataLoader(Path(data_path), batch_size=batch_size) as loader:
                total_samples = loader.total_samples
                n_batches = (total_samples + batch_size - 1) // batch_size

                for batch_idx in progress.track(range(n_batches), desc="Training"):
                    start_idx = batch_idx * batch_size
                    X_batch, y_batch = loader.get_batch(start_idx)

                    # Process batch
                    metrics = self.process_batch(
                        X_batch,
                        y_batch,
                        is_first_batch=(batch_idx == 0),
                    )

                    # Update progress
                    progress(
                        batch_idx / n_batches,
                        desc=f"Batch {batch_idx + 1}/{n_batches}",
                    )

            # Generate final plot
            fig = self.plot_training_progress()
            return fig, "Training completed successfully!"

        except Exception as e:
            msg = f"Training failed: {e}"
            console.print(f"[bold red]Error: {msg}[/bold red]")
            return plt.figure(), f"Error: {msg}"


def create_interface() -> gr.Interface:
    """Create the Gradio interface.

    Returns:
        A Gradio interface for the anomaly detection system.
    """
    visualizer = TrainingVisualizer()

    with gr.Blocks(title="Road Anomaly Detection") as interface:
        gr.Markdown("# Road Anomaly Detection Training Interface")

        with gr.Row():
            with gr.Column():
                # Input components
                data_file = gr.File(
                    label="Upload ROAD Dataset (HDF5)",
                    file_types=[".h5", ".hdf5"],
                )
                n_components = gr.Number(
                    label="Number of PCA Components",
                    value=None,
                    precision=0,
                )
                kernel = gr.Dropdown(
                    label="SVM Kernel",
                    choices=["rbf", "linear", "poly", "sigmoid"],
                    value="rbf",
                )
                nu = gr.Slider(
                    label="SVM Nu Parameter",
                    minimum=0.01,
                    maximum=1.0,
                    value=0.1,
                    step=0.01,
                )
                gamma = gr.Dropdown(
                    label="SVM Gamma Parameter",
                    choices=["scale", "auto"],
                    value="scale",
                )
                batch_size = gr.Number(
                    label="Batch Size",
                    value=32,
                    precision=0,
                )
                train_btn = gr.Button("Start Training", variant="primary")

            with gr.Column():
                # Output components
                plot = gr.Plot(label="Training Progress")
                status = gr.Textbox(label="Status")

        # Set up event handlers
        train_btn.click(
            fn=visualizer.train_model,
            inputs=[
                data_file,
                n_components,
                kernel,
                nu,
                gamma,
                batch_size,
                gr.Progress(),
            ],
            outputs=[plot, status],
        )

    return interface


if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=True)
