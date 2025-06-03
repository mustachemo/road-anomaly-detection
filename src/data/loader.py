"""Data loading module for the ROAD dataset."""

from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
from rich.console import Console
from rich.progress import Progress

console = Console()


class ROADDataLoader:
    """A class to handle loading and preprocessing of the ROAD dataset.

    This class provides methods to load data from the ROAD dataset HDF5 file,
    with support for batching and basic preprocessing.

    Attributes:
        data_path: Path to the ROAD dataset HDF5 file.
        batch_size: Number of samples to load in each batch.
    """

    def __init__(self, data_path: Path, batch_size: int = 32) -> None:
        """Initialize the ROAD data loader.

        Args:
            data_path: Path to the ROAD dataset HDF5 file.
            batch_size: Number of samples to load in each batch.

        Raises:
            FileNotFoundError: If the data file does not exist.
            ValueError: If the batch size is not positive.
        """
        if not data_path.exists():
            msg = f"Data file not found: {data_path}"
            console.print(f"[red]Error: {msg}[/red]")
            raise FileNotFoundError(msg)

        if batch_size <= 0:
            msg = f"Batch size must be positive, got {batch_size}"
            console.print(f"[red]Error: {msg}[/red]")
            raise ValueError(msg)

        self.data_path = data_path
        self.batch_size = batch_size
        self._file: Optional[h5py.File] = None
        self._total_samples: Optional[int] = None

    def __enter__(self) -> "ROADDataLoader":
        """Context manager entry point.

        Returns:
            The ROADDataLoader instance.
        """
        self._file = h5py.File(self.data_path, "r")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit point.

        Closes the HDF5 file if it's open.
        """
        if self._file is not None:
            self._file.close()
            self._file = None

    @property
    def total_samples(self) -> int:
        """Get the total number of samples in the dataset.

        Returns:
            The total number of samples.

        Raises:
            RuntimeError: If the data file is not open.
        """
        if self._file is None:
            msg = "Data file is not open. Use the context manager or call open() first."
            console.print(f"[red]Error: {msg}[/red]")
            raise RuntimeError(msg)

        if self._total_samples is None:
            # Assuming the dataset has a 'data' group with samples
            self._total_samples = len(self._file["data"])
        return self._total_samples

    def get_batch(self, start_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get a batch of data starting from the given index.

        Args:
            start_idx: Starting index for the batch.

        Returns:
            A tuple containing:
                - The batch of data samples
                - The corresponding labels

        Raises:
            RuntimeError: If the data file is not open.
            ValueError: If start_idx is out of bounds.
        """
        if self._file is None:
            msg = "Data file is not open. Use the context manager or call open() first."
            console.print(f"[red]Error: {msg}[/red]")
            raise RuntimeError(msg)

        if start_idx < 0 or start_idx >= self.total_samples:
            msg = f"Start index {start_idx} out of bounds [0, {self.total_samples})"
            console.print(f"[red]Error: {msg}[/red]")
            raise ValueError(msg)

        end_idx = min(start_idx + self.batch_size, self.total_samples)

        with Progress() as progress:
            task = progress.add_task(
                f"Loading batch {start_idx}-{end_idx}...", total=end_idx - start_idx
            )

            # Load data and labels
            data = self._file["data"][start_idx:end_idx]
            labels = self._file["labels"][start_idx:end_idx]

            progress.update(task, completed=end_idx - start_idx)

        return data, labels

    def get_dataset_info(self) -> Dict[str, any]:
        """Get information about the dataset.

        Returns:
            A dictionary containing dataset information.

        Raises:
            RuntimeError: If the data file is not open.
        """
        if self._file is None:
            msg = "Data file is not open. Use the context manager or call open() first."
            console.print(f"[red]Error: {msg}[/red]")
            raise RuntimeError(msg)

        info = {
            "total_samples": self.total_samples,
            "data_shape": self._file["data"].shape,
            "label_shape": self._file["labels"].shape,
            "data_dtype": str(self._file["data"].dtype),
            "label_dtype": str(self._file["labels"].dtype),
        }

        return info
