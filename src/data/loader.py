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
        data_group: The group in the HDF5 file to load data from (e.g., 'train_data', 'test_data').
    """

    def __init__(
        self, data_path: Path, batch_size: int = 32, data_group: str = "train_data"
    ) -> None:
        """Initialize the ROAD data loader.

        Args:
            data_path: Path to the ROAD dataset HDF5 file.
            batch_size: Number of samples to load in each batch.
            data_group: The group in the HDF5 file to load data from (e.g., 'train_data', 'test_data').

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
        self.data_group = data_group
        self._file: Optional[h5py.File] = None
        self._total_samples: Optional[int] = None

    def __enter__(self) -> "ROADDataLoader":
        """Context manager entry point.

        Returns:
            The ROADDataLoader instance.
        """
        self._file = h5py.File(self.data_path, "r")
        self._validate_file_structure()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit point.

        Closes the HDF5 file if it's open.
        """
        if self._file is not None:
            self._file.close()
            self._file = None

    def _validate_file_structure(self) -> None:
        """Validate the HDF5 file structure.

        Raises:
            ValueError: If the file structure is invalid.
        """
        if self._file is None:
            msg = "Data file is not open. Use the context manager or call open() first."
            console.print(f"[red]Error: {msg}[/red]")
            raise RuntimeError(msg)

        # Check if the specified data group exists
        if self.data_group not in self._file:
            available_groups = list(self._file.keys())
            msg = f"Data group '{self.data_group}' not found in HDF5 file. Available groups: {available_groups}"
            console.print(f"[red]Error: {msg}[/red]")
            raise ValueError(msg)

        group = self._file[self.data_group]
        required_datasets = ["data", "labels"]
        for dataset in required_datasets:
            if dataset not in group:
                msg = f"Required dataset '{dataset}' not found in group '{self.data_group}'. Available datasets: {list(group.keys())}"
                console.print(f"[red]Error: {msg}[/red]")
                raise ValueError(msg)

        # Check data shape
        data_shape = group["data"].shape
        if len(data_shape) != 4:  # (n_samples, height, width, channels)
            msg = f"Data must be 4D array (n_samples, height, width, channels), got shape {data_shape}"
            console.print(f"[red]Error: {msg}[/red]")
            raise ValueError(msg)

        # Check labels shape
        labels_shape = group["labels"].shape
        if len(labels_shape) != 1:
            msg = f"Labels must be 1D array, got shape {labels_shape}"
            console.print(f"[red]Error: {msg}[/red]")
            raise ValueError(msg)

        # Check if number of samples matches
        if data_shape[0] != labels_shape[0]:
            msg = f"Number of samples in data ({data_shape[0]}) does not match number of labels ({labels_shape[0]})"
            console.print(f"[red]Error: {msg}[/red]")
            raise ValueError(msg)

        console.print(f"[green]Validated HDF5 file structure:[/green]")
        console.print(f"  - Data group: {self.data_group}")
        console.print(f"  - Data shape: {data_shape}")
        console.print(f"  - Labels shape: {labels_shape}")

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
            self._total_samples = len(self._file[self.data_group]["data"])
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

            # Load data and labels from the specified group
            data = self._file[self.data_group]["data"][start_idx:end_idx]
            labels = self._file[self.data_group]["labels"][start_idx:end_idx]

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

        group = self._file[self.data_group]
        info = {
            "data_group": self.data_group,
            "total_samples": self.total_samples,
            "data_shape": group["data"].shape,
            "label_shape": group["labels"].shape,
            "data_dtype": str(group["data"].dtype),
            "label_dtype": str(group["labels"].dtype),
        }

        return info
