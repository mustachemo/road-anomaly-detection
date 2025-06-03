# Road Anomaly Detection 🛣️

A Python-based tool for detecting anomalies in road surface images using machine learning. This project provides both a command-line interface and a web interface for training and using anomaly detection models.

## Features

- **Data Processing**: Efficient handling of road surface images using HDF5 format
- **Feature Extraction**: Advanced feature processing with optional PCA dimensionality reduction
- **Anomaly Detection**: One-class SVM-based anomaly detection
- **Multiple Interfaces**:
  - Command-line interface for batch processing
  - Web interface (Gradio) for interactive training and visualization
- **Progress Tracking**: Real-time progress monitoring during training and prediction
- **Testing**: Comprehensive test suite for model validation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/road-anomaly-detection.git
cd road-anomaly-detection
```

2. Install dependencies using UV:
```bash
uv pip install -e .
```

## Usage

### Command Line Interface

Train a model:
```bash
uv run -m src.pipeline train data/road_dataset.h5 model.joblib
```

Predict anomalies:
```bash
uv run -m src.pipeline predict data/test_dataset.h5 model.joblib predictions.npy
```

### Web Interface

Launch the Gradio web interface:
```bash
uv run -m src.web_app
```

The web interface provides:
- Dataset upload and validation
- Interactive model training with parameter adjustment
- Real-time training progress visualization
- Anomaly prediction and visualization

## Project Structure

```
road-anomaly-detection/
├── src/
│   ├── data/
│   │   └── loader.py          # Data loading and preprocessing
│   ├── features/
│   │   └── processor.py       # Feature extraction and processing
│   ├── models/
│   │   └── anomaly_detector.py # Anomaly detection model
│   ├── pipeline.py            # Command-line interface
│   └── web_app.py            # Web interface
├── tests/
│   └── test_anomaly_detector.py
├── notebooks/
│   └── dataset_exploration.ipynb
└── pyproject.toml
```

## Development

### Running Tests

```bash
uv run pytest
```

### Code Style

The project follows PEP 8 guidelines and uses type hints throughout the codebase.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
