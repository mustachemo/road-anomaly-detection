# road-anomaly-detection

## Repository Layout

```text
svm‑anomaly‑detection/
├── README.md
├── pyproject.toml            # dependency & tooling pinning (poetry / hatch / pdm)
├── .env.example              # non‑secret environment variables
├── .gitignore
├── Makefile                  # one‑liners for common tasks
├── data/
│   ├── raw/                  # immutable source data
│   ├── interim/              # scratch / sampled data
│   └── processed/            # cleaned feature matrices & labels
├── notebooks/                # exploratory, throw‑away research
│   ├── 01_data_overview.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_svm_baseline.ipynb
├── src/
│   ├── __init__.py
│   ├── config/               # YAML configs (train / inference)
│   ├── data/                 # ingestion, validation, splitting
│   ├── features/             # feature engineering & scaling
│   ├── models/               # One‑Class / binary SVM wrappers
│   ├── evaluation/           # metrics & plots
│   ├── pipeline.py           # single CLI entry‑point (train / predict)
│   └── utils.py              # logging, timing, path helpers
├── models/                   # serialised artefacts (joblib / ONNX)
├── reports/
│   └── figures/              # auto‑generated evaluation graphics
├── tests/                    # pytest unit tests
├── docker/
│   ├── Dockerfile
│   └── start.sh
└── .github/
    └── workflows/            # CI / CD (lint, tests, image build)
````

---

## What lives where & why

| Layer / Dir            | Purpose                                                           | Key points                                        |
| ---------------------- | ----------------------------------------------------------------- | ------------------------------------------------- |
| **data/**              | Immutable lineage; raw stays read‑only                            | Makes experiments deterministic & auditable       |
| **notebooks/**         | Interactive research                                              | Once logic stabilises, migrate to **src/**        |
| **src/data/**          | Download, schema validation, deterministic train/val/test split   | Splits are frozen via on‑disk indices             |
| **src/features/**      | Transformations (e.g. FFT → log‑mag → StandardScaler → PCA)       | Good features make kernels shine                  |
| **src/models/**        | Thin wrappers around `sklearn.svm.*`                              | Encapsulate create → fit → predict → save         |
| **pipeline.py**        | Orchestrator & CLI (`python -m svm_anomaly_detection.pipeline …`) | Trains or serves with one command                 |
| **tests/**             | Fast (<1 s) unit tests with pytest                                | Keep coverage for critical paths                  |
| **docker/**            | Parity from laptop to prod                                        | Image can run `pipeline.py predict` behind an API |
| **.github/workflows/** | CI (lint, type‑check, unit tests) + optional CD                   | Fails fast before merge                           |
