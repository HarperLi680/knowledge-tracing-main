# Knowledge Tracing Models: Project Setup Guide

This project is managed using [`uv`](https://github.com/astral-sh/uv) as the Python packaging and virtual environment manager, and is designed to run on **Linux** and **macOS** systems.

---

## Environment Setup (using `uv`)

### 1. Install `uv`
If not already installed, install `uv` (if it does not work for you, please refer to the official uv doc for installation):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Set up project environment
From the root of the project:
```bash
uv venv .venv
source .venv/bin/activate
uv sync
```
This installs all dependencies listed in pyproject.toml.

---

## PyTorch Installation

**Important:** `pyproject.toml` does not include PyTorch as a dependency because PyTorch installation depends on your hardware (CUDA version, GPU architecture).

### For RTX 5xxx (or newer GPUs with CUDA 13.0+)

This project was developed using an RTX 5090 GPU, which requires PyTorch nightly with CUDA 13.0+ support:

```bash
uv pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu130
```

### For other GPUs

Please visit [PyTorch's official installation guide](https://pytorch.org/get-started/locally/) and select the appropriate configuration for your hardware. If you are using nvidia GPU, you may first find your cuda version with 
```bash
nvidia-smi
```

Then install using:

```bash
uv pip install torch torchvision --index-url <your-appropriate-index-url>
```

**Note:** Newer GPU architectures (like RTX 5090 with compute capability sm_120) may require nightly builds or specific CUDA versions. Adjust the PyTorch version based on your hardware capabilities.

---
## pyBKT Environment Setup
[`pyBKT`](https://github.com/CAHLR/pyBKT) is a library for Bayesian Knowledge Tracing. It supports both:
- A pure Python version (easy to install, slow on large datasets)
- A C++-accelerated version (requires manual build but much faster)

---
## Installing pyBKT
**Option 1: (Recommended for Large Datasets) — Install C++ Version Manually**
The uv package manager installs the Python-only version of pyBKT, which is significantly slower for model fitting on large datasets.

To install the **C++-accelerated version**, follow these steps:
```bash
# Step 1a: Install OpenMP (macOS)
brew install libomp

# Step 1b: Install OpenMP (Linux)
sudo apt-get update
sudo apt-get install -y libomp-dev build-essential python3-dev

# Step 2: Clone the repo
git clone https://github.com/CAHLR/pyBKT.git repos/pyBKT

# Step 3: Activate your project environment
source .venv/bin/activate

# Step 4: Export OpenMP flags (macOS only)
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib"
export CPPFLAGS="-I/opt/homebrew/opt/libomp/include"

# Step 5: Install using setup.py
cd repos/pyBKT
python setup.py install

# Step 6: Clean up 
cd ../..
rm -rf repos/
```

---
**Option 2: (Fine for Small Datasets) — Use Python Version**
```bash
uv add pyBKT
```
This will install the pure Python version from PyPI, which is simpler but much slower (e.g., 1–2 hours vs 1–2 seconds for large models).

--- 
**Option 3:**
Using Ryan's Brute Force BKT located at `Models/bkt_bf.py`.

## Data cleaning
The data preprocess pipeline has an entry script in `preprocess/preprocess.py`, where it preprocesses the raw Assessment09 data from `data/raw/skill_builder_data_corrected_collapsed.csv`. The cleaned data are train and test, split in `data/processed`. The train data are further processed for tabular and sequential models under `data/processed/train/tabular` and `data/processed/train/sequential`, respectively.

---

## Repository Structure

### Top-level scripts
- `preprocess/preprocess.py`: end-to-end preprocessing pipeline (clean → split → format).
- `generate_predictions.py`: trains each model across folds and writes raw predictions to `output/predictions.json`.
- `evaluate_predictions.py`: aligns predictions back to row-level data and produces:
	- `output/combined_output.csv` (original data + model prediction columns)
	- `output/metrics.json` (AUC/Accuracy/RMSE per model)
- `unit_test.py`: quick sanity check that each model train/predict function runs for one fold.

### Folders
- `Models/`: model implementations and `train_predict_*` entry points used by `generate_predictions.py`.
- `Utils/`: helper utilities used by deep learning models (e.g., ATKT/DSAKT).
- `preprocess/`: data cleaning/splitting/formatting modules.
- `data/`:
	- `raw/`: raw dataset file(s) (input to preprocessing)
	- `processed/`: preprocessed outputs
		- `train/tabular/`: fold CSVs for traditional models (BKT/PFA/KTM/Elo)
		- `train/sequential/`: fold files for sequential models (ATKT/DSAKT/DKT)
		- `test/`: held-out test split (currently not used by the provided scripts)
- `output/`: generated artifacts (`predictions.json`, `combined_output.csv`, `metrics.json`).

## Unit Test, Prediction, and Evaluation Generation
First, try to run `unit_test.py` from the repository root to ensure all model training prediction methods work properly for one fold of the training data. The main entry point for this repo is at `generate_predictions.py`, where it trains models on 5-fold data and validates the model on the remaining one fold of data. The script produces a prediction artifact `output/predictions.json`, which is used by the `evaluate_predictions.py` script to generate metrics `output/metrics.json` to compare the performance of different models. It is **important** to note that the test data in `data/processed/test` is NOT USED in the current scripts. The current performance metric is based on the validation dataset.
