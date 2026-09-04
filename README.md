# Detecting Individuals' Emotions Based on Voice Using Artificial Intelligence

This project is a final-year project focused on recognizing human emotions from speech using artificial intelligence. The repository currently contains the initial project structure and the verified core machine-learning/audio environment.

## Prerequisites and Setup

- Python 3.11 (64-bit)
- Root-level virtual environment: `.venv`

Create the virtual environment with Python 3.11:

```powershell
py -3.11 -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the core ML and audio-processing packages:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements\ml.txt
```

Verify the ML environment:

```powershell
.\.venv\Scripts\python.exe ml\src\verify_ml_environment.py
```

This setup uses TensorFlow on CPU in a native Windows environment. Zero detected GPUs is expected in this project configuration.

Librosa is intentionally pinned to `0.11.0` because this project uses Python 3.11, while Librosa 1.0 requires Python 3.12 or newer.

The official dataset source and the planned speaker-independent split strategy are documented in [docs/dataset-plan.md](docs/dataset-plan.md).

## Proposed System Workflow

1. Collect and organize speech data using the RAVDESS dataset.
2. Preprocess audio and extract relevant speech features.
3. Train and evaluate a baseline SVM model.
4. Train and evaluate a primary CNN model for emotion classification.
5. Serve predictions through a FastAPI backend.
6. Provide voice recording and audio-file upload through a React and TypeScript frontend.
7. Return predicted emotion labels with confidence scores.

## Initial Technical Choices

- Initial emotion classes: neutral, calm, happy, sad, angry, fearful, disgust, surprised
- Proposed dataset: RAVDESS
- Proposed baseline model: SVM
- Proposed primary model: CNN
- Proposed backend: FastAPI
- Proposed frontend: React with TypeScript

These are initial technical decisions and may be refined after experimentation.

## Project Structure

```text
speech-emotion-recognition-system/
|-- backend/
|   `-- app/
|-- ml/
|   |-- data/
|   |   |-- raw/
|   |   `-- processed/
|   |-- notebooks/
|   |-- src/
|   |-- models/
|   `-- reports/
|       `-- figures/
|-- frontend/
|-- docs/
|-- requirements/
|   |-- ml.txt
|   `-- ml-lock.txt
|-- README.md
`-- .gitignore
```

## Current Status

The finalized reusable preprocessing pipeline is implemented and all 1,440 recordings pass in-memory validation. Raw audio remains unchanged and ignored by Git where appropriate. No processed audio, features, or trained models exist yet; feature extraction and model training have not started. No backend or frontend application functionality has been implemented yet.

Controlled waveform, log-Mel, and MFCC feature previews have been generated for one consistent training example from each emotion; these are visual previews only, not saved feature datasets.
