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

## Local Inference

Run one prediction with the frozen selected CNN:

```powershell
.\.venv\Scripts\python.exe -m ml.src.predict_audio ml\data\raw\ravdess\extracted\Actor_01\03-01-05-01-01-01-01.wav
```

Inputs must be readable PCM_16 WAV recordings with one or two channels and a positive sample rate. The frozen pipeline handles stereo averaging where needed, resampling, silence trimming, fixed-length padding or cropping, Log-Mel extraction, and the training-only normalization statistics.

The JSON result contains `emotion`, `class_index`, `confidence`, `probabilities`, and frozen model identity/version fields. Confidence is the selected model's softmax probability; it is not a guarantee that the predicted emotion is correct.

## Backend Development

Run the development API from the project root:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

Health endpoint: `http://127.0.0.1:8000/api/v1/health`

Submit one WAV recording using the multipart field `file`:

```powershell
curl.exe -X POST http://127.0.0.1:8000/api/v1/predict -F "file=@path\to\recording.wav"
```

The response includes `emotion`, `class_index`, `confidence`, eight class `probabilities`, and `model.identifier` and `model.version`. The endpoint accepts files up to 10 MiB by default; set `SER_MAX_AUDIO_BYTES` to configure a different positive limit.

## Frontend Development

Start the React and TypeScript interface from the project root:

```powershell
cd frontend
npm install
npm run dev
```

Run frontend checks with `npm test`, `npm run lint`, and `npm run build`. The current interface validates WAV selection locally; API upload and microphone recording are intentionally not connected yet.

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

A fixed baseline SVM candidate has been trained and evaluated on validation actors; it is not the final selected model.

A speaker-grouped SVM tuning experiment did not improve validation performance; the original baseline remains the leading SVM candidate and the test split remains untouched.

The first baseline CNN was trained and evaluated on validation actors; it did not outperform the leading SVM, and the final test split remains untouched.

The reduced-regularization CNN improved validation performance beyond the baseline SVM; it remains a validation-only candidate and the final test split is untouched.

The regularized-and-augmented CNN did not meet the fixed promotion threshold, so the reduced-regularization CNN was selected and frozen before opening the final test split.

A frozen-Wav2Vec2 transfer-learning candidate was evaluated on validation actors only. It did not meet the predeclared promotion rule, and it was not evaluated on the final test split.

The FastAPI backend foundation provides a health endpoint only; audio-prediction endpoints and the frontend have not been built.

Model selection used training actors 01-16 and validation actors 17-20. The frozen reduced-regularization/no-dropout CNN was evaluated once on held-out actors 21-24, achieving 46.6667% test accuracy and 0.419553 macro F1. No post-test tuning, model selection, or alternative-model test evaluation was performed.
