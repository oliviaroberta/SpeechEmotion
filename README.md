# Detecting Individuals' Emotions Based on Voice Using Artificial Intelligence

This project is a final-year project focused on recognizing human emotions from speech using artificial intelligence. It contains a frozen CNN inference pipeline, FastAPI prediction API, and React interface for WAV upload and browser microphone recording.

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

The response includes `emotion`, `class_index`, `confidence`, eight class `probabilities`, and `model.identifier` and `model.version`. The endpoint accepts files up to 4 MiB by default; set `SER_MAX_AUDIO_BYTES` to configure a different positive limit.

## Production Preparation

The repository tracks only the frozen inference bundle needed at runtime:

- `ml/artifacts/inference/cnn_reduced_regularization.keras`
- `ml/artifacts/inference/normalization.npz`
- `ml/artifacts/inference/manifest.json`

Install the minimal CPU inference runtime with Python 3.11:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements\runtime.txt
```

Set `SER_ENVIRONMENT=production` and provide a comma-separated `ALLOWED_ORIGINS` value containing the deployed frontend origins. Wildcard CORS origins are not allowed. Set `PORT` to the hosting platform's port and run one TensorFlow worker without reload:

```text
uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT --workers 1
```

The included `Procfile` records that production start command for Linux-compatible process hosts. The frontend uses `VITE_API_BASE_URL`; set it to the public backend origin at build time. `frontend/.env.example` contains a non-secret example URL, and `frontend/vercel.json` provides the Vite single-page-app fallback when the frontend project root is deployed to Vercel.

### Vercel Projects

The backend and frontend are separate Vercel projects from this repository:

- Backend project Root Directory: repository root. `api/index.py` exports `backend.app.main:app`; root `vercel.json` configures a 300-second function duration and includes only the frozen inference artifacts and required Python source.
- Frontend project Root Directory: `frontend`. Its existing `frontend/vercel.json` keeps client-side routing separate from the FastAPI function.

The root `.python-version` requests Python 3.12 for Vercel only; local development remains on the existing Python 3.11 `.venv`. Configure these backend variables in Vercel:

```text
SER_ENVIRONMENT=production
ALLOWED_ORIGINS=https://speech-emotion-six.vercel.app,http://localhost:5173
SER_MAX_AUDIO_BYTES=4194304
VERCEL_SUPPORT_LARGE_FUNCTIONS=1
```

Configure `VITE_API_BASE_URL=https://your-backend.vercel.app` in the frontend project. Vercel Functions cap request bodies at 4.5 MB, so this application uses a conservative 4 MiB upload limit for both browser validation and backend enforcement. A 15-second mono PCM-16 microphone WAV remains below this limit at common browser sample rates. TensorFlow may exceed Vercel's standard Python bundle allowance, so `VERCEL_SUPPORT_LARGE_FUNCTIONS=1` might be required. The configuration does not guarantee that the backend will fit or build; verify that in a Vercel build before deployment.

## Frontend Development

Start the React and TypeScript interface from the project root:

```powershell
cd frontend
npm install
npm run dev
```

Run frontend checks with `npm test`, `npm run lint`, and `npm run build`. The interface validates WAV selection locally and sends both uploads and browser-recorded WAV files to the configured prediction API only after analysis is requested.

For local end-to-end WAV predictions, run the API in one PowerShell window:

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Then run `npm run dev` from `frontend` in a second window. The frontend uses `VITE_API_BASE_URL`, which defaults to `http://127.0.0.1:8000`; see `frontend/.env.example`. WAV upload and browser-recorded WAV analysis use the same prediction endpoint.

The browser microphone workflow requests permission only after selecting the Record tab and clicking **Start recording**. It captures up to 15 seconds, requires at least one second, creates an in-memory mono PCM-16 WAV, and submits it through the same prediction endpoint when **Analyse recording** is selected. Empty, non-finite, silent, and effectively silent captures are rejected locally before upload. Microphone APIs require a secure context: use `localhost` during development or HTTPS when deployed. The requested constraints prefer mono capture and disable automatic gain control, although browsers and devices may apply supported constraints differently.

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

## Model Limitations and Dataset Attribution

The frozen selected CNN achieved `46.6667%` accuracy on the final held-out RAVDESS test actors. Natural microphone speech may be less reliable because of accent, environment, equipment, background noise, and dataset differences. Predictions are estimates, not guaranteed emotion assessments.

This project uses the Ryerson Audio-Visual Database of Emotional Speech and Song (RAVDESS), created by Steven R. Livingstone and Frank A. Russo. Cite: Livingstone, S. R., & Russo, F. A. (2018), *PLOS ONE*, 13(5), e0196391, https://doi.org/10.1371/journal.pone.0196391. RAVDESS is available under CC BY-NC-SA 4.0; attribution and non-commercial licence obligations should be reviewed before reuse. This note is not legal advice.

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

The frozen reduced-regularization/no-dropout CNN is packaged for CPU inference together with its training-only normalization statistics. The FastAPI API supports health and WAV prediction endpoints, and the React interface supports WAV upload and browser microphone recording. Raw RAVDESS audio remains ignored and unchanged.

Model selection used actors 01-16 for training and 17-20 for validation. The selected CNN was evaluated once on held-out actors 21-24, achieving `46.6667%` test accuracy and `0.419553` macro F1. No post-test tuning, model selection, or alternative-model test evaluation was performed.
