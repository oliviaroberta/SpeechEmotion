from __future__ import annotations

import os
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MPLCONFIGDIR = REPO_ROOT / "ml" / ".cache" / "matplotlib"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPLCONFIGDIR)
os.environ.setdefault("PYTHONHASHSEED", "42")

import joblib
import librosa
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import soundfile
import tensorflow as tf
from sklearn import __version__ as sklearn_version
from sklearn.svm import SVC


def main() -> int:
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")

    versions = {
        "tensorflow": tf.__version__,
        "librosa": librosa.__version__,
        "scikit-learn": sklearn_version,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "matplotlib": matplotlib.__version__,
        "seaborn": sns.__version__,
        "soundfile": soundfile.__version__,
        "joblib": joblib.__version__,
    }
    for package_name, version in versions.items():
        print(f"{package_name} version: {version}")

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    print(f"Deterministic seed set to: {seed}")

    cpu_devices = tf.config.list_physical_devices("CPU")
    gpu_devices = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow CPU devices: {cpu_devices}")
    print(f"TensorFlow GPU devices: {gpu_devices}")
    print("Zero GPUs is expected on native Windows in this project setup.")
    if not cpu_devices:
        raise RuntimeError("TensorFlow did not detect a CPU device.")

    sample_rate = 22050
    duration_seconds = 0.5
    time_axis = np.linspace(0.0, duration_seconds, int(sample_rate * duration_seconds), endpoint=False)
    audio_signal = (0.5 * np.sin(2 * np.pi * 440.0 * time_axis)).astype(np.float32)
    mfcc = librosa.feature.mfcc(y=audio_signal, sr=sample_rate, n_mfcc=13)
    print(f"MFCC shape: {mfcc.shape}")
    if mfcc.shape[0] != 13 or mfcc.shape[1] <= 0:
        raise RuntimeError(f"Unexpected MFCC shape: {mfcc.shape}")

    features = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    labels = np.array([0, 0, 1, 1], dtype=np.int32)
    svm_model = SVC(kernel="linear", C=1.0, random_state=seed)
    svm_model.fit(features, labels)
    svm_prediction = svm_model.predict(np.array([[0.9, 0.2]], dtype=np.float32))
    print(f"SVM test prediction: {svm_prediction.tolist()}")
    if svm_prediction.shape != (1,):
        raise RuntimeError(f"Unexpected SVM prediction shape: {svm_prediction.shape}")

    keras_model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(4,)),
            tf.keras.layers.Dense(3, activation="relu"),
            tf.keras.layers.Dense(2, activation="softmax"),
        ]
    )
    tf_input = tf.constant([[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]], dtype=tf.float32)
    tf_output = keras_model(tf_input, training=False)
    print(f"TensorFlow forward-pass output shape: {tuple(tf_output.shape)}")
    if tuple(tf_output.shape) != (2, 2):
        raise RuntimeError(f"Unexpected TensorFlow output shape: {tuple(tf_output.shape)}")

    figures_dir = REPO_ROOT / "ml" / "reports" / "figures"
    temp_figure_path = figures_dir / "verify_ml_environment_temp.png"
    if temp_figure_path.exists():
        temp_figure_path.unlink()

    figure = plt.figure(figsize=(4, 2))
    try:
        axis = figure.add_subplot(111)
        axis.plot(time_axis[:200], audio_signal[:200])
        axis.set_title("Synthetic Waveform Check")
        axis.set_xlabel("Sample")
        axis.set_ylabel("Amplitude")
        figure.tight_layout()
        figure.savefig(temp_figure_path)
        print(f"Matplotlib temp file created: {temp_figure_path}")
        if not temp_figure_path.exists():
            raise RuntimeError("Matplotlib test figure was not created.")
    finally:
        plt.close(figure)
        if temp_figure_path.exists():
            temp_figure_path.unlink()

    print(f"Matplotlib temp file exists after cleanup: {temp_figure_path.exists()}")
    if temp_figure_path.exists():
        raise RuntimeError("Matplotlib temp figure was not deleted.")

    print("ML environment verification completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
