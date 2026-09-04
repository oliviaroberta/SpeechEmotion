from __future__ import annotations
import json
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf

def load_preprocessing_config(path: Path | None = None) -> dict[str, object]:
    path = path or Path(__file__).resolve().parents[2] / "ml/config/preprocessing.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {"target_sample_rate": 16000, "target_length_samples": 56000, "silence_top_db": 40, "frame_length": 2048, "trim_hop_length": 512, "resample_method": "soxr_hq", "output_dtype": "float32"}
    if any(config.get(k) != v for k, v in required.items()): raise ValueError("Invalid preprocessing configuration")
    return config

def preprocess_waveform(samples: np.ndarray, sample_rate: int, config: dict[str, object] | None = None) -> tuple[np.ndarray, dict[str, object]]:
    config = config or load_preprocessing_config()
    x = np.asarray(samples)
    if sample_rate <= 0 or x.size == 0 or not np.isfinite(x).all(): raise ValueError("Audio must be non-empty, finite, and have a positive sample rate")
    if x.ndim == 1: mono=x; channels=1
    elif x.ndim == 2 and x.shape[1] in (1,2): mono=x[:,0] if x.shape[1]==1 else x.mean(axis=1); channels=x.shape[1]
    else: raise ValueError("Audio must have mono or stereo channel dimensions")
    mono=np.asarray(mono,dtype=np.float32); original=len(mono)
    y=mono if sample_rate==16000 else librosa.resample(mono,orig_sr=sample_rate,target_sr=16000,res_type="soxr_hq")
    if y.size==0 or not np.isfinite(y).all(): raise ValueError("No meaningful finite signal remains after resampling")
    trimmed,bounds=librosa.effects.trim(y,top_db=40,frame_length=2048,hop_length=512); start,end=map(int,bounds)
    if trimmed.size==0 or not np.isfinite(trimmed).all(): raise ValueError("No meaningful finite signal remains after trimming")
    target=56000; action="unchanged"; pad_before=pad_after=0; crop_start=0
    if len(trimmed)>target:
        energy=np.concatenate(([0.0],np.cumsum(np.square(trimmed,dtype=np.float64)))); starts=np.arange(len(trimmed)-target+1); window_energy=energy[starts+target]-energy[starts]; maximum=window_energy.max(); crop_start=int(np.flatnonzero(np.isclose(window_energy,maximum,rtol=1e-12,atol=1e-12))[0]); final=trimmed[crop_start:crop_start+target]; action="truncated"
    elif len(trimmed)<target:
        missing=target-len(trimmed); pad_before=missing//2;pad_after=missing-pad_before;final=np.pad(trimmed,(pad_before,pad_after));action="padded"
    else: final=trimmed
    final=np.ascontiguousarray(final,dtype=np.float32)
    if final.shape!=(target,) or not np.isfinite(final).all(): raise ValueError("Invalid final waveform")
    return final,{"source_sample_rate":sample_rate,"source_channels":channels,"original_length":original,"resampled_length":len(y),"trim_start":start,"trim_end":end,"trimmed_length":len(trimmed),"final_action":action,"padding_before":pad_before,"padding_after":pad_after,"crop_start":crop_start,"crop_end":crop_start+len(final),"resampled_peak":float(np.abs(y).max()),"final_peak":float(np.abs(final).max())}

def preprocess_audio_file(path: Path, config: dict[str, object] | None = None) -> tuple[np.ndarray, dict[str, object]]:
    info=sf.info(path)
    if info.format!="WAV" or info.subtype!="PCM_16" or info.samplerate<=0: raise ValueError(f"Unsupported audio file: {path}")
    samples,sr=sf.read(path,dtype="float32",always_2d=True)
    return preprocess_waveform(samples,sr,config)
