from __future__ import annotations
import json
from pathlib import Path
import librosa,numpy as np
def load_feature_config(path:Path|None=None)->dict[str,object]:
 c=json.loads((path or Path(__file__).resolve().parents[2]/'ml/config/features.json').read_text());
 if c.get('sample_rate')!=16000 or c.get('waveform_samples')!=56000 or c.get('n_mels')!=64 or c.get('n_mfcc')!=13:raise ValueError('Invalid feature config')
 return c
def check(y,sr,c):
 y=np.asarray(y)
 if sr!=c['sample_rate'] or y.ndim!=1 or y.shape!=(c['waveform_samples'],) or not np.isfinite(y).all():raise ValueError('Expected finite 56000-sample mono waveform at 16 kHz')
 return y.astype(np.float32,copy=False)
def extract_log_mel_spectrogram(y,sample_rate=16000,config=None):
 c=config or load_feature_config();y=check(y,sample_rate,c);m=librosa.feature.melspectrogram(y=y,sr=sample_rate,n_fft=c['n_fft'],win_length=c['win_length'],hop_length=c['hop_length'],window=c['window'],center=c['center'],pad_mode=c['pad_mode'],n_mels=c['n_mels'],fmin=c['fmin'],fmax=c['fmax'],power=c['power']);z=librosa.power_to_db(m,ref=c['power_to_db_ref'],amin=c['amin'],top_db=c['top_db']);return np.ascontiguousarray(z,dtype=np.float32)
def extract_mfcc_matrix(y,sample_rate=16000,config=None):
 c=config or load_feature_config();mel=extract_log_mel_spectrogram(y,sample_rate,c);z=librosa.feature.mfcc(S=mel,n_mfcc=c['n_mfcc'],dct_type=c['dct_type'],norm=c['mfcc_norm'],lifter=c['lifter']);return np.ascontiguousarray(z,dtype=np.float32)
def extract_svm_feature_vector(y,sample_rate=16000,config=None):
 c=config or load_feature_config();m=extract_mfcc_matrix(y,sample_rate,c);d=librosa.feature.delta(m,width=c['delta_width'],order=1,mode=c['delta_mode']);dd=librosa.feature.delta(m,width=c['delta_width'],order=2,mode=c['delta_mode']);z=np.concatenate([m.mean(1),m.std(1),d.mean(1),d.std(1),dd.mean(1),dd.std(1)]);return np.ascontiguousarray(z,dtype=np.float32)
