from __future__ import annotations
import csv,hashlib,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]; CACHE=ROOT/'ml/.cache/matplotlib';CACHE.mkdir(parents=True,exist_ok=True);os.environ['MPLCONFIGDIR']=str(CACHE)
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa,librosa.display,numpy as np,soundfile as sf
from audio_preprocessing import preprocess_audio_file,load_preprocessing_config
SET={'sr':16000,'n_fft':1024,'win_length':1024,'hop_length':256,'n_mels':64,'fmin':20,'fmax':8000,'power':2.0,'n_mfcc':13}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 rows=list(csv.DictReader((ROOT/'ml/metadata/ravdess_manifest.csv').open(encoding='utf-8'))); selected=[r for r in rows if r['actor_id']=='01' and r['split']=='train' and r['statement_id']==r['repetition_id']==r['intensity_id']=='01']
 if len(selected)!=8 or {r['emotion_id'] for r in selected}!={f'{x:02d}' for x in range(1,9)}:raise ValueError('Invalid controlled selection')
 out=ROOT/'ml/reports/figures/feature_preview';out.mkdir(parents=True,exist_ok=True); config=load_preprocessing_config(); data=[]
 for r in selected:
  p=ROOT/Path(r['relative_path']);original,sr=sf.read(p,dtype='float32',always_2d=True); original=original.mean(axis=1);y,a=preprocess_audio_file(p,config); mel=librosa.feature.melspectrogram(y=y,**{k:v for k,v in SET.items() if k!='n_mfcc'});db=librosa.power_to_db(mel,ref=1.0);mfcc=librosa.feature.mfcc(S=db,n_mfcc=13)
  fig,axs=plt.subplots(2,2,figsize=(12,7),constrained_layout=True);fig.suptitle(f"{r['emotion'].title()} | Actor 01 | {r['filename']}")
  axs[0,0].plot(np.arange(len(original))/sr,original);axs[0,0].set(title='Original waveform',xlabel='Seconds',ylabel='Amplitude')
  axs[0,1].plot(np.arange(len(y))/16000,y);axs[0,1].set(title='Preprocessed waveform',xlabel='Seconds',ylabel='Amplitude')
  im=librosa.display.specshow(db,sr=16000,hop_length=256,x_axis='time',y_axis='mel',ax=axs[1,0]);axs[1,0].set_title('Log-Mel spectrogram');fig.colorbar(im,ax=axs[1,0],format='%+2.0f dB')
  im=librosa.display.specshow(mfcc,sr=16000,hop_length=256,x_axis='time',ax=axs[1,1]);axs[1,1].set_title('13 MFCC coefficients');fig.colorbar(im,ax=axs[1,1])
  image=out/f"{r['emotion']}_preview.png";fig.savefig(image,dpi=120);plt.close(fig);data.append((r,db,mfcc,image))
 fig,axs=plt.subplots(4,2,figsize=(12,14),constrained_layout=True)
 for ax,(r,db,_,_) in zip(axs.flat,data): im=librosa.display.specshow(db,sr=16000,hop_length=256,x_axis='time',y_axis='mel',ax=ax);ax.set_title(r['emotion'].title());fig.colorbar(im,ax=ax,format='%+2.0f')
 overview=out/'mel_spectrogram_overview.png';fig.savefig(overview,dpi=120);plt.close(fig)
 summary={'configuration_sha256':sha(ROOT/'ml/config/preprocessing.json'),'preview_settings':SET,'examples':[{'emotion':r['emotion'],'relative_path':r['relative_path'],'waveform_shape':[56000],'mel_shape':list(db.shape),'mfcc_shape':list(mfcc.shape),'image_path':image.relative_to(ROOT).as_posix(),'image_sha256':sha(image)} for r,db,mfcc,image in data],'overview_path':overview.relative_to(ROOT).as_posix(),'overview_sha256':sha(overview)}
 (ROOT/'ml/metadata/ravdess_feature_preview.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n',encoding='utf-8')
if __name__=='__main__':main()
