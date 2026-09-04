from __future__ import annotations
import csv,hashlib,json,os,sys
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import soundfile as sf
from audio_preprocessing import load_preprocessing_config,preprocess_audio_file
M="079884EA3724C1ADAB1C53E33D9EB669B58FCC58E4F577379BBFF807A46C398D"
def root(): return Path(__file__).resolve().parents[2]
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def raw_tree(files):
 h=hashlib.sha256()
 for p in files:
  h.update(p.relative_to(root()).as_posix().encode()+b'\0')
  with p.open('rb') as f:
   for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def main():
 try:
  r=root(); manifest=r/'ml/metadata/ravdess_manifest.csv'; config_path=r/'ml/config/preprocessing.json'; out=r/'ml/metadata/ravdess_preprocessing_validation.json'
  if digest(manifest).upper()!=M:raise ValueError('Manifest hash mismatch')
  config=load_preprocessing_config(config_path); config_hash=digest(config_path)
  rows=list(csv.DictReader(manifest.open(encoding='utf-8',newline=''))); files=[r/Path(*Path(x['relative_path']).parts) for x in rows]; before=raw_tree(files)
  splits=Counter();actions=Counter();by_split=defaultdict(Counter);by_emotion=defaultdict(Counter);paths=[];lo=float('inf');hi=float('-inf');above=below=out_samples=0
  for row,path in zip(rows,files):
   info=sf.info(path)
   if str(info.samplerate)!=row['sample_rate'] or str(info.channels)!=row['channels'] or info.subtype!=row['subtype']:raise ValueError('Manifest/header mismatch')
   y,a=preprocess_audio_file(path,config)
   if y.shape!=(56000,) or y.dtype!=np.float32 or not y.flags.c_contiguous or not np.isfinite(y).all():raise ValueError('Invalid output')
   action=a['final_action'];splits[row['split']]+=1;actions[action]+=1;by_split[row['split']][action]+=1;by_emotion[row['emotion']][action]+=1
   lo=min(lo,float(y.min()));hi=max(hi,float(y.max()));above+=int((y>1).sum());below+=int((y<-1).sum());out_samples+=int((np.abs(y)>1).sum())
   paths.append({'relative_path':row['relative_path'],'sha256':hashlib.sha256(np.ascontiguousarray(y.astype('<f4',copy=False)).tobytes()).hexdigest()})
  after=raw_tree(files)
  if before!=after:raise ValueError('Raw tree changed')
  if dict(by_split['train'])!={'padded':928,'truncated':32}:raise ValueError(f"Training actions differ: {dict(by_split['train'])}")
  report={'configuration_sha256':config_hash,'source_manifest_sha256':M,'raw_tree_sha256_before':before,'raw_tree_sha256_after':after,'total_recordings':len(rows),'counts_per_split':dict(sorted(splits.items())),'actions_overall':dict(sorted(actions.items())),'actions_per_split':{k:dict(sorted(v.items())) for k,v in sorted(by_split.items())},'actions_per_emotion':{k:dict(sorted(v.items())) for k,v in sorted(by_emotion.items())},'output_shape':[56000],'output_dtype':'float32','all_finite':True,'minimum_output_value':lo,'maximum_output_value':hi,'outputs_above_one':above,'outputs_below_negative_one':below,'total_out_of_range_samples':out_samples,'fingerprints':paths,'audio_written':False,'configuration_tuned_using_evaluation':False}
  tmp=out.with_name(out.name+'.tmp')
  if tmp.exists():raise ValueError('Temporary report exists')
  with tmp.open('w',encoding='utf-8',newline='\n') as f:json.dump(report,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
  tmp.replace(out);print('RAVDESS preprocessing validation completed successfully.')
 except Exception as e:print(f'Preprocessing validation failed: {e}',file=sys.stderr);return 1
 return 0
if __name__=='__main__':raise SystemExit(main())
