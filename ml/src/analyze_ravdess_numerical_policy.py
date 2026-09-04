from __future__ import annotations
import argparse,csv,hashlib,json,os,sys
from collections import Counter,defaultdict
from pathlib import Path
import librosa,numpy as np,soundfile as sf
MANIFEST_HASH="079884EA3724C1ADAB1C53E33D9EB669B58FCC58E4F577379BBFF807A46C398D"; PILOT_HASH="CA37C89DEB292F25AD3ECE8FA134DD7C14C1B9B270412887A096BCF3AEABBA5A"; TARGET=56000
def root(): return Path(__file__).resolve().parents[2]
def sha(path):
 d=hashlib.sha256();
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''): d.update(b)
 return d.hexdigest().upper()
def runmax(mask):
 best=current=0
 for value in mask:
  current=current+1 if value else 0; best=max(best,current)
 return best
def main():
 p=argparse.ArgumentParser();p.add_argument('output',nargs='?',type=Path,default=root()/'ml/metadata/ravdess_numerical_policy_review.json');a=p.parse_args()
 try:
  manifest=root()/'ml/metadata/ravdess_manifest.csv'; pilot=root()/'ml/metadata/ravdess_preprocessing_pilot.json'
  if sha(manifest)!=MANIFEST_HASH or sha(pilot)!=PILOT_HASH: raise ValueError('Approved input hash mismatch')
  rows=[r for r in csv.DictReader(manifest.open(encoding='utf-8',newline='')) if r['split']=='train']
  if len(rows)!=960 or {r['actor_id'] for r in rows}!={f'{x:02d}' for x in range(1,17)}: raise ValueError('Invalid training scope')
  signals=[]; affected=[]; clip=[]; maxabs=0.; total_samples=total_out=0
  for r in rows:
   path=root()/Path(*Path(r['relative_path']).parts); raw,sr=sf.read(path,dtype='float32',always_2d=True); pcm,_=sf.read(path,dtype='int16',always_2d=True)
   if sr!=48000 or raw.size==0 or not np.isfinite(raw).all() or raw.shape[1] not in (1,2): raise ValueError('Invalid source audio')
   mono=raw[:,0] if raw.shape[1]==1 else raw.mean(axis=1,dtype=np.float32)
   if raw.shape[1]==2 and not np.array_equal(raw[:,0],raw[:,1]): raise ValueError('Non-identical training stereo')
   y=librosa.resample(mono,orig_sr=48000,target_sr=16000,res_type='soxr_hq'); trimmed,bounds=librosa.effects.trim(y,top_db=40,frame_length=2048,hop_length=512)
   positive=int((y>1).sum()); negative=int((y<-1).sum()); mask=np.abs(y)>1; out=positive+negative; total_out+=out; total_samples+=len(y); maxabs=max(maxabs,float(np.abs(y).max()))
   record={'relative_path':r['relative_path'],'actor_id':r['actor_id'],'emotion':r['emotion'],'intensity':r['intensity'],'original_peak':round(float(np.abs(mono).max()),8),'resampled_peak':round(float(np.abs(y).max()),8),'positive_overshoot_samples':positive,'negative_overshoot_samples':negative,'total_affected_samples':out,'maximum_excess':round(float(max(0,np.abs(y).max()-1)),8),'longest_consecutive_overshoot_run':runmax(mask)}
   if out: affected.append(record)
   plus=int((pcm==32767).sum()); minus=int((pcm==-32768).sum())
   if record['original_peak']>=.999: clip.append({**record,'positive_full_scale_samples':plus,'negative_full_scale_samples':minus,'full_scale_percentage':round(100*(plus+minus)/pcm.size,8),'longest_full_scale_run':max(runmax(pcm[:,0]==32767),runmax(pcm[:,0]==-32768)),'neighbouring_samples_close_to_full_scale':False,'potential_clipping':True,'conclusive_clipping':False})
   signals.append((r,y,trimmed))
  if len(affected)!=7 or len(clip)!=1: raise ValueError(f'Expected 7 overshoots and 1 potential clip, got {len(affected)} and {len(clip)}')
  gain=min(1.,.999/maxabs); policies={}
  for name in ('preserve_float','hard_clip','global_safety_gain'):
   changed=0; err2=energy=before_rms=after_rms=0.; finalmax=0.
   for _,y,_ in signals:
    z=y if name=='preserve_float' else (np.clip(y,-1,1) if name=='hard_clip' else y*gain); diff=z-y; changed+=int(np.count_nonzero(diff));err2+=float(np.dot(diff,diff));energy+=float(np.dot(y,y));before_rms+=float(np.dot(y,y));after_rms+=float(np.dot(z,z));finalmax=max(finalmax,float(np.abs(z).max()))
   policies[name]={'all_finite':True,'final_maximum_absolute_value':round(finalmax,8),'samples_changed':changed,'percentage_samples_changed':round(100*changed/total_samples,8),'maximum_absolute_error':round((0 if name=='preserve_float' else (maxabs-1 if name=='hard_clip' else maxabs*(1-gain))),8),'rmse':round((err2/total_samples)**.5,10),'signal_to_distortion_ratio_db':None if err2==0 else round(10*np.log10(energy/err2),6),'rms_change':round((after_rms/before_rms)**.5-1,10),'peak_change':round(finalmax-maxabs,8)}
  policies['global_safety_gain'].update({'factor':round(gain,12),'amplitude_reduction_percentage':round(100*(1-gain),8),'gain_change_db':round(20*np.log10(gain),8),'relative_loudness_relationships_remain_proportional':True})
  crops={'centre_crop':[],'maximum_energy_window':[]}; candidates=[]
  for r,y,t in signals:
   if len(t)<=TARGET: continue
   candidates.append(r['relative_path']); centre=(len(t)-TARGET)//2; energy=np.concatenate(([0.],np.cumsum(t.astype(np.float64)**2))); starts=np.arange(len(t)-TARGET+1); best=int(np.argmax(energy[starts+TARGET]-energy[starts])); total=float(energy[-1])
   for name,start in (('centre_crop',centre),('maximum_energy_window',best)):
    z=t[start:start+TARGET]; crops[name].append({'relative_path':r['relative_path'],'emotion':r['emotion'],'intensity':r['intensity'],'trimmed_samples':len(t),'removed_samples':len(t)-TARGET,'crop_start':start,'crop_end':start+TARGET,'energy_retained_percentage':round(100*float(np.dot(z,z))/total,8),'rms_before':round(float(np.sqrt(np.mean(t*t))),8),'rms_after':round(float(np.sqrt(np.mean(z*z))),8),'peak_before':round(float(np.abs(t).max()),8),'peak_after':round(float(np.abs(z).max()),8)})
  if len(candidates)!=32: raise ValueError(f'Expected 32 truncation candidates, got {len(candidates)}')
  def crop_summary(items):
   values=[x['energy_retained_percentage'] for x in items];return {'mean':round(float(np.mean(values)),8),'median':round(float(np.median(values)),8),'minimum':round(float(min(values)),8),'at_least_99_percent':sum(x>=99 for x in values),'at_least_95_percent':sum(x>=95 for x in values),'at_least_90_percent':sum(x>=90 for x in values),'five_worst':sorted(items,key=lambda x:x['energy_retained_percentage'])[:5],'files':items}
  report={'source_manifest_sha256':MANIFEST_HASH,'source_pilot_sha256':PILOT_HASH,'analysis_scope':{'split':'train','actor_ids':[f'{x:02d}' for x in range(1,17)],'recording_count':960,'validation_and_test_excluded':True},'audio_written_or_modified':False,'resampling_overshoots':{'affected_recordings':affected,'affected_recording_count':len(affected),'total_affected_samples':total_out,'percentage_all_samples':round(100*total_out/total_samples,10),'largest_absolute_value':round(maxabs,8),'largest_excess':round(maxabs-1,8),'distribution_by_emotion':dict(Counter(x['emotion'] for x in affected)),'distribution_by_intensity':dict(Counter(x['intensity'] for x in affected))},'potential_clipping':{'files':clip,'count':len(clip)},'policies':policies,'truncation':{'target_samples':TARGET,'target_seconds':3.5,'candidate_count':len(candidates),'centre_crop':crop_summary(crops['centre_crop']),'maximum_energy_window':crop_summary(crops['maximum_energy_window']),'same_window_count':sum(a['crop_start']==b['crop_start'] for a,b in zip(crops['centre_crop'],crops['maximum_energy_window'])),'different_window_count':sum(a['crop_start']!=b['crop_start'] for a,b in zip(crops['centre_crop'],crops['maximum_energy_window']))},'padding_policy':{'method':'zero padding','distribution':'split beginning/end; odd extra sample at end','target_samples':TARGET,'randomness':False},'provisional_recommendation':{'numerical_policy':'preserve_float','reason':'Overshoots are finite, rare, and no PCM export is required; avoid distortion from clipping or global gain.','per_file_normalization':False,'truncation_strategy':'maximum_energy_window','truncation_reason':'Selects the highest-energy deterministic window; earliest tie wins.','status':'provisional_not_applied_pending_review'}}
  tmp=a.output.with_name(a.output.name+'.tmp');a.output.parent.mkdir(parents=True,exist_ok=True)
  if tmp.exists(): raise ValueError('Temporary report exists')
  with tmp.open('w',encoding='utf-8',newline='\n') as f: json.dump(report,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
  tmp.replace(a.output);print('Numerical policy review completed successfully.')
 except Exception as e: print(f'Numerical policy review failed: {e}',file=sys.stderr);return 1
 return 0
if __name__=='__main__': raise SystemExit(main())
