from __future__ import annotations
import argparse,json,sys
from pathlib import Path
M="079884EA3724C1ADAB1C53E33D9EB669B58FCC58E4F577379BBFF807A46C398D";P="CA37C89DEB292F25AD3ECE8FA134DD7C14C1B9B270412887A096BCF3AEABBA5A"
def root(): return Path(__file__).resolve().parents[2]
def main():
 a=argparse.ArgumentParser();a.add_argument('report',nargs='?',type=Path,default=root()/'ml/metadata/ravdess_numerical_policy_review.json');x=a.parse_args()
 try:
  data=json.load(x.report.open(encoding='utf-8')); text=json.dumps(data)
  if 'timestamp' in text.lower() or 'C:\\' in text or 'olivia.dogbey' in text.lower() or '\\\\' in text: raise ValueError('Forbidden local content')
  if data['source_manifest_sha256']!=M or data['source_pilot_sha256']!=P: raise ValueError('Input hash mismatch')
  scope=data['analysis_scope']
  if scope['recording_count']!=960 or scope['actor_ids']!=[f'{i:02d}' for i in range(1,17)] or not scope['validation_and_test_excluded']: raise ValueError('Invalid training scope')
  if data['audio_written_or_modified'] or data['resampling_overshoots']['affected_recording_count']!=7 or len(data['resampling_overshoots']['affected_recordings'])!=7: raise ValueError('Overshoot reconciliation failed')
  if data['potential_clipping']['count']!=1 or len(data['potential_clipping']['files'])!=1: raise ValueError('Clipping reconciliation failed')
  for policy in ('preserve_float','hard_clip','global_safety_gain'):
   if policy not in data['policies'] or not data['policies'][policy]['all_finite']: raise ValueError('Invalid policy comparison')
  t=data['truncation']
  if t['candidate_count']!=32 or len(t['centre_crop']['files'])!=32 or len(t['maximum_energy_window']['files'])!=32: raise ValueError('Truncation count mismatch')
  for strategy in ('centre_crop','maximum_energy_window'):
   for item in t[strategy]['files']:
    if item['crop_end']-item['crop_start']!=56000: raise ValueError('Invalid crop window')
  if data['provisional_recommendation']['numerical_policy'] not in data['policies'] or data['provisional_recommendation']['per_file_normalization']: raise ValueError('Unsupported recommendation')
 except Exception as e: print(f'Numerical policy verification failed: {e}',file=sys.stderr);return 1
 print('RAVDESS numerical policy verification completed successfully.');return 0
if __name__=='__main__': raise SystemExit(main())
