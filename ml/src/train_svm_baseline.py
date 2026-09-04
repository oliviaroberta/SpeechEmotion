from __future__ import annotations
import csv,hashlib,json,os
from pathlib import Path
import joblib,numpy as np
ROOT=Path(__file__).resolve().parents[2];os.environ['MPLCONFIGDIR']=str(ROOT/'ml/.cache/matplotlib')
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score,balanced_accuracy_score,classification_report,confusion_matrix,precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from audio_preprocessing import preprocess_audio_file
from audio_features import extract_svm_feature_vector
def root():return Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 r=root();cfg=json.loads((r/'ml/config/svm_baseline.json').read_text());labels=cfg['label_order'];rows=list(csv.DictReader((r/'ml/metadata/ravdess_manifest.csv').open(encoding='utf-8')));tr=[x for x in rows if x['split']=='train'];va=[x for x in rows if x['split']=='validation']
 def feats(rs):return np.array([extract_svm_feature_vector(preprocess_audio_file(r/Path(*Path(x['relative_path']).parts))[0]) for x in rs])
 X,Y=feats(tr),feats(va);y=np.array([x['emotion'] for x in tr]);yv=np.array([x['emotion'] for x in va]);pipe=Pipeline([('scaler',StandardScaler()),('svc',SVC(kernel='rbf',C=10,gamma='scale',class_weight='balanced',probability=True,random_state=42))]);pipe.fit(X,y);pred=pipe.predict(Y);prob=pipe.predict_proba(Y);cm=confusion_matrix(yv,pred,labels=labels);p,r_,f,s=precision_recall_fscore_support(yv,pred,labels=labels,zero_division=0)
 bundle={'pipeline':pipe,'label_order':labels,'preprocessing_configuration_hash':sha(r/'ml/config/preprocessing.json'),'feature_configuration_hash':sha(r/'ml/config/features.json'),'svm_configuration_hash':sha(r/'ml/config/svm_baseline.json'),'expected_waveform_shape':[56000],'expected_feature_shape':[78],'model_version':'1.0.0','status':'baseline_candidate'};model=r/'ml/models/svm/baseline_svm.joblib';model.parent.mkdir(parents=True,exist_ok=True);joblib.dump(bundle,model)
 reload=joblib.load(model);assert np.array_equal(pred,reload['pipeline'].predict(Y)) and np.allclose(prob,reload['pipeline'].predict_proba(Y),atol=1e-12)
 samples=[]
 for label in labels:
  i=next(i for i,row in enumerate(va) if row['emotion']==label);samples.append({'filename':va[i]['filename'],'actor_id':va[i]['actor_id'],'actual_emotion':label,'predicted_emotion':pred[i],'correct':bool(pred[i]==label),'predicted_probability':float(prob[i,list(pipe.classes_).index(pred[i])]),'probabilities':{z:float(prob[i,list(pipe.classes_).index(z)]) for z in labels}})
 figure=r/'ml/reports/figures/svm_baseline_validation_confusion_matrix.png';figure.parent.mkdir(parents=True,exist_ok=True);fig,ax=plt.subplots(figsize=(9,8));im=ax.imshow(cm,cmap='Blues');fig.colorbar(im,ax=ax);ax.set(xticks=range(8),yticks=range(8),xticklabels=labels,yticklabels=labels,xlabel='Predicted emotion',ylabel='Actual emotion',title='SVM baseline validation confusion matrix');plt.setp(ax.get_xticklabels(),rotation=35,ha='right');
 for i in range(8):
  for j in range(8):ax.text(j,i,str(cm[i,j]),ha='center',va='center')
 fig.tight_layout();fig.savefig(figure,dpi=150);plt.close(fig)
 report={'configuration_hashes':{k:bundle[k] for k in bundle if k.endswith('hash')},'train_actor_ids':[f'{i:02d}' for i in range(1,17)],'validation_actor_ids':[f'{i:02d}' for i in range(17,21)],'test_actors_excluded':True,'feature_shapes':{'train':list(X.shape),'validation':list(Y.shape)},'metrics':{'accuracy':accuracy_score(yv,pred),'balanced_accuracy':balanced_accuracy_score(yv,pred),'macro':dict(zip(['precision','recall','f1'],precision_recall_fscore_support(yv,pred,average='macro',zero_division=0)[:3])),'weighted':dict(zip(['precision','recall','f1'],precision_recall_fscore_support(yv,pred,average='weighted',zero_division=0)[:3]))},'per_class':{z:{'precision':float(a),'recall':float(b),'f1':float(c),'support':int(d)} for z,a,b,c,d in zip(labels,p,r_,f,s)},'confusion_matrix':cm.tolist(),'correct':int((pred==yv).sum()),'incorrect':int((pred!=yv).sum()),'sample_predictions':samples,'model_relative_path':model.relative_to(r).as_posix(),'confusion_matrix_relative_path':figure.relative_to(r).as_posix()}
 out=r/'ml/metadata/svm_baseline_validation.json';out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print('SVM baseline completed')
if __name__=='__main__':main()
