"""Train the single mild-regularization and training-only masking CNN experiment."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.environ["MPLCONFIGDIR"] = str(ROOT / "ml" / ".cache" / "matplotlib")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from cnn_model import EXPECTED_CLASSES, build_cnn_regularized_augmented

DATA_DIR = ROOT / "ml/data/processed/cnn_baseline"
CONFIG_PATH = ROOT / "ml/config/cnn_regularized_augmented.json"
MODEL_PATH = ROOT / "ml/models/cnn/cnn_regularized_augmented.keras"
REPORT_PATH = ROOT / "ml/metadata/cnn_regularized_augmented_validation.json"
HISTORY_PATH = ROOT / "ml/reports/figures/cnn_regularized_augmented_history.png"
CONFUSION_PATH = ROOT / "ml/reports/figures/cnn_regularized_augmented_validation_confusion_matrix.png"
PROMOTION_RULE = {"minimum_validation_macro_f1": 0.546975, "minimum_validation_accuracy": 0.544167}

def sha256(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def load_config() -> dict[str, Any]:
    config=json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if config.get("version")!="1.0.0" or config.get("dropout_rates")!=[0.05,0.1,0.15] or config.get("dense_dropout")!=0.2 or config.get("maximum_epochs")!=80: raise ValueError("Invalid regularized augmented CNN configuration.")
    return config
def load_arrays() -> tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    arrays=(np.load(DATA_DIR/"train_log_mel.npy"),np.load(DATA_DIR/"train_labels.npy"),np.load(DATA_DIR/"validation_log_mel.npy"),np.load(DATA_DIR/"validation_labels.npy"))
    if arrays[0].shape!=(960,64,219,1) or arrays[2].shape!=(240,64,219,1) or not all(np.isfinite(array).all() for array in arrays): raise RuntimeError("Invalid prepared arrays.")
    return arrays
def weights(labels: np.ndarray) -> dict[int,float]:
    classes,counts=np.unique(labels,return_counts=True)
    if not np.array_equal(classes,np.arange(8)): raise RuntimeError("Training labels lack an emotion class.")
    return {int(c):float(len(labels)/(len(classes)*n)) for c,n in zip(classes,counts)}
class MacroF1(tf.keras.callbacks.Callback):
    def __init__(self,x:np.ndarray,y:np.ndarray)->None: super().__init__();self.x=x;self.y=y;self.records:list[dict[str,float]]=[]
    def on_epoch_end(self,epoch:int,logs:dict[str,float]|None=None)->None:
        logs=logs if logs is not None else {}; probs=self.model.predict(self.x,batch_size=32,verbose=0); logs["val_macro_f1"]=float(f1_score(self.y,np.argmax(probs,axis=1),average="macro",zero_division=0));logs["learning_rate"]=float(tf.keras.backend.get_value(self.model.optimizer.learning_rate));self.records.append({k:float(logs[k]) for k in ("loss","accuracy","val_loss","val_accuracy","val_macro_f1","learning_rate")})
def calculate_metrics(y:np.ndarray,p:np.ndarray,loss:float)->dict[str,Any]:
    a,b,c,d=precision_recall_fscore_support(y,p,labels=range(8),zero_division=0); macro=precision_recall_fscore_support(y,p,average="macro",zero_division=0);weighted=precision_recall_fscore_support(y,p,average="weighted",zero_division=0)
    return {"loss":loss,"accuracy":float(accuracy_score(y,p)),"balanced_accuracy":float(balanced_accuracy_score(y,p)),"macro":{"precision":float(macro[0]),"recall":float(macro[1]),"f1":float(macro[2])},"weighted":{"precision":float(weighted[0]),"recall":float(weighted[1]),"f1":float(weighted[2])},"prediction_distribution":{emotion:int(np.sum(p==i)) for i,emotion in enumerate(EXPECTED_CLASSES)},"per_class":{emotion:{"precision":float(q),"recall":float(r),"f1":float(s),"support":int(t)} for emotion,q,r,s,t in zip(EXPECTED_CLASSES,a,b,c,d)}}
def figures(records:list[dict[str,float]],matrix:np.ndarray)->None:
    HISTORY_PATH.parent.mkdir(parents=True,exist_ok=True);e=range(1,len(records)+1);fig,axes=plt.subplots(1,3,figsize=(15,4))
    for axis,key,title,ylabel in zip(axes,("loss","accuracy","val_macro_f1"),("Loss","Accuracy","Validation macro F1"),("Loss","Accuracy","Macro F1")):
        axis.plot(e,[r[key] for r in records],label="Training" if key in ("loss","accuracy") else "Validation macro F1")
        if key in ("loss","accuracy"): axis.plot(e,[r["val_"+key] for r in records],label="Validation")
        axis.set(title=title,xlabel="Epoch",ylabel=ylabel);axis.legend()
    fig.tight_layout();fig.savefig(HISTORY_PATH,dpi=150);plt.close(fig)
    fig,axis=plt.subplots(figsize=(10,8));im=axis.imshow(matrix,cmap="Blues");fig.colorbar(im,ax=axis,label="Recording count");axis.set(xticks=range(8),yticks=range(8),xticklabels=EXPECTED_CLASSES,yticklabels=EXPECTED_CLASSES,xlabel="Predicted emotion",ylabel="Actual emotion",title="Regularized augmented CNN validation confusion matrix");plt.setp(axis.get_xticklabels(),rotation=35,ha="right")
    for i in range(8):
        for j in range(8):axis.text(j,i,str(matrix[i,j]),ha="center",va="center",color="white" if matrix[i,j]>matrix.max()/2 else "black")
    fig.tight_layout();fig.savefig(CONFUSION_PATH,dpi=150);plt.close(fig)
def main()->None:
    config=load_config()
    if MODEL_PATH.exists():raise RuntimeError(f"Refusing to overwrite existing model: {MODEL_PATH}")
    x,y,xv,yv=load_arrays();MODEL_PATH.parent.mkdir(parents=True,exist_ok=True);tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(42);model=build_cnn_regularized_augmented();macro=MacroF1(xv,yv);callbacks=[macro,tf.keras.callbacks.ModelCheckpoint(MODEL_PATH,monitor="val_macro_f1",mode="max",save_best_only=True),tf.keras.callbacks.EarlyStopping(monitor="val_macro_f1",mode="max",patience=12,restore_best_weights=True),tf.keras.callbacks.ReduceLROnPlateau(monitor="val_macro_f1",mode="max",patience=4,factor=.5,min_lr=1e-5)]
    started=time.perf_counter();model.fit(x,y,validation_data=(xv,yv),batch_size=32,epochs=80,class_weight=weights(y),callbacks=callbacks,verbose=2);duration=time.perf_counter()-started
    prob=model.predict(xv,batch_size=32,verbose=0);pred=np.argmax(prob,axis=1);reloaded=tf.keras.models.load_model(MODEL_PATH);reload_prob=reloaded.predict(xv,batch_size=32,verbose=0);diff=float(np.max(np.abs(prob-reload_prob)))
    if not np.allclose(prob,reload_prob,rtol=0,atol=1e-7):raise RuntimeError("Reloaded probabilities differ.")
    train_prob=model.predict(x,batch_size=32,verbose=0);train_pred=np.argmax(train_prob,axis=1);train_loss=float(model.evaluate(x,y,batch_size=32,verbose=0,return_dict=True)["loss"]);val_loss=float(model.evaluate(xv,yv,batch_size=32,verbose=0,return_dict=True)["loss"]);train_metrics=calculate_metrics(y,train_pred,train_loss);val_metrics=calculate_metrics(yv,pred,val_loss);matrix=confusion_matrix(yv,pred,labels=range(8));best=int(np.argmax([r["val_macro_f1"] for r in macro.records])+1);promoted=val_metrics["macro"]["f1"]>=PROMOTION_RULE["minimum_validation_macro_f1"] and val_metrics["accuracy"]>=PROMOTION_RULE["minimum_validation_accuracy"];figures(macro.records,matrix)
    report={"configuration_hashes":{"configuration":sha256(CONFIG_PATH),"dataset_summary":sha256(ROOT/"ml/metadata/cnn_dataset_summary.json")},"actor_ids":{"train":[f"{i:02d}" for i in range(1,17)],"validation":[f"{i:02d}" for i in range(17,21)],"test_actors_21_to_24_accessed":False},"parameter_count":int(model.count_params()),"epochs_completed":len(macro.records),"best_epoch":best,"training_duration_seconds":duration,"final_learning_rate":float(tf.keras.backend.get_value(model.optimizer.learning_rate)),"epoch_history":macro.records,"training_metrics":train_metrics,"validation_metrics":val_metrics,"confusion_matrix":matrix.tolist(),"overfitting_macro_f1_gap":train_metrics["macro"]["f1"]-val_metrics["macro"]["f1"],"comparison":{"leading_cnn_accuracy":.5541666666666667,"leading_cnn_macro_f1":.5369751467634547,"candidate_accuracy":val_metrics["accuracy"],"candidate_macro_f1":val_metrics["macro"]["f1"]},"promotion_rule":PROMOTION_RULE,"promoted":promoted,"selected_leading_model":"cnn_regularized_augmented" if promoted else "cnn_reduced_regularization","model_relative_path":MODEL_PATH.relative_to(ROOT).as_posix(),"history_figure_relative_path":HISTORY_PATH.relative_to(ROOT).as_posix(),"confusion_figure_relative_path":CONFUSION_PATH.relative_to(ROOT).as_posix(),"reload_probabilities_match":True,"reload_probability_max_abs_difference":diff}
    REPORT_PATH.write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8");print(f"Regularized augmented CNN completed in {duration:.2f} seconds.")
if __name__=="__main__":main()
