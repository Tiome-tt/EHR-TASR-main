import math, numpy as np, torch
from pathlib import Path
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             average_precision_score, f1_score,
                             mean_absolute_error)

from config.config import TASK, LOS_TOLERANCE, HIDDEN, DATASET, MLP_HIDDEN
from model.ehrPredictModel import EHRPredictor
from ehr_tas_trainer import PatientDS, coll           # 直接复用 Dataset & collate

TASK    = TASK.lower()
DATASET = DATASET
LOS_TOL = LOS_TOLERANCE
MLP_HIDDEN = MLP_HIDDEN

CACHE = Path(f"data/{DATASET}/preprocessed/cache")
device= "cuda" if torch.cuda.is_available() else "cpu"

ds = PatientDS(CACHE/"test.npz")
dl = torch.utils.data.DataLoader(ds,32,False,collate_fn=coll)

model = EHRPredictor(ds.x[0].shape[1], HIDDEN,
                            mlp_hidden=MLP_HIDDEN, out_dim=1).to(device)
model.load_state_dict(torch.load(CACHE/f"best_model_{DATASET}_{TASK}.pt",
                                 map_location=device))
model.eval()

# 若保存了阈值就读；否则默认为 0.5
threshold = 0.5
th_path = CACHE/"best_threshold.txt"
if th_path.exists():
    threshold = float(open(th_path).read().strip())

y_true,y_prob = [],[]
with torch.no_grad():
    for x,mask,y in dl:
        logit = model(x.to(device), mask.to(device))
        y_true.append(y.numpy())
        y_prob.append(torch.sigmoid(logit).cpu().numpy())
y_true = np.concatenate(y_true); y_prob = np.concatenate(y_prob)
y_pred = (y_prob >= threshold).astype(int)

print(f"Threshold used: {threshold:.3f}")
print("Accuracy :", accuracy_score(y_true, y_pred))
print("AUROC    :", roc_auc_score(y_true, y_prob))
print("AUPRC    :", average_precision_score(y_true, y_prob))
print("F1 Score :", f1_score(y_true, y_pred))
