import math, numpy as np, torch
from pathlib import Path
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             precision_score, recall_score,
                             log_loss)

from config.config import TASK, LOS_TOLERANCE, HIDDEN, DATASET, MLP_HIDDEN, ROOT
from model.ehrPredictModel import EHRPredictor
from ehr_tas_trainer import PatientDS, coll
import os

os.chdir(ROOT) # Change to the project directory

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

threshold = 0.5
th_path = CACHE/"best_threshold.txt"
if th_path.exists():
    threshold = float(open(th_path).read().strip())

y_true, y_prob, tot_loss = [], [], 0.0
bce = torch.nn.BCEWithLogitsLoss(reduction="sum")

with torch.no_grad():
    for x, mask, y in dl:
        logits = model(x.to(device), mask.to(device))
        loss   = bce(logits.squeeze(), y.to(device))
        tot_loss += loss.item()

        y_true.append(y.numpy())
        y_prob.append(torch.sigmoid(logits).cpu().numpy())

y_true = np.concatenate(y_true)
y_prob = np.concatenate(y_prob)
y_pred = (y_prob >= threshold).astype(int)

auroc = roc_auc_score(y_true, y_prob)
auprc = average_precision_score(y_true, y_prob)
precision = precision_score(y_true, y_pred)
recall    = recall_score(y_true, y_pred)
min_ps    = min(precision, recall)
avg_loss  = tot_loss / len(y_true)

# print(f"Threshold used: {threshold:.3f}")
print(f"AUROC  : {auroc:.4f}")
print(f"AUPRC  : {auprc:.4f}")
print(f"min(P,S): {min_ps:.4f}   "
      f"(Precision={precision:.4f}, Sensitivity={recall:.4f})")
print(f"Loss   : {avg_loss:.4f}")