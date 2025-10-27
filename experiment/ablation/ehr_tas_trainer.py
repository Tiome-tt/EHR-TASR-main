import math, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             average_precision_score, f1_score,
                             mean_absolute_error,
                             precision_score, recall_score)
import torch, torch.nn as nn, torch.optim as optim

from config.config import (TASK, EPOCHS, BATCH_SIZE, LR, HIDDEN,
                           LOS_TOLERANCE, DATASET, MLP_HIDDEN, ROOT)
from model.ehrPredictModel import EHRPredictor
import os

os.chdir(ROOT)

# ---------- config ----------
TASK        = TASK.lower()
EPOCHS      = EPOCHS
BATCH_SIZE  = BATCH_SIZE
LR          = LR
HIDDEN      = HIDDEN
MLP_HIDDEN  = MLP_HIDDEN
LOS_TOL     = LOS_TOLERANCE
TIME_AWARE_SCORE = True  # Whether to remove the time-aware score column
THRESHOLD_CLASSIFICATION = False  # Whether to set the classification threshold to 0.5
CLASSIFICATION_THRESHOLD = 0.5

SPLIT_DIR = Path(f"data/{DATASET}/preprocessed/splits")
CACHE_DIR = Path(f"data/{DATASET}/preprocessed/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Data processing
class PatientDS(torch.utils.data.Dataset):
    def __init__(self, path):
        d = np.load(path, allow_pickle=True)
        self.x, self.len, self.y = d["records"], d["rec_len"], d["label"]
    def __len__(self): return len(self.y)
    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]), int(self.len[i]), torch.tensor(self.y[i])

def coll(batch):
    xs,lens,ys = zip(*batch)
    lens = torch.tensor(lens)
    pad  = torch.nn.utils.rnn.pad_sequence(xs, True)
    mask = (torch.arange(pad.size(1))[None] < lens[:, None])
    return pad, mask, torch.stack(ys)

def main():

    # load data
    def load_split(split: str):
        X = pd.read_csv(SPLIT_DIR / f"{split}_features.csv")
        y = pd.read_csv(SPLIT_DIR / f"{split}_labels.csv")[["PatientID",
                                                           "Outcome", "LOS",
                                                           "Readmission"]]
        return X, y.drop_duplicates("PatientID")

    # wheather to load cache file
    SAMPLE_PATH = SPLIT_DIR / "train_features.csv"

    ALL_COLS = pd.read_csv(SAMPLE_PATH, nrows=0).columns
    if DATASET == "mimic3":
        DROP_COLS = [f"D{i}" for i in range(24)]
    elif DATASET == "mimic4":
        DROP_COLS = [f"D{i}" for i in range(12)]

    if not TIME_AWARE_SCORE:
        DROP_COLS = []

    NUM_COLS = (
        ALL_COLS
        .difference(["PatientID"], sort=False)
        .difference(DROP_COLS, sort=False)
        .tolist()
    )

    SCALER_PATH = CACHE_DIR / "scaler.pkl"

    if SCALER_PATH.exists():
        scaler = joblib.load(SCALER_PATH)
        print("√  find scaler.pkl, loading (identity transformer)...")
    else:
        scaler = FunctionTransformer(validate=False)
        joblib.dump(scaler, SCALER_PATH)
        print("×  scaler.pkl not found, using identity transformer.")

    def make_npz(split: str):
        X, y = load_split(split)
        X[NUM_COLS] = scaler.transform(X[NUM_COLS])

        seqs,lens,labels = [],[],[]
        for pid, g in X.groupby("PatientID"):
            g = g.sort_values(["VisitID", "RecordTime"])
            seqs.append(g[NUM_COLS].to_numpy(np.float32))
            lens.append(len(g))
            row = y[y.PatientID == pid].iloc[0]
            if   TASK=="outcome":     labels.append(float(row["Outcome"]))
            elif TASK=="readmission": labels.append(float(row["Readmission"]))
            elif TASK=="los":         labels.append(float(row["LOS"]))
            else:
                labels.append(row[["Outcome","LOS","Readmission"]].to_numpy(np.float32))
        np.savez_compressed(CACHE_DIR/f"{split}.npz",
                            records=np.array(seqs, dtype=object),
                            rec_len=np.array(lens),
                            label=np.array(labels, dtype=np.float32))

    for s in ("train", "val", "test"):
        if not (CACHE_DIR / f"{s}.npz").exists():
            print(f"×  {s}.npz not found, generating...")
            make_npz(s)
        else:
            print(f"√  Found {s}.npz — Reusing directly")

    # ---------- DataLoader ----------
    ds_tr = PatientDS(CACHE_DIR/"train.npz")
    ds_va = PatientDS(CACHE_DIR/"val.npz")
    dl_tr = torch.utils.data.DataLoader(ds_tr, BATCH_SIZE, True,  collate_fn=coll)
    dl_va = torch.utils.data.DataLoader(ds_va, BATCH_SIZE, False, collate_fn=coll)

    # ---------- Model ----------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dim = 3 if TASK == "multitask" else 1
    model = EHRPredictor(ds_tr.x[0].shape[1], HIDDEN,
                                mlp_hidden=MLP_HIDDEN,
                                out_dim=out_dim).to(device)

    # ---------- loss function ----------
    if TASK in {"outcome","readmission"}:
        pos_w = (1. / ds_tr.y.mean()) - 1.
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device))
    elif TASK == "los":
        criterion = nn.MSELoss()
    else:   # multitask
        pos_w = (1. / ds_tr.y[:, [0, 2]].mean(0)) - 1.
        bce1 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w[0]], device=device))
        bce2 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w[1]], device=device))
        mse  = nn.MSELoss()
        def criterion(p, t):
            return (bce1(p[:, 0], t[:, 0]) +
                    mse(p[:, 1], t[:, 1]) +
                    bce2(p[:, 2], t[:, 2]))

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    # optimizer = optim.AdamW(model.parameters(), lr=LR)

    # ---------- define the best threshold ----------
    def best_minps_threshold(y_true, y_prob):
        ths = np.linspace(0.05, 0.95, 19)
        minps_scores = []
        for t in ths:
            pred_binary = y_prob >= t
            precision = precision_score(y_true, pred_binary, zero_division=0)
            recall = recall_score(y_true, pred_binary, zero_division=0)
            minps = min(precision, recall)
            minps_scores.append(minps)
        i = int(np.argmax(minps_scores))
        return ths[i], minps_scores[i]

    def run(loader, train):
        model.train(train)
        ys, ps, tot = [], [], 0.
        with torch.set_grad_enabled(train):
            for x, mask, y in loader:
                x, mask, y = x.to(device), mask.to(device), y.to(device)
                pred = model(x, mask)
                loss = criterion(pred, y)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                tot += loss.item() * len(y)
                ys.append(y.cpu().numpy())
                ps.append(pred.detach().cpu().numpy())

        ys = np.concatenate(ys)
        ps = np.concatenate(ps)
        prob = torch.sigmoid(torch.tensor(ps)).numpy()

        if THRESHOLD_CLASSIFICATION and not train:
            best_thresh, _ = best_minps_threshold(ys, prob)
            threshold = best_thresh
        else:
            threshold = CLASSIFICATION_THRESHOLD

        precision = precision_score(ys, prob >= threshold, zero_division=0)
        recall = recall_score(ys, prob >= threshold, zero_division=0)
        min_ps = min(precision, recall)

        met = dict(
            auroc=roc_auc_score(ys, prob),
            auprc=average_precision_score(ys, prob),
            minps=min_ps,
            loss=tot / len(loader.dataset),
        )
        return met, ys, prob

    STOP_KEY = "minps"
    best, patience = -1, 0

    for ep in range(1, EPOCHS + 1):
        tr, _, _ = run(dl_tr, True)
        va, ys_va, prob_va = run(dl_va, False)

        print(f"E{ep:02d}",
              {k: f'{v:.3f}' for k, v in tr.items()},
              "||",
              {k: f'{v:.3f}' for k, v in va.items()})

        if va[STOP_KEY] > best:
            best, patience = va[STOP_KEY], 0

            torch.save(model.state_dict(),
                       CACHE_DIR / f"best_model_{DATASET}_{TASK}.pt")

            best_thresh, _ = best_minps_threshold(ys_va, prob_va)
            with open(CACHE_DIR / "best_threshold.txt", "w") as f:
                f.write(str(float(best_thresh)))
        else:
            patience += 1
            if patience >= 3:
                print(f"Early stopped. Best {STOP_KEY} = {best:.3f}")
                break


if __name__ == "__main__":
    main()

