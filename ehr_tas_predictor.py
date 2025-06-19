import math, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             average_precision_score, f1_score,
                             mean_absolute_error)
import torch, torch.nn as nn, torch.optim as optim

from config.config import (TASK, EPOCHS, BATCH_SIZE, LR, HIDDEN,
                           LOS_TOLERANCE, DATASET, MLP_HIDDEN)
from model.ehrPredictModel import EHRPredictor

# ---------- config ----------
TASK        = TASK.lower()
EPOCHS      = EPOCHS
BATCH_SIZE  = BATCH_SIZE
LR          = LR
HIDDEN      = HIDDEN
MLP_HIDDEN  = MLP_HIDDEN
LOS_TOL     = LOS_TOLERANCE

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
    NUM_COLS = pd.read_csv(SAMPLE_PATH, nrows=0).columns.difference(["PatientID"]).tolist()

    SCALER_PATH = CACHE_DIR / "scaler.pkl"
    if SCALER_PATH.exists():
        scaler = joblib.load(SCALER_PATH)
        print("√  find scaler.pkl, loading...")
    else:
        print("×  scaler.pkl does not exist, refitting...")
        train_X, _ = load_split("train")
        scaler = StandardScaler().fit(train_X[NUM_COLS])
        joblib.dump(scaler, SCALER_PATH)

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

    optimizer = optim.AdamW(model.parameters(), lr=LR)

    # ---------- define the best threshold ----------
    def best_f1_threshold(y_true, y_prob):
        ths = np.linspace(0.05, 0.95, 19)
        f1s = [f1_score(y_true, y_prob >= t) for t in ths]
        i   = int(np.argmax(f1s))
        return ths[i], f1s[i]

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
                    optimizer.step()
                tot += loss.item() * len(y)
                ys.append(y.cpu().numpy())
                ps.append(pred.detach().cpu().numpy())
        ys = np.concatenate(ys); ps = np.concatenate(ps)
        if TASK in {"outcome","readmission"}:
            met = dict(zip(
                ["acc", "auroc", "auprc", "f1"],
                [accuracy_score(ys, (torch.sigmoid(torch.tensor(ps)) >= 0.5).numpy()),
                 roc_auc_score(ys, torch.sigmoid(torch.tensor(ps)).numpy()),
                 average_precision_score(ys, torch.sigmoid(torch.tensor(ps)).numpy()),
                 f1_score(ys, (torch.sigmoid(torch.tensor(ps)) >= 0.5).numpy())]))
        elif TASK == "los":
            met = dict(mae=mean_absolute_error(ys, ps),
                       rmse=math.sqrt(((ys - ps) ** 2).mean()),
                       acc1=(np.abs(ys - ps) <= LOS_TOL).mean())
        else:
            prob = torch.sigmoid(torch.tensor(ps[:, 0])).numpy()
            met = dict(out_f1=f1_score(ys[:, 0], prob >= 0.5))
        met["loss"] = tot / len(loader.dataset)
        return met, ys, ps

    # ---------- train ----------
    STOP_KEY = "best_f1" if TASK in {"outcome", "readmission"} else "acc1"
    best, patience = -1, 0
    for ep in range(1, EPOCHS + 1):
        tr, _, _ = run(dl_tr, True)
        va, yv, pv = run(dl_va, False)

        if TASK in {"outcome", "readmission"}:
            prob = torch.sigmoid(torch.tensor(pv)).numpy()
            thr, f1_val = best_f1_threshold(yv, prob)
            va["best_f1"] = f1_val

        print(f"E{ep:02d}", {k:f'{v:.3f}' for k, v in tr.items()},
              "||", {k:f'{v:.3f}' for k, v in va.items()})

        if va.get(STOP_KEY, -1) > best:
            best, patience = va[STOP_KEY], 0
            torch.save(model.state_dict(),
                       CACHE_DIR / f"best_model_{DATASET}_{TASK}.pt")
            if TASK in {"outcome", "readmission"}:
                with open(CACHE_DIR / "best_threshold.txt", "w") as f:
                    f.write(f"{thr:.4f}")
        else:
            patience += 1
            if patience >= 3:
                print(f"Early stopped. Best {STOP_KEY} = {best:.3f}")
                break
    print("✔ Best model & threshold saved")

if __name__ == "__main__":
    main()
