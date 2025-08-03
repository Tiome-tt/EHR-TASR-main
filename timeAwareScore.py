import os
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Tuple
from config.config import DATASET, ROOT, BLOCK_LEN, POOL_DIM, SEED

SRC_CSV   = f"{ROOT}/data/{DATASET}/preprocessed/timeseries_{DATASET}.csv"
OUT_DIR   = f"{ROOT}/data/{DATASET}/preprocessed/splits"
RATIO     = (0.7, 0.1, 0.2) # split data ratio
CLIP_LOW, CLIP_HIGH = -250, 250 # isolate dirty data

COLUMN_MAP = {
    "patient_id":       "PatientID",
    "visit_id":         "VisitID",
    "time":             "RecordTime",
    "age":              "Age",
    "dbp":              "Diastolic blood pressure",
    "sbp":              "Systolic blood pressure",
    "mbp":              "Mean blood pressure",
    "hr":               "Heart Rate",
    "rr":               "Respiratory rate",
    "temp":             "Temperature",
    "spo2":             "Oxygen saturation",
    "fio2":             "Fraction inspired oxygen",
    "glucose":          "Glucose",
    "ph":               "PH",
    "cr_less3":         "Capillary refill rate-> less than 3s",
    "cr_gt3":           "Capillary refill rate-> greater than 3s",
}

# Standard for scoring the degree of abnormality
RANGES: Dict[str, List[Tuple[float, float, int]]] = {
    "age":     [(np.nan, 64, 0), (65, 74, 1), (75, 84, 2), (85, np.nan, 3)],
    "dbp":     [(60, 80, 0), (50, 59, 1), (81, 90, 1),
                (40, 49, 2), (91, 100, 2), (np.nan, 39, 3), (101, 150, 3)],
    "sbp":     [(100, 120, 0), (90, 99, 1), (121, 140, 1),
                (80, 89, 2), (141, 160, 2), (np.nan, 79, 3), (161, 240, 3)],
    "mbp":     [(70, 100, 0), (60, 69, 1), (101, 110, 1),
                (50, 59, 2), (111, 130, 2), (np.nan, 49, 3), (131, 180, 3)],
    "hr":      [(60, 100, 0), (50, 59, 1), (101,110,1),
                (40, 49, 2), (111,130,2), (np.nan,39,3), (131,170,3)],
    "rr":      [(15, 18, 0), (11, 14, 1), (19, 22, 1),
                (8, 10, 2), (23, 25, 2), (np.nan,7,3), (26,34,3)],
    "temp":    [(36.1, 37.2, 0), (35.5, 36.0, 1), (37.3, 38.0, 1),
                (34.0, 35.4, 2), (38.1, 39.5, 2), (30.0, 33.9, 3), (39.6, 45, 3)],
    "spo2":    [(95, 100, 0), (92, 94, 1), (90, 91, 2), (np.nan, 89, 3)],
    "fio2":    [(21, 100, 0), (18, 20, 1), (15, 17, 2), (np.nan, 14, 3)],
    "glucose": [(70, 100, 0), (60, 69, 1), (101, 140, 1),
                (50, 59, 2), (141, 180, 2), (np.nan,49,3), (181,220,3)],
    "ph":      [(7.35, 7.45, 0), (7.30, 7.34, 1), (7.46, 7.5, 1),
                (7.20, 7.29, 2), (7.51, 7.6, 2), (7.00, 7.19, 3), (7.61, 7.8, 3)],
}

GCS_PREFIX = "Glascow coma scale total->"

def _score_continuous(val: float, feat: str) -> int:
    if pd.isna(val):
        return np.nan
    for lo, hi, s in RANGES[feat]:
        if (np.isnan(lo) or val >= lo) and (np.isnan(hi) or val <= hi):
            return s
    return np.nan

def _score_cap_refill(row) -> int:
    if row[COLUMN_MAP["cr_less3"]] == 1: return 0
    if row[COLUMN_MAP["cr_gt3"]] == 1:   return 2
    return np.nan

def _score_gcs(row) -> float:
    cols = [c for c in row.index if c.startswith(GCS_PREFIX) and row[c] == 1]
    if not cols: return np.nan
    return np.mean([int(c.split("->")[-1]) for c in cols])

CONT_FEATS = ["age", "dbp", "sbp", "mbp", "hr", "rr", "temp",
              "spo2", "fio2", "glucose", "ph"]

def time_aware_score(csv_path: str, block_lens=BLOCK_LEN) -> Dict[Tuple[str, str], List[float]]:
    df = pd.read_csv(csv_path)
    df[COLUMN_MAP["time"]] = pd.to_datetime(df[COLUMN_MAP["time"]])
    df = df.sort_values([COLUMN_MAP["patient_id"], COLUMN_MAP["visit_id"], COLUMN_MAP["time"]])

    for feat in CONT_FEATS:
        df[f"score_{feat}"] = df[COLUMN_MAP[feat]].apply(_score_continuous, feat=feat)
    df["score_cap_refill"] = df.apply(_score_cap_refill, axis=1)
    df["score_gcs"] = df.apply(_score_gcs, axis=1)
    fea_cols = [c for c in df.columns if c.startswith("score_") and c != "score_gcs"]

    tas = {}
    for (pid, vid), g in tqdm(df.groupby([COLUMN_MAP["patient_id"], COLUMN_MAP["visit_id"]]), desc="TAS calc", unit="visit"):
        g = g.reset_index(drop=True)
        L = len(g)

        all_scores = []

        for blk_len in block_lens:
            if blk_len == "full":
                blk_len = L
            if L < blk_len:
                continue
            for st in range(L - blk_len + 1):
                blk = g.iloc[st:st + blk_len]
                gcs = blk["score_gcs"].mean(skipna=True)
                fea = blk[fea_cols].mean(skipna=True).mean(skipna=True)
                gcs = 0 if pd.isna(gcs) else gcs
                fea = 0 if pd.isna(fea) else fea
                score = float((gcs * 4 / 15 + fea) / 8)
                all_scores.append(score)

        tas[(pid, vid)] = all_scores
    return tas

def pool_sequence(seq:List[float], dim=24)->np.ndarray:
    if not seq: return np.zeros(dim,np.float32)
    return np.interp(np.linspace(0,1,dim,dtype=np.float32),
                     np.linspace(0,1,len(seq),dtype=np.float32),
                     np.asarray(seq,np.float32)).astype(np.float32)

def build_visit_df(tas_dict, pool_dim=24)->pd.DataFrame:
    rows=[[pid,vid,*pool_sequence(seq,pool_dim)]
          for (pid,vid),seq in tqdm(tas_dict.items(),desc="Pooling",unit="visit")]
    return pd.DataFrame(rows,columns=["PatientID","VisitID"]+[f"D{i}"for i in range(pool_dim)])

def split_by_patient(df, ratio, seed=42):
    pids=df["PatientID"].unique().tolist()
    random.Random(seed).shuffle(pids)
    n=len(pids); n_tr=int(n*ratio[0]); n_val=int(n*ratio[1])
    set_tr=set(pids[:n_tr]); set_val=set(pids[n_tr:n_tr+n_val])
    mask_tr=df["PatientID"].isin(set_tr)
    mask_val=df["PatientID"].isin(set_val)
    mask_te=~(mask_tr|mask_val)
    return df[mask_tr], df[mask_val], df[mask_te]

def robust_minmax(col, mn, mx, lo, hi):
    bad=(col<lo)|(col>hi)
    out=np.ones(len(col),dtype=np.float32)
    denom=mx-mn
    good=~bad
    if denom>0:
        out[good]=(col[good]-mn)/denom
    return pd.Series(out,index=col.index)

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"----------Read {DATASET} data----------")
    df_ts = pd.read_csv(SRC_CSV)

    print("----------Calculate TAS----------")
    # 多尺度滑窗
    tas = time_aware_score(SRC_CSV, block_lens=BLOCK_LEN)

    print(f"----------Pooling to {POOL_DIM} dimentions----------")
    df_visit = build_visit_df(tas, POOL_DIM)
    df_all = df_ts.merge(df_visit, on=["PatientID", "VisitID"], how="left") \
        .drop(columns=[c for c in ["RecordTimestamp", "AdmissionTime", "DischargeTime"] if c in df_ts])

    outcome_max = (df_all.groupby("PatientID")["Outcome"]
                   .max())
    df_all["Outcome"] = df_all["PatientID"].map(outcome_max)

    print("----------Positional encoding----------")
    visit_unique = (df_all[["PatientID", "VisitID"]]
                    .drop_duplicates()
                    .sort_values(["PatientID", "VisitID"]))
    visit_unique["rank_visit"] = visit_unique.groupby("PatientID").cumcount() + 1
    visit_unique["n_visit"] = visit_unique.groupby("PatientID")["VisitID"].transform("size")
    visit_unique["VisitPos"] = np.where(visit_unique["n_visit"] == 1, 1.0,
                                        (visit_unique["rank_visit"] - 1) / (visit_unique["n_visit"] - 1))
    df_all = df_all.merge(visit_unique[["PatientID", "VisitID", "VisitPos"]],
                          on=["PatientID", "VisitID"], how="left")
    df_all["VisitID"] = df_all["VisitPos"]
    df_all = df_all.drop(columns="VisitPos")

    df_all["rank_rec"] = df_all.groupby(["PatientID", "VisitID"]).cumcount() + 1
    df_all["n_rec"] = df_all.groupby(["PatientID", "VisitID"])["RecordTime"].transform("size")
    df_all["RecordTime"] = np.where(df_all["n_rec"] == 1, 1.0,
                                    (df_all["rank_rec"] - 1) / (df_all["n_rec"] - 1))
    df_all = df_all.drop(columns=["rank_rec", "n_rec"])

    # Splitting data
    df_tr, df_val, df_te = split_by_patient(df_all, RATIO, SEED)

    # Normalizing
    phys_cols = ["Age", "Weight", "Height", "Diastolic blood pressure", "Systolic blood pressure",
                 "Mean blood pressure", "Heart Rate", "Respiratory rate", "Temperature",
                 "Oxygen saturation", "Fraction inspired oxygen", "Glucose", "PH"]
    d_cols = [f"D{i}" for i in range(POOL_DIM)]
    cont_cols = ["VisitID", "RecordTime"] + phys_cols + d_cols

    stats = {}
    for c in cont_cols:
        tmp = df_tr[c].clip(CLIP_LOW, CLIP_HIGH)
        stats[c] = (tmp.min(), tmp.max())

    # for df_ in (df_tr, df_val, df_te):
    #     for c, (mn, mx) in stats.items():
    #         result = robust_minmax(df_[c], mn, mx, CLIP_LOW, CLIP_HIGH)
    #         result = np.nan_to_num(result, nan=0.0).astype(np.float32)
    #         if df_[c].dtype != np.float32:
    #             df_[c] = df_[c].astype(np.float32)
    #         df_.loc[:, c] = result

    for df_ in (df_tr, df_val, df_te):
        for c, (mn, mx) in stats.items():
            result = robust_minmax(df_[c], mn, mx, CLIP_LOW, CLIP_HIGH)
            df_.loc[:, c] = np.nan_to_num(result, nan=0.0).astype(np.float32)


    label_cols = ["Outcome", "LOS", "Readmission"]
    feat_cols = [c for c in df_all.columns if c not in label_cols]
    for tag, d in zip(["train", "val", "test"], [df_tr, df_val, df_te]):
        d[feat_cols].to_csv(f"{OUT_DIR}/{tag}_features.csv", float_format="%.6f", index=False)
        d[["PatientID", "VisitID"] + label_cols].to_csv(
            f"{OUT_DIR}/{tag}_labels.csv", float_format="%.6f", index=False)

    print("----------Finished----------")
    for tag, d in zip(["train", "val", "test"], [df_tr, df_val, df_te]):
        print(f"{tag:5}", d.shape)

# ----------Split CMPs file & compute GapTime----------
print("----------Split CMPs file----------")

CMP_CSV  = f"{ROOT}/data/{DATASET}/preprocessed/CMPs_{DATASET}.csv"

CMP_COLS = ["PatientID", "VisitID",
            "AdmissionTime", "DischargeTime",
            "Conditions_Long", "Medications", "Procedures_Long"]

df_cmp = pd.read_csv(CMP_CSV, usecols=CMP_COLS)

# compute GapTime (day)
df_cmp["AdmissionTime"]  = pd.to_datetime(df_cmp["AdmissionTime"])
df_cmp["DischargeTime"]  = pd.to_datetime(df_cmp["DischargeTime"])

df_cmp = df_cmp.sort_values(["PatientID", "AdmissionTime"])

prev_discharge = (df_cmp
                  .groupby("PatientID")["DischargeTime"]
                  .shift())

df_cmp["GapTime"] = (df_cmp["AdmissionTime"] - prev_discharge).dt.days.astype("float32")

df_cmp = df_cmp.drop(columns=["AdmissionTime", "DischargeTime"])

# unique set
pids_tr  = set(df_tr["PatientID"].unique())
pids_val = set(df_val["PatientID"].unique())
pids_te  = set(df_te["PatientID"].unique())

df_cmp_tr  = df_cmp[df_cmp["PatientID"].isin(pids_tr)]
df_cmp_val = df_cmp[df_cmp["PatientID"].isin(pids_val)]
df_cmp_te  = df_cmp[df_cmp["PatientID"].isin(pids_te)]

# save to csv
keep_cols = ["PatientID", "VisitID", "GapTime",
             "Conditions_Long", "Medications", "Procedures_Long"]

df_cmp_tr [keep_cols].to_csv(f"{OUT_DIR}/train_cmps.csv", index=False)
df_cmp_val[keep_cols].to_csv(f"{OUT_DIR}/val_cmps.csv",   index=False)
df_cmp_te [keep_cols].to_csv(f"{OUT_DIR}/test_cmps.csv",  index=False)

print("CMPs split saved:",
      {k: v.shape for k, v in
       zip(["train", "val", "test"],
           [df_cmp_tr, df_cmp_val, df_cmp_te])})

def task_stats(tag: str, df: pd.DataFrame, tasks: List[str]) -> None:
    print(f"\n===== {tag.upper()} SET =====")
    print(f"Total rows      : {len(df):,}")
    print(f"Unique patients : {df['PatientID'].nunique():,}")

    visit_grp   = df.groupby(['PatientID', 'VisitID'])
    patient_grp = df.groupby('PatientID')

    avg_visits_per_patient = visit_grp.ngroups / patient_grp.ngroups
    print(f"Avg visits/patient: {avg_visits_per_patient:.2f}")

    for t in tasks:
        visit_lbl = visit_grp[t].max()
        pos_v = (visit_lbl > 0).sum()
        neg_v = (visit_lbl == 0).sum()

        patient_lbl = patient_grp[t].max()
        pos_p = (patient_lbl > 0).sum()
        neg_p = (patient_lbl == 0).sum()

        print(f"{t:<12} | Visit‑level  Pos:{pos_v:6}  Neg:{neg_v:6}  "
              f"Pos‑Ratio:{pos_v/(pos_v+neg_v):.3%}  |  "
              f"Patient‑level  Pos:{pos_p:6}  Neg:{neg_p:6}  "
              f"Pos‑Ratio:{pos_p/(pos_p+neg_p):.3%}")

label_cols = ["Outcome", "Readmission"]
for tag, d in zip(["train", "val", "test"], [df_tr, df_val, df_te]):
    task_stats(tag, d, label_cols)
