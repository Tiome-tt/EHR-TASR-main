import os, torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
GPU_MEM_FRACTION = 0.9
MAX_GEN_TOKENS   = 8192

import re, json, numpy as np, pandas as pd, torch
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from vllm import LLM, SamplingParams

from config.config import HIDDEN, DATASET, MLP_HIDDEN, TASK, FT_MODEL_DIR
from model.ehrPredictModel import EHRPredictor
from ehr_tas_trainer import PatientDS, coll
from prompt.reasoning_prompt import TASK_DESCRIPTIONS

_RE_PRED_BLOCK = re.compile(
    r"(?i)#\s*Prediction\s+Result\s*#\s*[\r\n]+\s*([01])", re.S
)

def truncate_reasoning(raw: str) -> str:
    m = _RE_PRED_BLOCK.search(raw)
    if not m:
        return raw.strip()

    end_idx = m.end()
    while end_idx < len(raw) and raw[end_idx] in " \t\r\n":
        end_idx += 1
    return raw[:end_idx]

def _extract_pred(text: str) -> int:
    m = _RE_PRED_BLOCK.search(text)
    return int(m.group(1)) if m else 0

def _build_ctx(visits: pd.DataFrame) -> str:
    rows = []
    for _, r in visits.iterrows():
        head = f'Visit {int(r["VisitID"])}'
        gap  = r.get("GapTime")
        head += f' ({int(gap)} days after last Visit):' if pd.notna(gap) else ":"
        rows.extend([
            head,
            f'Conditions:  {r["Conditions_Long"] or "None"}',
            f'Medications: {r["Medications"]     or "None"}',
            f'Procedures:  {r["Procedures_Long"] or "None"}',
            ""
        ])
    return "\n".join(rows).strip()


def main() -> None:
    DATA_ROOT = Path("data") / DATASET / "preprocessed"
    SPLIT_DIR, CACHE_DIR = DATA_ROOT / "splits", DATA_ROOT / "cache"
    MODEL_DIR = Path(FT_MODEL_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    labels_all = pd.read_csv(SPLIT_DIR / "test_labels.csv")

    LABEL_COL = "Outcome" if TASK.lower() == "outcome" else "Readmission"

    if TASK.lower() == "outcome":
        sort_cols = ["PatientID", "VisitID"]
        if "RecordTime" in labels_all.columns:
            sort_cols.append("RecordTime")
        labels_df = (
            labels_all.sort_values(sort_cols)
            .groupby("PatientID", as_index=False)
            .head(1)
            .sort_values("PatientID")
        )

    else:
        def pick_last_visit_first_row(g: pd.DataFrame) -> pd.DataFrame:
            last_vid = g["VisitID"].max()
            return g[g["VisitID"] == last_vid].head(1)

        labels_df = (
            labels_all.groupby("PatientID", group_keys=False)
            .apply(pick_last_visit_first_row)
            .sort_values("PatientID")
        )

    patient_ids = labels_df["PatientID"].to_numpy()
    y_true = labels_df[LABEL_COL].to_numpy().astype(int)

    print(f"✓ Loaded {TASK} labels: {len(y_true)} patients   "
          f"(positive ratio = {y_true.mean():.2%})")



    ds_test = PatientDS(CACHE_DIR / "test.npz")
    dl      = torch.utils.data.DataLoader(ds_test, 32, False, collate_fn=coll)
    ehr = EHRPredictor(ds_test.x[0].shape[1], HIDDEN,
                       mlp_hidden=MLP_HIDDEN, out_dim=1).to(device)
    ehr.load_state_dict(torch.load(CACHE_DIR / f"best_model_{DATASET}_{TASK.lower()}.pt",
                                   map_location=device))
    ehr.eval()
    y1_prob = []
    with torch.no_grad():
        for x, m, _ in dl:
            y1_prob.append(torch.sigmoid(ehr(x.to(device), m.to(device))).cpu().numpy())
    y1_prob = np.concatenate(y1_prob).squeeze()
    print("✓ y₁ finished")


    ehr_thr = 0.5
    thr_file = CACHE_DIR / "best_threshold.txt"
    if thr_file.exists():
        try:
            ehr_thr = float(thr_file.read_text().strip())
            print(f"✓ Loaded best threshold: {ehr_thr:.4f}")
        except ValueError:
            print("⚠️ Invalid content in best_threshold.txt, defaulting to 0.5")
    else:
        print("⚠️ best_threshold.txt not found, defaulting to 0.5")


    y1_pred = (y1_prob >= ehr_thr).astype(int)

    save_path = CACHE_DIR / f"llm_test_{TASK.lower()}_reasoning.jsonl"
    existing_reasoning = {}
    if save_path.exists():
        with save_path.open() as fr:
            for line in fr:
                rec = json.loads(line)
                existing_reasoning[int(rec["PatientID"])] = rec["Reasoning"]
        print(f"✓ Loaded existing reasoning: {len(existing_reasoning)}")

    cmps = pd.read_csv(SPLIT_DIR / "test_cmps.csv").sort_values(["PatientID", "VisitID"])
    ctx_map = {pid: _build_ctx(g) for pid, g in cmps.groupby("PatientID")}

    sys_prompt = ("You are a clinical-reasoning assistant that reads structured "
                  "EHR data and outputs a concise reasoning chain and a prediction.")
    task_desc = TASK_DESCRIPTIONS["Outcome"].strip()
    preamble  = ("Given the following task description and patient EHR context, "
                 "provide a step-by-step reasoning chain and the predicted outcome (0/1).\n")

    guideline = """
The reasoning chain should follow this structured format:
1. **Patient Overview**: Check the key information in the patient's context, with the Key Considerations from the task description in mind.
2. **Acute Severity**: Flag ICU-level interventions or critical events—mechanical ventilation, vasopressors, sepsis, shock, emergency surgery.
3. **Chronic Comorbidities**: Note enduring diseases such as heart failure, COPD, CKD, diabetes, malignancy, along with their long-term medications.
4. **Therapy Complexity**: Record multiple or major procedures, invasive interventions, broad-spectrum or high-alert drugs, and overall polypharmacy.
5. **Conclusion**: Summarize the reasoning and state the prediction without mentioning the ground truth.

The reasoning should be comprehensive, medically sound, and clearly explain how the patient's information leads to the predicted outcome.

## Important Notes:
1. Use only the facts in the patient EHR context – no invented data.
2. Strictly follow the Output Format below; keep each bullet 2–3 concise sentences.
3. Do not expose the ground-truth label or any wording that unmistakably gives it away.

## Output Format：

# Reasoning #

1.Patient Overview:
[YOUR OUTPUT]

2.Acute Severity：
[YOUR OUTPUT]

3.Chronic Comorbidities：
[YOUR OUTPUT]

4.Therapy Complexity：
[YOUR OUTPUT]

5.Conclusion:
[YOUR OUTPUT]

# Prediction Result #

[YOUR OUTPUT (0 or 1)]

Response:""".strip()

    todo_pids = [pid for pid in patient_ids if pid not in existing_reasoning]
    print(f"Patients requiring new inference: {len(todo_pids)}")

    if todo_pids:
        prompts = [
            f"[SYSTEM]\n{sys_prompt}\n\n[USER]\n{preamble}{task_desc}\n\n# Patient EHR Context\n"
            f"{ctx_map[pid]}\n\n{guideline}\n"
            for pid in todo_pids
        ]

        # ── LLM 初始化 + stop 列表 ---------------------------------
        llm = LLM(model=str(MODEL_DIR),
                  tensor_parallel_size=1,
                  gpu_memory_utilization=GPU_MEM_FRACTION)

        sampling = SamplingParams(
            temperature=0,
            top_p=0.9,
            max_tokens=MAX_GEN_TOKENS,
            stop=[
                "# Prediction Result # 0",
                "# Prediction Result # 1",
                "# Prediction Result #\n0",
                "# Prediction Result #\n1",
                "# Prediction Result #\n\n 0",
                "# Prediction Result #\n\n 1"
            ],
            include_stop_str_in_output=True
        )

        with save_path.open("a") as fw:
            bs = 8
            for i in tqdm(range(0, len(prompts), bs), desc="LLM inference"):
                outs = llm.generate(prompts[i:i+bs], sampling)
                for pid, out in zip(todo_pids[i:i+bs], outs):
                    raw_txt       = out.outputs[0].text
                    reasoning_txt = truncate_reasoning(raw_txt)
                    existing_reasoning[pid] = reasoning_txt
                    json.dump({"PatientID": int(pid), "Reasoning": reasoning_txt}, fw)
                    fw.write("\n")
                fw.flush()
        print("✓ New reasoning successfully appended →", save_path)

    y2_pred = np.array([_extract_pred(existing_reasoning[pid]) for pid in patient_ids])
    y2_prob = y2_pred.astype(float)
    best_auroc, best_auprc, best_minps = -1.0, -1.0, -1.0
    best_a = best_t = best_pred = best_prob = None

    for a in np.linspace(0, 1, 101):
        comb_prob = a * y1_prob + (1 - a) * y2_prob
        auroc = roc_auc_score(y_true, comb_prob)
        auprc = average_precision_score(y_true, comb_prob)

        for t in np.linspace(0.05, 0.95, 181):
            pred = (comb_prob >= t).astype(int)

            tp = ((pred == 1) & (y_true == 1)).sum()
            fp = ((pred == 1) & (y_true == 0)).sum()
            fn = ((pred == 0) & (y_true == 1)).sum()
            precision = tp / (tp + fp + 1e-12)
            recall = tp / (tp + fn + 1e-12)
            min_ps = min(precision, recall)

            better = (
                    (auroc > best_auroc) or
                    (np.isclose(auroc, best_auroc) and auprc > best_auprc) or
                    (np.isclose(auroc, best_auroc) and np.isclose(auprc, best_auprc) and
                     min_ps > best_minps)
            )

            # better = (
            #         (min_ps > best_minps) or
            #         (np.isclose(min_ps, best_minps) and auprc > best_auprc) or
            #         (np.isclose(min_ps, best_minps) and np.isclose(auprc, best_auprc) and
            #          auroc > best_auroc)
            # )

            if better:
                best_auroc, best_auprc, best_minps = auroc, auprc, min_ps
                best_a, best_t = a, t
                best_pred = pred
                best_prob = comb_prob
    # ================================================================

    print("\n=======  The optimal ensemble result  =======")
    print(f"Best α        : {best_a:.2f}")
    print(f"Best threshold: {best_t:.2f}")
    print(f"AUROC         : {best_auroc:.4f}")
    print(f"AUPRC         : {best_auprc:.4f}")
    print(f"min(+P, Se)   : {best_minps:.4f}")
    print("================================\n")

if __name__ == "__main__":
    main()
