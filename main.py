
import os, torch.multiprocessing as mp
import re, json, numpy as np, pandas as pd, torch
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from vllm import LLM, SamplingParams
from config.config import HIDDEN, DATASET, MLP_HIDDEN, TASK, FT_MODEL_DIR
from model.ehrPredictModel import EHRPredictor
from ehr_tas_trainer import PatientDS, coll
from prompt.reasoning_prompt import TASK_DESCRIPTIONS
from transformers import AutoTokenizer

mp.set_start_method("spawn", force=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
GPU_MEM_FRACTION = 0.9
MAX_GEN_TOKENS   = 8192
MAX_MODEL_LEN = 32768
SAFETY_MARGIN = 64
MAX_PROMPT_TOKENS = MAX_MODEL_LEN - MAX_GEN_TOKENS - SAFETY_MARGIN

_RE_PRED_BLOCK = re.compile(r"(?i)#\s*Prediction\s+Result\s*#\s*[\r\n]+\s*([01])", re.S)

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

def _load_labels_for_split(SPLIT_DIR: Path, split: str) -> pd.DataFrame:
    df_all = pd.read_csv(SPLIT_DIR / f"{split}_labels.csv")
    task = TASK.lower()
    label_col = "Outcome" if task == "outcome" else "Readmission"
    if task == "outcome":
        sort_cols = ["PatientID", "VisitID"]
        if "RecordTime" in df_all.columns:
            sort_cols.append("RecordTime")
        df = (
            df_all.sort_values(sort_cols)
            .groupby("PatientID", as_index=False)
            .head(1)
            .sort_values("PatientID")
        )
    else:
        def pick_last_visit_first_row(g: pd.DataFrame) -> pd.DataFrame:
            last_vid = g["VisitID"].max()
            return g[g["VisitID"] == last_vid].head(1)
        df = (
            df_all.groupby("PatientID", group_keys=False)
            .apply(pick_last_visit_first_row)
            .sort_values("PatientID")
        )
    return df[["PatientID", label_col]].rename(columns={label_col: "Label"})

def _ensure_llm_reasoning_for_split(
    split: str,
    SPLIT_DIR: Path,
    CACHE_DIR: Path,
    MODEL_DIR: Path,
    tok: AutoTokenizer,
) -> dict:
    save_path = CACHE_DIR / f"llm_{split}_{TASK.lower()}_reasoning.jsonl"
    existing_reasoning = {}
    # if save_path.exists():
    #     with save_path.open() as fr:
    #         for line in fr:
    #             rec = json.loads(line)
    #             existing_reasoning[int(rec["PatientID"])] = rec["Reasoning"]
    #     print(f"✓ [{split}] Loaded existing reasoning: {len(existing_reasoning)}")

    if save_path.exists():
        valid_lines = []
        with save_path.open() as fr:
            for i, line in enumerate(fr, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    existing_reasoning[int(rec["PatientID"])] = rec["Reasoning"]
                    valid_lines.append(line)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON at line {i}: {e}")
                    continue

        if len(valid_lines) < sum(1 for line in open(save_path) if line.strip()):
            print(f"Rewriting {save_path} to remove invalid lines")
            with save_path.open("w") as fw:
                for line in valid_lines:
                    fw.write(line + "\n")

        print(f"✓ [{split}] Loaded existing reasoning: {len(existing_reasoning)}")

    cmps = pd.read_csv(SPLIT_DIR / f"{split}_cmps.csv").sort_values(["PatientID", "VisitID"])
    ctx_map = {pid: _build_ctx(g) for pid, g in cmps.groupby("PatientID")}

    labels_df = _load_labels_for_split(SPLIT_DIR, split)
    patient_ids = labels_df["PatientID"].to_numpy()

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

    def safe_prompt(text: str) -> str:
        ids = tok.encode(text)
        if len(ids) > MAX_PROMPT_TOKENS:
            print(f"⚠️ [{split}] Prompt too long: {len(ids)} → Truncated to {MAX_PROMPT_TOKENS}")
            ids = ids[-MAX_PROMPT_TOKENS:]
        return tok.decode(ids, skip_special_tokens=True)

    todo_pids = [pid for pid in patient_ids if pid not in existing_reasoning]
    print(f"[{split}] Patients requiring new inference: {len(todo_pids)}")

    if todo_pids:
        prompts = [
            safe_prompt(
                f"[SYSTEM]\n{sys_prompt}\n\n[USER]\n{preamble}{task_desc}\n\n# Patient EHR Context\n"
                f"{ctx_map[pid]}\n\n{guideline}\n"
            )
            for pid in todo_pids
        ]

        llm = LLM(model=str(MODEL_DIR),
                  tensor_parallel_size=1,
                  gpu_memory_utilization=GPU_MEM_FRACTION)

        sampling = SamplingParams(
            temperature=0.1,
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
            for i in tqdm(range(0, len(prompts), bs), desc=f"LLM inference ({split})"):
                outs = llm.generate(prompts[i:i+bs], sampling)
                for pid, out in zip(todo_pids[i:i+bs], outs):
                    raw_txt       = out.outputs[0].text
                    reasoning_txt = truncate_reasoning(raw_txt)
                    existing_reasoning[pid] = reasoning_txt
                    json.dump({"PatientID": int(pid), "Reasoning": reasoning_txt}, fw)
                    fw.write("\n")
                fw.flush()
        print(f"✓ [{split}] New reasoning appended → {save_path}")

    return existing_reasoning  # {pid: reasoning_str}

def _ehr_probs_for_split(CACHE_DIR: Path, split: str, device: str, ehr_model: EHRPredictor) -> np.ndarray:
    ds = PatientDS(CACHE_DIR / f"{split}.npz")
    dl = torch.utils.data.DataLoader(ds, 32, False, collate_fn=coll)
    ehr_model.eval()
    probs = []
    with torch.no_grad():
        for x, m, _ in dl:
            probs.append(torch.sigmoid(ehr_model(x.to(device), m.to(device))).cpu().numpy())
    return np.concatenate(probs).squeeze()

def _labels_and_ids_for_split(SPLIT_DIR: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    labels_df = _load_labels_for_split(SPLIT_DIR, split)
    pids = labels_df["PatientID"].to_numpy()
    y_true = labels_df["Label"].to_numpy().astype(int)
    return pids, y_true

def _y2_probs_from_reasoning(patient_ids: np.ndarray, reasoning_map: dict) -> np.ndarray:
    y2_pred = np.array([_extract_pred(reasoning_map[pid]) for pid in patient_ids])
    return y2_pred.astype(float)

def search_best_alpha_threshold(y_true, y1_prob, y2_prob):
    best_auroc, best_auprc, best_minps = -1.0, -1.0, -1.0
    best_a = best_t = None
    for a in np.linspace(0, 1, 101):
        comb_prob = a * y1_prob + (1 - a) * y2_prob
        auroc = roc_auc_score(y_true, comb_prob)
        auprc = average_precision_score(y_true, comb_prob)
        for t in np.linspace(0.01, 0.99, 181):
            pred = (comb_prob >= t).astype(int)
            tp = ((pred == 1) & (y_true == 1)).sum()
            fp = ((pred == 1) & (y_true == 0)).sum()
            fn = ((pred == 0) & (y_true == 1)).sum()
            precision = tp / (tp + fp + 1e-12)
            recall    = tp / (tp + fn + 1e-12)
            minps = min(precision, recall)
            better = (
                (auroc > best_auroc) or
                (np.isclose(auroc, best_auroc) and auprc > best_auprc) or
                (np.isclose(auroc, best_auroc) and np.isclose(auprc, best_auprc) and minps > best_minps)
            )
            if better:
                best_auroc, best_auprc, best_minps = auroc, auprc, minps
                best_a, best_t = a, t
    return best_a, best_t

def evaluate_fixed(y_true, y1_prob, y2_prob, a, t):
    comb_prob = a * y1_prob + (1 - a) * y2_prob
    auroc = roc_auc_score(y_true, comb_prob)
    auprc = average_precision_score(y_true, comb_prob)
    pred = (comb_prob >= t).astype(int)
    tp = ((pred == 1) & (y_true == 1)).sum()
    fp = ((pred == 1) & (y_true == 0)).sum()
    fn = ((pred == 0) & (y_true == 1)).sum()
    precision = tp / (tp + fp + 1e-12)
    recall    = tp / (tp + fn + 1e-12)
    minps = min(precision, recall)
    return auroc, auprc, minps

def main() -> None:
    DATA_ROOT = Path("data") / DATASET / "preprocessed"
    SPLIT_DIR, CACHE_DIR = DATA_ROOT / "splits", DATA_ROOT / "cache"
    MODEL_DIR = Path(FT_MODEL_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)

    ds_tmp = PatientDS(CACHE_DIR / "val.npz")
    ehr = EHRPredictor(ds_tmp.x[0].shape[1], HIDDEN, mlp_hidden=MLP_HIDDEN, out_dim=1).to(device)
    ehr.load_state_dict(torch.load(CACHE_DIR / f"best_model_{DATASET}_{TASK.lower()}.pt",
                                   map_location=device))

    print("\n=== Searching best α, threshold on VALIDATION set ===")
    val_reasoning = _ensure_llm_reasoning_for_split("val", SPLIT_DIR, CACHE_DIR, MODEL_DIR, tok)
    val_pids, y_true_val = _labels_and_ids_for_split(SPLIT_DIR, "val")
    y1_prob_val = _ehr_probs_for_split(CACHE_DIR, "val", device, ehr)
    y2_prob_val = _y2_probs_from_reasoning(val_pids, val_reasoning)

    ehr_thr = 0.5
    thr_file = CACHE_DIR / "best_threshold.txt"
    if thr_file.exists():
        try:
            ehr_thr = float(thr_file.read_text().strip())
            print(f"✓ Loaded EHR best threshold (for log only): {ehr_thr:.4f}")
        except ValueError:
            print("⚠️ Invalid best_threshold.txt, defaulting to 0.5")

    best_a, best_t = search_best_alpha_threshold(y_true_val, y1_prob_val, y2_prob_val)
    print(f"Validation best α={best_a:.2f}, threshold={best_t:.2f}")

    print("\n=== Final evaluation on TEST set with fixed α, threshold ===")
    test_reasoning = _ensure_llm_reasoning_for_split("test", SPLIT_DIR, CACHE_DIR, MODEL_DIR, tok)
    test_pids, y_true_test = _labels_and_ids_for_split(SPLIT_DIR, "test")
    y1_prob_test = _ehr_probs_for_split(CACHE_DIR, "test", device, ehr)
    y2_prob_test = _y2_probs_from_reasoning(test_pids, test_reasoning)

    auroc, auprc, minps = evaluate_fixed(y_true_test, y1_prob_test, y2_prob_test, best_a, best_t)

    print("\n=======  Final Test Result (α,t from validation)  =======")
    print(f"α             : {best_a:.2f}")
    print(f"β             : {best_t:.2f}")
    print(f"AUROC         : {auroc:.4f}")
    print(f"AUPRC         : {auprc:.4f}")
    print(f"min(+P, Se)   : {minps:.4f}")
    print("=========================================================\n")

if __name__ == "__main__":
    main()
