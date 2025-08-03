import os, re, json, numpy as np, pandas as pd, torch.multiprocessing as mp
mp.set_start_method("spawn", force=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, \
                            precision_score, recall_score
from vllm import LLM, SamplingParams
from config.config import DATASET, TASK, FT_MODEL_DIR
from ehr_tas_trainer import PatientDS, coll
from prompt.reasoning_prompt import TASK_DESCRIPTIONS

_RE_PRED_BLOCK = re.compile(
    r"(?i)#\s*Prediction\s+Result\s*#\s*(?:[\r\n]+\s*)?([01])",
    re.S
)

def truncate_reasoning(raw: str) -> str:
    m = _RE_PRED_BLOCK.search(raw)
    if not m:
        return raw.strip()
    end_idx = m.end()
    while end_idx < len(raw) and raw[end_idx] in ' \\t\\r\\n':
        end_idx += 1
    return raw[:end_idx]

def _extract_pred(text: str) -> int:

    m = _RE_PRED_BLOCK.search(text)
    return int(m.group(1)) if m else 0

DATA_ROOT  = Path("data")/DATASET/"preprocessed"
SPLIT_DIR  = DATA_ROOT/"splits"
CACHE_DIR  = DATA_ROOT/"cache"
MODEL_DIR  = Path(FT_MODEL_DIR)

LABEL_COL  = "Outcome" if TASK.lower()=="outcome" else "Readmission"
labels_all = pd.read_csv(SPLIT_DIR/"test_labels.csv")

if TASK.lower()=="outcome":
    sort_cols = ["PatientID","VisitID"]
    if "RecordTime" in labels_all.columns:
        sort_cols.append("RecordTime")
    labels_df = (labels_all.sort_values(sort_cols)
                           .groupby("PatientID",as_index=False).head(1)
                           .sort_values("PatientID"))
else:
    def _last_visit_first_row(g):
        return g[g["VisitID"]==g["VisitID"].max()].head(1)
    labels_df = (labels_all.groupby("PatientID",group_keys=False)
                           .apply(_last_visit_first_row)
                           .sort_values("PatientID"))

patient_ids = labels_df["PatientID"].to_numpy()
y_true      = labels_df[LABEL_COL].to_numpy().astype(int)
print(f"✓ Loaded {len(y_true)} labels  (positive={y_true.mean():.2%})")

save_path = CACHE_DIR/f"llm_test_{TASK.lower()}_reasoning.jsonl"
existing   = {}
if save_path.exists():
    with save_path.open() as fr:
        for line in fr:
            rec = json.loads(line)
            existing[int(rec["PatientID"])] = rec["Reasoning"]
    print(f"✓ Found existing reasoning: {len(existing)}")

todo_pids = [pid for pid in patient_ids if pid not in existing]
if todo_pids:
    cmps = pd.read_csv(SPLIT_DIR/"test_cmps.csv").sort_values(["PatientID","VisitID"])
    def _ctx(g):
        rows=[]
        for _,r in g.iterrows():
            head=f'Visit {int(r["VisitID"])}:'
            rows+= [head,
                    f'Conditions:  {r["Conditions_Long"] or "None"}',
                    f'Medications: {r["Medications"]     or "None"}',
                    f'Procedures:  {r["Procedures_Long"] or "None"}',
                    ""]
        return "\\n".join(rows).strip()
    ctx_map = {pid:_ctx(g) for pid,g in cmps.groupby("PatientID")}

    sys_prompt = ("You are a clinical-reasoning assistant that reads structured "
                  "EHR data and outputs a concise reasoning chain and a prediction.")
    task_desc  = TASK_DESCRIPTIONS["Outcome"].strip()
    preamble   = ("Given the following task description and patient EHR context, "
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

Response:"""

    prompts = [
        f"[SYSTEM]\\n{sys_prompt}\\n\\n[USER]\\n{preamble}{task_desc}"
        f"\\n\\n# Patient EHR Context\\n{ctx_map[pid]}\\n\\n{guideline}\\n"
        for pid in todo_pids
    ]

    llm = LLM(model=str(MODEL_DIR),
              tensor_parallel_size=1,
              gpu_memory_utilization=0.9)

    sampling = SamplingParams(
        temperature=0,
        top_p=0.9,
        max_tokens=8192,
        stop=[
            "# Prediction Result # 0", "# Prediction Result # 1",
            "# Prediction Result #\\n0", "# Prediction Result #\\n1",
            "# Prediction Result #\\n\\n 0", "# Prediction Result #\\n\\n 1"
        ],
        include_stop_str_in_output=True
    )

    with save_path.open("a") as fw:
        bs = 8
        for i in tqdm(range(0,len(prompts),bs),desc="LLM inference"):
            outs = llm.generate(prompts[i:i+bs], sampling)
            for pid,out in zip(todo_pids[i:i+bs], outs):
                raw_txt = out.outputs[0].text
                reasoning_txt = truncate_reasoning(raw_txt)
                existing[pid] = reasoning_txt
                json.dump({"PatientID":int(pid),
                           "Reasoning":reasoning_txt}, fw)
                fw.write("\\n")
    print("✓ New reasoning saved →", save_path)

y_pred = np.array([_extract_pred(existing[pid]) for pid in patient_ids])
y_prob = y_pred.astype(float)

auroc = roc_auc_score(y_true, y_prob)
auprc = average_precision_score(y_true, y_prob)
precision = precision_score(y_true, y_pred)
recall    = recall_score(y_true, y_pred)
min_ps    = min(precision, recall)

print("\\n========  LLM‑only Test Metrics  ========")
print(f"AUROC             : {auroc:.4f}")
print(f"AUPRC             : {auprc:.4f}")
print(f"min(Precision,Se) : {min_ps:.4f}   "
      f"(Precision={precision:.4f}, Sensitivity={recall:.4f})")
print("==========================================")
