import os, json
from pathlib import Path
from itertools import islice
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from config.config import DATASET, ROOT, SPLIT, MODEL_DIR
from prompt.reasoning_prompt import SYSTEM_PROMPT, USER_PROMPT, TASK

# environment
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["VLLM_USE_COMPILE"] = "0"
ROOT        = Path(ROOT)
MODEL_DIR   = Path(MODEL_DIR)
BATCH_SIZE  = 10
MAX_NEW     = 1048
KV_MAX_LEN  = 20000
GT_COL      = {"Outcome": "Outcome",
               "Readmission": "Readmission",
               "LOS": "LOS"}[TASK]

def build_ctx(v: pd.DataFrame) -> str:
    rows = []
    for _, r in v.iterrows():
        head = f'Visit {int(r["VisitID"])}'
        gap  = r["GapTime"]
        head += f' ("{int(gap)}" days after last Visit):' if pd.notna(gap) else ":"
        rows += [
            head,
            f'Conditions:  {r["Conditions_Long"] or "None"}',
            f'Medications: {r["Medications"]      or "None"}',
            f'Procedures:  {r["Procedures_Long"]  or "None"}',
            ""
        ]
    return "\n".join(rows).strip()

def build_gt(v: pd.DataFrame) -> str:
    v = v.drop_duplicates("VisitID", keep="first")
    return "\n".join(
        f'Visit {int(i)}: {"NA" if pd.isna(y) else int(y)}'
        for i, y in v[["VisitID", GT_COL]].values
    )

def batched(it, n):
    it = iter(it)
    while True:
        chunk = list(islice(it, n))
        if not chunk:
            break
        yield chunk

def main() -> None:
    data_dir = ROOT / "data" / DATASET / "preprocessed" / "splits"
    out_dir  = ROOT / "data" / DATASET / "preprocessed" / "reasoning_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    p_reason = out_dir / f"{TASK.lower()}_reasoning_chain.jsonl"

    done_pids = set()
    if p_reason.exists():
        with p_reason.open("r", encoding="utf-8") as fr:
            for line in fr:
                try:
                    done_pids.add(json.loads(line)["PatientID"])
                except Exception:
                    continue
    mode = "a" if done_pids else "w"

    # read data
    cmps   = pd.read_csv(data_dir / f"{SPLIT}_cmps.csv")
    labels = pd.read_csv(data_dir / f"{SPLIT}_labels.csv")
    cmps["PatientID"]   = cmps["PatientID"].astype("int64")
    labels["PatientID"] = labels["PatientID"].astype("int64")

    if TASK in {"Outcome", "Readmission"}:
        labels = (labels.sort_values(["PatientID", "VisitID"])
                        .drop_duplicates(["PatientID", "VisitID"], keep="first"))
    else:
        labels = (labels.groupby(["PatientID", "VisitID"], as_index=False)
                        .agg({GT_COL: "mean"}))

    cmps["VisitRank"]   = cmps.groupby("PatientID").cumcount() + 1
    labels["VisitRank"] = labels.groupby("PatientID").cumcount() + 1

    df = (cmps.merge(labels[["PatientID", "VisitRank", GT_COL]],
                     on=["PatientID", "VisitRank"],
                     how="left",
                     validate="one_to_one")
              .sort_values(["PatientID", "VisitRank"])
              .reset_index(drop=True))

    # Filter out patients who have already been processed
    if done_pids:
        df = df[~df["PatientID"].isin(done_pids)]
        if df.empty:
            print("All patients already processed. Nothing to do.")
            return

    # model
    print("Loading model")
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    llm = LLM(
        model=str(MODEL_DIR),
        quantization="gptq",
        dtype="half",
        trust_remote_code=True,
        max_model_len=KV_MAX_LEN,
        enforce_eager=True,
        tensor_parallel_size=len(os.environ["CUDA_VISIBLE_DEVICES"].split(",")),
        gpu_memory_utilization=0.90,
    )
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        max_tokens=MAX_NEW,
        stop=["<|im_end|>", "</s>"],
    )

    # model inference
    with p_reason.open(mode, encoding="utf-8") as f_r:
        g_iter = df.groupby("PatientID")
        for batch in tqdm(batched(g_iter, BATCH_SIZE),
                          total=(g_iter.ngroups + BATCH_SIZE - 1) // BATCH_SIZE,
                          unit="batch", desc="Generate"):
            pids, prompts = [], []

            for pid, grp in batch:
                user_prompt = USER_PROMPT.replace("{EHR_CONTEXT}", build_ctx(grp)) \
                    .replace("{GROUND_TRUTH}", build_gt(grp))
                prompt = tok.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_PROMPT.strip()},
                     {"role": "user", "content": user_prompt.strip()}],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False).strip() + " "

                # Check if the prompt exceeds the max length
                if len(tok(prompt)["input_ids"]) > KV_MAX_LEN:
                    print(f"Skipping PatientID {pid} due to prompt length exceeding {KV_MAX_LEN}.")
                    continue  # Skip this patient

                pids.append(pid)
                prompts.append(prompt)

            # Generate outputs for remaining patients
            outs = llm.generate(prompts, sampling_params)
            for pid, out in zip(pids, outs):
                json.dump({"PatientID": pid,
                           "Reasoning_chain": out.outputs[0].text.strip(),
                           "Task": TASK},
                          f_r, ensure_ascii=False)
                f_r.write("\n")
            f_r.flush()

    print(f"Finished. Reasonings → {p_reason.resolve()}")

if __name__ == "__main__":
    main()
