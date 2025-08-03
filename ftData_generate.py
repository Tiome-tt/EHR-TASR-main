import json, random
from pathlib import Path
from typing import Union
import pandas as pd
from prompt.reasoning_prompt import TASK_DESCRIPTIONS
from config.config import ROOT, SEED, DATASET, TASK


def _build_ctx(visits: pd.DataFrame) -> str:
    rows = []
    for _, r in visits.iterrows():
        head = f'Visit {int(r["VisitID"])}'
        gap  = r["GapTime"]
        head += f' ({int(gap)} days after last Visit):' if pd.notna(gap) else ":"
        rows.extend([
            head,
            f'Conditions:  {r["Conditions_Long"] or "None"}',
            f'Medications: {r["Medications"] or "None"}',
            f'Procedures:  {r["Procedures_Long"] or "None"}',
            ""
        ])
    return "\n".join(rows).strip()


# ----------------------------------------------------------------------
def build_finetune_dataset(
    cmps_csv: Union[str, Path],
    labels_csv: Union[str, Path],
    reasoning_jsonl: Union[str, Path],
    output_jsonl: Union[str, Path] = "outcome_reasoning_chain.jsonl",
    task: str = "Outcome",          # Outcome / Readmission / LOS
    sample_size: int = 10_000,
    random_state: int = SEED,
    system_prompt: str = (
        "You are a clinical-reasoning assistant that reads structured EHR data "
        "and outputs a concise, evidence-based reasoning chain and a prediction result "
        "for a specified prediction task."
    ),
) -> Path:
    cmps_csv, labels_csv, reasoning_jsonl, output_jsonl = map(Path, (
        cmps_csv, labels_csv, reasoning_jsonl, output_jsonl
    ))

    # load datas
    cmps   = pd.read_csv(cmps_csv)
    labels = pd.read_csv(labels_csv)
    if not reasoning_jsonl.exists() or reasoning_jsonl.stat().st_size == 0:
        raise FileNotFoundError(f"{reasoning_jsonl} not found or empty.")
    rc_df  = pd.read_json(str(reasoning_jsonl), lines=True)

    cmps["PatientID"]   = cmps["PatientID"].astype("int64")
    labels["PatientID"] = labels["PatientID"].astype("int64")

    if task in {"Outcome", "Readmission"}:
        labels = (labels.sort_values(["PatientID", "VisitID"])
                        .drop_duplicates(["PatientID", "VisitID"], keep="first"))
    else:
        labels = (labels.groupby(["PatientID", "VisitID"], as_index=False)
                        .agg({task: "mean"}))

    cmps["VisitRank"]   = cmps.groupby("PatientID").cumcount() + 1
    labels["VisitRank"] = labels.groupby("PatientID").cumcount() + 1

    data = (cmps.merge(labels[["PatientID", "VisitRank", task]],
                       on=["PatientID", "VisitRank"],
                       how="left",
                       validate="one_to_one")
                .sort_values(["PatientID", "VisitRank"]))

    # Filter existing reasoning chains
    data = data[data["PatientID"].isin(rc_df["PatientID"])]
    if data.empty:
        raise ValueError("No patients with generated reasoning chains were found.")

    # Random sampling
    rng = random.Random(random_state)
    all_ids = list(data["PatientID"].unique())
    if len(all_ids) < sample_size:
        raise ValueError(f"Only {len(all_ids)} patients available, fewer than requested {sample_size}.")
    chosen = set(rng.sample(all_ids, sample_size))
    data   = data[data["PatientID"].isin(chosen)]
    rc_df  = rc_df[rc_df["PatientID"].isin(chosen)]

    # Merge the reasoning chain and labels
    merged = (
        data.groupby("PatientID")
            .apply(lambda g: {
                "PatientID": int(g.name),
                "Input": _build_ctx(g),
                "Label": int(g[task].iloc[-1])
            })
            .apply(pd.Series)
            .reset_index(drop=True)
            .merge(rc_df[["PatientID", "Reasoning_chain"]], on="PatientID")
    )

    task_desc = TASK_DESCRIPTIONS[task].strip()

    preamble = (
        "Given the following task description and patient EHR context, "
        "provide a step-by-step reasoning process that leads to the real "
        "situation of the patient and the predicted outcome (0 or 1).\n"
    )

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

    with output_jsonl.open("w", encoding="utf-8") as fw:
        for _, row in merged.iterrows():
            user_content = (
                preamble +
                task_desc +
                "\n\n# Patient EHR Context\n" +
                row["Input"] +
                "\n\n" +
                guideline
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content},
                {"role": "assistant",
                 "content": row["Reasoning_chain"].strip() + f"\n\n# Prediction Result #\n\n {row['Label']}"}
            ]
            json.dump({"messages": messages}, fw, ensure_ascii=False)
            fw.write("\n")

    print(f"Fine-tune dataset saved → {output_jsonl.resolve()} "
          f"({len(merged)} patients)")
    return output_jsonl


if __name__ == "__main__":
    build_finetune_dataset(
        cmps_csv=f"{ROOT}/data/{DATASET}/preprocessed/splits/train_cmps.csv",
        labels_csv=f"{ROOT}/data/{DATASET}/preprocessed/splits/train_labels.csv",
        reasoning_jsonl=f"{ROOT}/data/{DATASET}/preprocessed/reasoning_outputs/{TASK.lower()}_reasoning_chain.jsonl",
        output_jsonl=f"{ROOT}/data/{DATASET}/preprocessed/reasoning_outputs/finetune_{DATASET}_{TASK.lower()}_data.jsonl",
        task=TASK,
        sample_size=10_000,
        random_state=42,
    )
