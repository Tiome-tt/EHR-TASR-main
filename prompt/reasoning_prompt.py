from config.config import TASK

TASK_DESCRIPTIONS = {
    "Outcome": """
Outcome Task: Predict in-hospital mortality (1 = death, 0 = survival) for each hospital visit using the patient’s conditions, medications, and procedures before discharge.
Note: Focus on combinations of conditions, procedures, and medications that indicate critical illness or a high risk of mortality. Consider how these factors interact and potentially exacerbate each other. Only the patients with extremely very high risk of mortality should be predicted as 1.
""",
    "Readmission": """
Readmission Task: Predict whether a patient will be readmitted within 30 days after discharge (1 = readmission, 0 = no readmission) for each hospital visit using the patient’s conditions, medications, and procedures before discharge.
Note: Analyze the information comprehensively to determine the likelihood of readmission. The goal is to accurately distinguish between patients who are likely to be readmitted and those who are not.
""",
    "LOS": """
LOS Task: Predict the patient’s total length of stay (in days) for each hospital visit using the conditions, medications, and procedures before discharge.
"""
}

if TASK not in TASK_DESCRIPTIONS:
    raise ValueError(f"Unknown task: {TASK}")

TASK_DESCRIPTION = TASK_DESCRIPTIONS[TASK].strip()

SYSTEM_PROMPT = """
    You are a clinical-reasoning assistant that reads structured EHR data and outputs a concise, evidence-based reasoning chain for a specified prediction task.
"""

USER_PROMPT = f"""
    Given the following task description, patient EHR context and ground-truth, provide a step-by-step reasoning process that leads to the real situation of patient:

# Task description #
{TASK_DESCRIPTION}

# Patient EHR context #
{{EHR_CONTEXT}}

# Ground-truth #
{{GROUND_TRUTH}}

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

Response:
"""
