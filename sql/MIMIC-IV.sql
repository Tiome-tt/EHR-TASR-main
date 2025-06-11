-- Construct time-series EHR
DROP TABLE IF EXISTS format_mimic4;

CREATE TABLE format_mimic4 AS

WITH base_info AS (
  SELECT 
    p.subject_id AS patientid,
    a.hadm_id AS visitid,
    a.admittime AS admissiontime,
    a.dischtime AS dischargetime,
    CASE WHEN a.deathtime IS NOT NULL THEN 1 ELSE 0 END AS outcome,
    ROUND(EXTRACT(EPOCH FROM (a.dischtime - a.admittime)) / 86400.0, 2) AS los,
    CASE 
      WHEN LEAD(a.admittime) OVER (PARTITION BY p.subject_id ORDER BY a.admittime) IS NOT NULL 
      THEN 1 ELSE 0 
    END AS readmission,
    CASE WHEN LOWER(p.gender) = 'm' THEN 1 ELSE 0 END AS sex,
    CASE 
      WHEN p.anchor_age IS NOT NULL AND p.anchor_year IS NOT NULL THEN
        ROUND((p.anchor_age + DATE_PART('year', a.admittime) - p.anchor_year)::numeric, 1)
      ELSE NULL
    END AS age
  FROM mimiciv_hosp.patients p
  JOIN mimiciv_hosp.admissions a ON p.subject_id = a.subject_id
),

time_windows AS (
  SELECT *,
         ROW_NUMBER() OVER (PARTITION BY patientid, visitid ORDER BY record_start) AS recordtime
  FROM (
    SELECT
      bi.patientid,
      bi.visitid,
      bi.admissiontime,
      bi.dischargetime,
      bi.outcome,
      bi.los,
      bi.readmission,
      bi.sex,
      bi.age,
      generate_series(
        bi.admissiontime,
        bi.dischargetime - interval '1 second',
        interval '12 hours'
      ) AS record_start
    FROM base_info bi
  ) t
),

all_events AS (
  SELECT 
    subject_id, 
    hadm_id, 
    charttime AS time, 
    itemid, 
    valuenum, 
    value AS chartvalue  
  FROM mimiciv_icu.chartevents
  WHERE itemid IN (
    224639, 226512, 226730, 223951, 224308,
    220739, 223900, 223901,
    220051, 220050, 220052, 220045, 224690, 223762,
    223761, 676, 678, 229280
  ) AND (valuenum IS NOT NULL OR value IS NOT NULL)

  UNION ALL

  SELECT 
    subject_id, 
    hadm_id, 
    charttime AS time, 
    itemid, 
    valuenum, 
    NULL AS chartvalue 
  FROM mimiciv_hosp.labevents
  WHERE itemid IN (50817, 50809, 50820)
),

windowed_events AS (
  SELECT
    tw.*,
    ae.itemid,
    ae.valuenum,
    ae.chartvalue 
  FROM time_windows tw
  LEFT JOIN all_events ae
    ON ae.subject_id = tw.patientid AND ae.hadm_id = tw.visitid
    AND ae.time >= tw.record_start AND ae.time < tw.record_start + interval '12 hours'
),

pivoted AS (
  SELECT
    patientid,
    visitid,
    recordtime,
    record_start AS "RecordTimestamp",
    admissiontime,
    dischargetime,
    outcome,
    los,
    readmission,
    sex,
    age,

    MAX(CASE WHEN itemid in (224639, 226512) THEN valuenum END) AS weight,
    MAX(CASE WHEN itemid = 226730 THEN valuenum END) AS height,

    MAX(CASE 
      WHEN itemid IN (223951, 224308) AND LOWER(chartvalue) = 'normal <3 seconds' THEN 1 
      ELSE 0 
    END) AS capillary_0,

    MAX(CASE 
      WHEN itemid IN (223951, 224308) AND LOWER(chartvalue) = 'abnormal >3 seconds' THEN 1 
      ELSE 0 
    END) AS capillary_1,

    MAX(CASE WHEN itemid = 220739 AND valuenum = 1 THEN 1 ELSE 0 END) AS gcs_eye_1,
    MAX(CASE WHEN itemid = 220739 AND valuenum = 2 THEN 1 ELSE 0 END) AS gcs_eye_2,
    MAX(CASE WHEN itemid = 220739 AND valuenum = 3 THEN 1 ELSE 0 END) AS gcs_eye_3,
    MAX(CASE WHEN itemid = 220739 AND valuenum = 4 THEN 1 ELSE 0 END) AS gcs_eye_4,

    MAX(CASE WHEN itemid = 223900 AND valuenum = 1 THEN 1 ELSE 0 END) AS gcs_verbal_1,
    MAX(CASE WHEN itemid = 223900 AND valuenum = 2 THEN 1 ELSE 0 END) AS gcs_verbal_2,
    MAX(CASE WHEN itemid = 223900 AND valuenum = 3 THEN 1 ELSE 0 END) AS gcs_verbal_3,
    MAX(CASE WHEN itemid = 223900 AND valuenum = 4 THEN 1 ELSE 0 END) AS gcs_verbal_4,
    MAX(CASE WHEN itemid = 223900 AND valuenum = 5 THEN 1 ELSE 0 END) AS gcs_verbal_5,

    MAX(CASE WHEN itemid = 223901 AND valuenum = 1 THEN 1 ELSE 0 END) AS gcs_motor_1,
    MAX(CASE WHEN itemid = 223901 AND valuenum = 2 THEN 1 ELSE 0 END) AS gcs_motor_2,
    MAX(CASE WHEN itemid = 223901 AND valuenum = 3 THEN 1 ELSE 0 END) AS gcs_motor_3,
    MAX(CASE WHEN itemid = 223901 AND valuenum = 4 THEN 1 ELSE 0 END) AS gcs_motor_4,
    MAX(CASE WHEN itemid = 223901 AND valuenum = 5 THEN 1 ELSE 0 END) AS gcs_motor_5,
    MAX(CASE WHEN itemid = 223901 AND valuenum = 6 THEN 1 ELSE 0 END) AS gcs_motor_6,

    (
      MAX(CASE WHEN itemid = 220739 THEN valuenum END) + 
      MAX(CASE WHEN itemid = 223900 THEN valuenum END) + 
      MAX(CASE WHEN itemid = 223901 THEN valuenum END)
    ) AS gcs_total,

    MAX(CASE WHEN itemid = 220051 THEN valuenum END) AS diastolic_bp,
    MAX(CASE WHEN itemid = 220050 THEN valuenum END) AS systolic_bp,
    MAX(CASE WHEN itemid = 220052 THEN valuenum END) AS mean_bp,
    MAX(CASE WHEN itemid = 220045 THEN valuenum END) AS heart_rate,
    MAX(CASE WHEN itemid = 224690 THEN valuenum END) AS respiratory_rate,
    MAX(CASE WHEN itemid IN (223762, 676, 223761, 678) THEN 
      CASE 
        WHEN itemid IN (676, 223762) THEN valuenum
        WHEN itemid IN (678, 223761) THEN ROUND(((valuenum - 32) * 5.0 / 9.0)::numeric, 2)
      END
    END) AS temperature,
    MAX(CASE WHEN itemid = 50817 THEN valuenum END) AS spo2,
    MAX(CASE WHEN itemid = 229280 THEN valuenum END) AS fio2,
    MAX(CASE WHEN itemid = 50809 THEN valuenum END) AS glucose,
    MAX(CASE WHEN itemid = 50820 THEN valuenum END) AS ph
  FROM windowed_events
  GROUP BY patientid, visitid, recordtime, record_start, admissiontime, dischargetime, outcome, los, readmission, sex, age
),

conditions AS (
  SELECT
    d.subject_id,
    d.hadm_id,
    STRING_AGG(d.icd_code, ';') AS Conditions_ICD9,
    STRING_AGG(cd.long_title, ';') AS Conditions_Long
  FROM mimiciv_hosp.diagnoses_icd d
  JOIN mimiciv_hosp.d_icd_diagnoses cd
    ON d.icd_code = cd.icd_code AND d.icd_version = cd.icd_version
  WHERE d.icd_version = 9
  GROUP BY d.subject_id, d.hadm_id
),

medications AS (
  SELECT
    me.subject_id,
    me.hadm_id,
    STRING_AGG(me.drug, ';') AS Medications
  FROM mimiciv_hosp.prescriptions me
  GROUP BY me.subject_id, me.hadm_id
),

procedures AS (
  SELECT
    pr.subject_id,
    pr.hadm_id,
    STRING_AGG(pr.icd_code, ';') AS Procedures_ICD9,
    STRING_AGG(dp.long_title, ';') AS Procedures_Long
  FROM mimiciv_hosp.procedures_icd pr
  JOIN mimiciv_hosp.d_icd_procedures dp
    ON pr.icd_code = dp.icd_code AND pr.icd_version = dp.icd_version
  WHERE pr.icd_version = 9
  GROUP BY pr.subject_id, pr.hadm_id
)

SELECT
  p.patientid AS "PatientID",
  p.visitid AS "VisitID",
  p.recordtime AS "RecordTime",
  p."RecordTimestamp",
  p.admissiontime AS "AdmissionTime",
  p.dischargetime AS "DischargeTime",
  p.outcome AS "Outcome",
  p.los AS "LOS",
  p.readmission AS "Readmission",
  p.sex AS "Sex",
  p.age AS "Age",
  p.weight AS "Weight",
  p.height AS "Height",
  p.capillary_0 AS "Capillary refill  rate-> less than 3s",
  p.capillary_1 AS "Capillary refill rate-> greater than 3s",
  p.gcs_eye_1 AS "Glascow coma scale eye opening->1 No Response",
  p.gcs_eye_2 AS "Glascow coma scale eye opening->2 To Pain",
  p.gcs_eye_3 AS "Glascow coma scale eye opening->3 To speech",
  p.gcs_eye_4 AS "Glascow coma scale eye opening->4 Spontaneously",
  p.gcs_verbal_1 AS "Glascow coma scale verbal response->1 No Response",
  p.gcs_verbal_2 AS "Glascow coma scale verbal response->2 Incomprehensible sounds",
  p.gcs_verbal_3 AS "Glascow coma scale verbal response->3 Inappropriate Words",
  p.gcs_verbal_4 AS "Glascow coma scale verbal response->4 Confused",
  p.gcs_verbal_5 AS "Glascow coma scale verbal response->5 Oriented",
  p.gcs_motor_1 AS "Glascow coma scale motor response->1 No Response",
  p.gcs_motor_2 AS "Glascow coma scale motor response->2 Abnormal extension",
  p.gcs_motor_3 AS "Glascow coma scale motor response->3 Abnormal Flexion",
  p.gcs_motor_4 AS "Glascow coma scale motor response->4 Flex-withdraws",
  p.gcs_motor_5 AS "Glascow coma scale motor response->5 Localizes Pain",
  p.gcs_motor_6 AS "Glascow coma scale motor response->6 Obeys Commands",
  CASE WHEN p.gcs_total = 3 THEN 1 ELSE 0 END AS "Glascow coma scale total->3",
  CASE WHEN p.gcs_total = 4 THEN 1 ELSE 0 END AS "Glascow coma scale total->4",
  CASE WHEN p.gcs_total = 5 THEN 1 ELSE 0 END AS "Glascow coma scale total->5",
  CASE WHEN p.gcs_total = 6 THEN 1 ELSE 0 END AS "Glascow coma scale total->6",
  CASE WHEN p.gcs_total = 7 THEN 1 ELSE 0 END AS "Glascow coma scale total->7",
  CASE WHEN p.gcs_total = 8 THEN 1 ELSE 0 END AS "Glascow coma scale total->8",
  CASE WHEN p.gcs_total = 9 THEN 1 ELSE 0 END AS "Glascow coma scale total->9",
  CASE WHEN p.gcs_total = 10 THEN 1 ELSE 0 END AS "Glascow coma scale total->10",
  CASE WHEN p.gcs_total = 11 THEN 1 ELSE 0 END AS "Glascow coma scale total->11",
  CASE WHEN p.gcs_total = 12 THEN 1 ELSE 0 END AS "Glascow coma scale total->12",
  CASE WHEN p.gcs_total = 13 THEN 1 ELSE 0 END AS "Glascow coma scale total->13",
  CASE WHEN p.gcs_total = 14 THEN 1 ELSE 0 END AS "Glascow coma scale total->14",
  CASE WHEN p.gcs_total = 15 THEN 1 ELSE 0 END AS "Glascow coma scale total->15",
  p.diastolic_bp AS "Diastolic blood pressure",
  p.systolic_bp AS "Systolic blood pressure",
  p.mean_bp AS "Mean blood pressure",
  p.heart_rate AS "Heart Rate",
  p.respiratory_rate AS "Respiratory rate",
  p.temperature AS "Temperature",
  p.spo2 AS "Oxygen saturation",
  p.fio2 AS "Fraction inspired oxygen",
  p.glucose AS "Glucose",
  p.ph AS "PH",
  CASE WHEN p.recordtime = 1 THEN c.Conditions_ICD9 ELSE NULL END AS "Conditions_ICD9",
  CASE WHEN p.recordtime = 1 THEN c.Conditions_Long ELSE NULL END AS "Conditions_Long",
  CASE WHEN p.recordtime = 1 THEN m.Medications ELSE NULL END AS "Medications",
  CASE WHEN p.recordtime = 1 THEN pr.Procedures_ICD9 ELSE NULL END AS "Procedures_ICD9",
  CASE WHEN p.recordtime = 1 THEN pr.Procedures_Long ELSE NULL END AS "Procedures_Long"
FROM pivoted p
LEFT JOIN conditions c ON p.patientid = c.subject_id AND p.visitid = c.hadm_id
LEFT JOIN medications m ON p.patientid = m.subject_id AND p.visitid = m.hadm_id
LEFT JOIN procedures pr ON p.patientid = pr.subject_id AND p.visitid = pr.hadm_id
ORDER BY "PatientID", "VisitID", "RecordTime";


-- De-identification
SELECT setseed(0.42);  

DROP TABLE IF EXISTS format_mimic4_anonymized;

CREATE TABLE format_mimic4_anonymized AS
WITH unique_patients AS (
  SELECT DISTINCT "PatientID" FROM format_mimic4
),
patient_map AS (
  SELECT 
    "PatientID",
    ROW_NUMBER() OVER (ORDER BY RANDOM()) + 10000 AS new_patient_id
  FROM unique_patients
),
visit_ordered AS (
  SELECT 
    *,
    ROW_NUMBER() OVER (PARTITION BY "PatientID" ORDER BY "AdmissionTime") AS visit_number
  FROM format_mimic4
),
final_visit AS (
  SELECT 
    *,
    DENSE_RANK() OVER (PARTITION BY "PatientID" ORDER BY "AdmissionTime") AS visitid
  FROM visit_ordered
),
joined AS (
  SELECT 
    pm.new_patient_id AS "PatientID",
    fv.visitid AS "VisitID",  
    fv."RecordTime",
    fv."RecordTimestamp",
    fv."AdmissionTime",
    fv."DischargeTime",
    fv."Outcome",
    fv."LOS",
    fv."Readmission",
    fv."Sex",
    fv."Age",
    fv."Weight",
    fv."Height",
    fv."Capillary refill  rate-> less than 3s",
    fv."Capillary refill rate-> greater than 3s",
    fv."Glascow coma scale eye opening->1 No Response",
    fv."Glascow coma scale eye opening->2 To Pain",
    fv."Glascow coma scale eye opening->3 To speech",
    fv."Glascow coma scale eye opening->4 Spontaneously",
    fv."Glascow coma scale verbal response->1 No Response",
    fv."Glascow coma scale verbal response->2 Incomprehensible sounds",
    fv."Glascow coma scale verbal response->3 Inappropriate Words",
    fv."Glascow coma scale verbal response->4 Confused",
    fv."Glascow coma scale verbal response->5 Oriented",
    fv."Glascow coma scale motor response->1 No Response",
    fv."Glascow coma scale motor response->2 Abnormal extension",
    fv."Glascow coma scale motor response->3 Abnormal Flexion",
    fv."Glascow coma scale motor response->4 Flex-withdraws",
    fv."Glascow coma scale motor response->5 Localizes Pain",
    fv."Glascow coma scale motor response->6 Obeys Commands",
    fv."Glascow coma scale total->3",
    fv."Glascow coma scale total->4",
    fv."Glascow coma scale total->5",
    fv."Glascow coma scale total->6",
    fv."Glascow coma scale total->7",
    fv."Glascow coma scale total->8",
    fv."Glascow coma scale total->9",
    fv."Glascow coma scale total->10",
    fv."Glascow coma scale total->11",
    fv."Glascow coma scale total->12",
    fv."Glascow coma scale total->13",
    fv."Glascow coma scale total->14",
    fv."Glascow coma scale total->15",
    fv."Diastolic blood pressure",
    fv."Systolic blood pressure",
    fv."Mean blood pressure",
    fv."Heart Rate",
    fv."Respiratory rate",
    fv."Temperature",
    fv."Oxygen saturation",
    fv."Fraction inspired oxygen",
    fv."Glucose",
    fv."PH",
    fv."Conditions_ICD9",
    fv."Conditions_Long",
    fv."Medications",
    fv."Procedures_ICD9",
    fv."Procedures_Long"
  FROM final_visit fv
  JOIN patient_map pm ON fv."PatientID" = pm."PatientID"
)

SELECT * FROM joined
ORDER BY "PatientID", "VisitID", "RecordTime";

