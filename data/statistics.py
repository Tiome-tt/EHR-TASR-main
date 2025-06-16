import os
import pandas as pd
from collections import Counter

print(os.getcwd())
os.chdir('/home/user1/MXY/EHRScore') # Change to the project directory

mimic3_data_path = "data/mimic3/preprocessed/timeseries_mimic3.csv"
mimic4_data_path = "data/mimic4/preprocessed/timeseries_mimic4.csv"
mimic3_CMPs_path = "data/mimic3/preprocessed/CMPs_mimic3.csv"
mimic4_CMPs_path = "data/mimic4/preprocessed/CMPs_mimic4.csv"

mimic3_df = pd.read_csv(mimic3_data_path)
mimic4_df = pd.read_csv(mimic4_data_path)

mimic3_cmps = pd.read_csv(mimic3_CMPs_path)
mimic4_cmps = pd.read_csv(mimic4_CMPs_path)

# Basic Information Statistics

def analyze_timeseries(df, name="Dataset"):
    total_patients = df['PatientID'].nunique()

    avg_visits_per_patient = df.groupby('PatientID')['VisitID'].nunique().mean()

    dead = int(df.groupby('PatientID')['Outcome'].max().sum())
    alive = total_patients - dead
    mortality_rate = dead / total_patients

    avg_los = (
        df.groupby(['PatientID', 'VisitID'])['LOS'].first()
        .groupby(level='PatientID')
        .mean()
        .mean()
    )

    readmissions = df.groupby(['PatientID', 'VisitID'])['Readmission'].first().groupby('PatientID').mean().mean()

    avg_records_per_visit_per_patient = (
        df.groupby('PatientID').apply(lambda g: len(g) / g['VisitID'].nunique())
        .mean()
    )

    print(f"📊 {name} Statistics:")
    print(f"Total Patients: {total_patients}")
    print(f"Avg Visits per Patient: {avg_visits_per_patient:.2f}")
    print(f"Avg Records per Visit per Patient: {avg_records_per_visit_per_patient:.2f}")
    print(f"Dead: {dead}, Alive: {alive}")
    print(f"Mortality Rate: {mortality_rate:.2%}")
    print(f"Avg LOS: {avg_los:.2f} days")
    print(f"Avg Readmission Rate: {readmissions:.2%}\n")

analyze_timeseries(mimic3_df, "MIMIC-III")
analyze_timeseries(mimic4_df, "MIMIC-IV")

# Top-k Conditions, Medications, and Procedures Statistics

def get_deceased_patients(timeseries_df):
    patient_outcome = timeseries_df.groupby('PatientID')['Outcome'].max()
    deceased_patients = patient_outcome[patient_outcome == 1].index.tolist()
    return deceased_patients

def top_conditions_procedures(cmps_df, name="Dataset"):
    cond_counter = Counter()   # (Long Description, ICD Code)
    med_counter = Counter()    # Medication Name
    proc_counter = Counter()   # (Long Description, ICD Code)

    for _, row in cmps_df.iterrows():
        # Conditions_ICD9 + Conditions_Long
        if pd.notna(row['Conditions_ICD9']) and pd.notna(row['Conditions_Long']):
            icds = str(row['Conditions_ICD9']).split(";")
            longs = str(row['Conditions_Long']).split(";")
            for icd, long in zip(icds, longs):
                key = (long.strip(), icd.strip())
                cond_counter[key] += 1

        # Medications
        if pd.notna(row['Medications']):
            meds = str(row['Medications']).split(";")
            for med in meds:
                med_counter[med.strip()] += 1

        # Procedures_ICD9 + Procedures_Long
        if pd.notna(row['Procedures_ICD9']) and pd.notna(row['Procedures_Long']):
            icds = str(row['Procedures_ICD9']).split(";")
            longs = str(row['Procedures_Long']).split(";")
            for icd, long in zip(icds, longs):
                key = (long.strip(), icd.strip())
                proc_counter[key] += 1

    print(f"\n🔥 Top 10 Conditions (Name, ICD Code, Count) in {name}:")
    for (cond_name, icd_code), count in cond_counter.most_common(10):
        print(f"{cond_name} ({icd_code}): {count}")

    print(f"\n🔥 Top 10 Medications (Name, Count) in {name}:")
    for med_name, count in med_counter.most_common(10):
        print(f"{med_name}: {count}")

    print(f"\n🔥 Top 10 Procedures (Name, ICD Code, Count) in {name}:")
    for (proc_name, icd_code), count in proc_counter.most_common(10):
        print(f"{proc_name} ({icd_code}): {count}")

# MIMIC-III
print("📊 [MIMIC-III] Statistics for All Patients:")
top_conditions_procedures(mimic3_cmps, "MIMIC-III")

deceased_ids_iii = get_deceased_patients(mimic3_df)
deceased_cmps_iii = mimic3_cmps[mimic3_cmps['PatientID'].isin(deceased_ids_iii)]

print("\n\n💀 [MIMIC-III] Statistics for Dead Patients:")
top_conditions_procedures(deceased_cmps_iii, "MIMIC-III")

# MIMIC-IV
print("\n\n📊 [MIMIC-IV] Statistics for All Patients:")
top_conditions_procedures(mimic4_cmps, "MIMIC-IV")

deceased_ids_iv = get_deceased_patients(mimic4_df)
deceased_cmps_iv = mimic4_cmps[mimic4_cmps['PatientID'].isin(deceased_ids_iv)]

print("\n\n💀 [MIMIC-IV] Statistics for Dead Patients:")
top_conditions_procedures(deceased_cmps_iv, "MIMIC-IV")


# Physiological Features Statistics
def analyze_physiological_stats(df, name="Dataset"):
    physio_features = [
        'Age', 'Diastolic blood pressure', 'Systolic blood pressure',
        'Mean blood pressure', 'Heart Rate', 'Respiratory rate',
        'Temperature', 'Oxygen saturation', 'Fraction inspired oxygen',
        'Glucose', 'PH'
    ]

    print(f"\n🧬 {name} - Min and Max Values for Physiological Features:")

    for feature in physio_features:
        if feature in df.columns:
            min_val = df[feature].min()
            max_val = df[feature].max()
            print(f"{feature}: Min={min_val:.2f}, Max={max_val:.2f}")
        else:
            print(f"{feature}: ❌ Not found in dataset")

# Analyze physiological features
analyze_physiological_stats(mimic3_df, "MIMIC-III")
analyze_physiological_stats(mimic4_df, "MIMIC-IV")
