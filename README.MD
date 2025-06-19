Code for paper "waiting for name"

### 🚀quick start
✅Step 1. Dataset Formatting

Due to regulations, we cannot provide the MIMIC dataset. You will need to download it yourself from the official website.

1).Download the MIMIC-III and MIMIC-IV datasets from their official website and import them into PostgreSQL.

2).Run `sql/MIMIC-III.sql` and `sql/MIMIC-IV.sql` in PostgreSQL, and export the results to CSV files. Then, place the generated CSV files in the data directory, for example:
`data/mimic3/format_mimic3_anonymized.csv` and `data/mimic4/format_mimic4_anonymized.csv`.

✅Step 2. Environment Setup

You can easily set up the environment by installing the required dependencies using the provided `requirements.txt` file by running `pip install -r requirements.txt` in the terminal.

✅Step 3. Data Preprocessing

Run `data/mimic3/preprocessing_mimic3.ipynb` and `data/mimic3/preprocessing_mimic3.ipynb` to preprocess the data

✅Step 4.
