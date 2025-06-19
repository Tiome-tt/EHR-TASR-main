### ⚙️Data Source for Formatting
First, run `sql/MIMIC-III.sql` and `sql/MIMIC-IV.sql` in PostgreSQL, and export the results to CSV files.
*(Before running the scripts, you need to download the MIMIC-III and MIMIC-IV datasets from their official website and import them into PostgreSQL.)*

Then, place the generated CSV files in the data directory, for example:
`data/mimic3/format_mimic3_anonymized.csv` and `data/mimic4/format_mimic4_anonymized.csv`.