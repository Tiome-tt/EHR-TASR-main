# Code for the Paper: **"EHR-TASR: Leveraging Time-Aware Scoring and LLM Reasoning for Clinical Prediction"**

### 🚀 Quick Start

---

#### ✅ Step 1. Dataset Preparation

Due to data-sharing regulations, we cannot provide the MIMIC datasets. Please download them yourself from the official website.

1. Download the **MIMIC-III** and **MIMIC-IV** datasets and import them into **PostgreSQL**.

2. Run the SQL scripts `sql/MIMIC-III.sql` and `sql/MIMIC-IV.sql` in PostgreSQL, then export the query results to CSV files.

3. Place the exported CSV files in the `data` directory, for example:  

   ```
   data/mimic3/format_mimic3_anonymized.csv  
   data/mimic4/format_mimic4_anonymized.csv
   ```

---

#### ✅ Step 2. Environment Setup

Set up your environment by installing the required dependencies:

```bash
pip install -r requirements.txt
```

---

#### ✅ Step 3. Data Preprocessing

1. Modify the configuration file `config/config.py` to specify your task and dataset.
2. Run the preprocessing notebooks:
   - `data/mimic3/preprocessing_mimic3.ipynb`
   - `data/mimic4/preprocessing_mimic4.ipynb`

---

#### ✅ Step 4. Compute Multi-Scale Temporally-Aware Scores & Train Sequence Models

1. Run `timeAwareScore.py` to compute temporal awareness scores.
2. Run `ehr_tas_trainer.py` to train the sequence model.

---

#### ✅ Step 5. Fine-Tune the Local-Reasoning LLM & Perform Final Ensemble Prediction

1. Run `reasoning_generation.py` to deploy the 32B model and generate reasoning chains for training.

2. Run `filter.py` to remove problematic data.

3. Execute `ftData_generate.py` to create the fine-tuning dataset.  
   The generated dataset will be saved to:

   ```
   data/{datasets}/preprocessed/cache
   ```

4. Use [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) to fine-tune your model using this dataset.
## Citation

If you find this paper or code useful for your research, please consider citing our work:

5. Finally, run `main.py` to perform the ensemble prediction.

Note: When running different datasets, you need to delete the [.npz](file://D:\LearningFiles\EHRScore\data\mimic3\preprocessed\cache\val.npz) files in `data/{datasets}/preprocessed/cache`.

---

### BibTeX:
```bibtex
@inproceedings{meng2026ehrtasr,
  title={{EHR-TASR}: Leveraging Time-Aware Scoring and LLM Reasoning for Clinical Prediction},
  author={Meng, Xiaoyang and Zhou*, Xiaoding and Liu, Yang and Zhang*, Weiyu and Guan, Hongjiao and Peng, Xueping and Lu, Wenpeng},
  booktitle={The International Joint Conference on Neural Networks (IJCNN 2026)},
  address={Netherlands},
  year={2026},
  organization={IEEE},
  url={[https://github.com/Tiome-tt/EHR-TASR-main](https://github.com/Tiome-tt/EHR-TASR-main)}
}
