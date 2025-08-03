# config

ROOT = "EHR-TARS-main"
DATASET = "mimic4"  # mimic3 or mimic4
TASK = "Outcome"    # Outcome or Readmission
SEED = 42

# time-aware score
if DATASET == "mimic3":
    BLOCK_LEN = [5, 10, "full"]
    POOL_DIM  = 24
elif DATASET == "mimic4":
    BLOCK_LEN = [3, 5, "full"]
    POOL_DIM  = 12
else:
    raise ValueError(f"Unknown dataset: {DATASET}")
LOS_TOLERANCE = 0.25

# train
EPOCHS      = 30
BATCH_SIZE  = 256
LR          = 0.001
HIDDEN      = 128
MLP_HIDDEN  = 64
PRECISION   = "16-mixed"

# reasoning-chain-generation
MODEL_DIR = "models/Qwen2.5-32B-Instruct-GPTQ-Int4"   # generate reasoning chain
SPLIT = "train" # train or val

# main-test
FT_MODEL_DIR = "export_models/qwen2.5_7b_ins_mimic3_readmission_p7000"