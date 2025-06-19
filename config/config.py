# config

ROOT = "/home/user1/MXY/EHRScore"
DATASET = "mimic3" # mimic3 or mimic4
TASK = "Outcome" # Outcome or LOS or Readmission
SEED = 42

# time-aware score
BLOCK_LEN = 5
if DATASET == "mimic3":
    POOL_DIM  = 24
elif DATASET == "mimic4":
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
