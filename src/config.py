import os

from dotenv import load_dotenv

from utils import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")

dataset_path = os.getenv("DATASET_PATH")
if not dataset_path:
    raise RuntimeError("DATASET_PATH not found in .env")

chexpert_path = os.getenv("CHEXPERT_PATH")  # optional; the CheXpert-v1.0-small directory

NUM_WORKERS = 6
EVAL_NUM_WORKERS = 4  # not persistent (see train.py), so these exist only during evaluation
RANDOM_SEED = 42
VAL_SPLIT = 0.15
