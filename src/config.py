import os

from dotenv import load_dotenv

from utils import PROJECT_ROOT

load_dotenv(PROJECT_ROOT / ".env")

dataset_path = os.getenv("DATASET_PATH")
if not dataset_path:
    raise RuntimeError("DATASET_PATH not found in .env")

chexpert_path = os.getenv("CHEXPERT_PATH")  # optional; the CheXpert-v1.0-small directory

NUM_WORKERS = 8
RANDOM_SEED = 42
VAL_SPLIT = 0.15
