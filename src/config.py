import os
import sys
from pathlib import Path
from dotenv import load_dotenv


# Anchor paths to this file, not to the current working directory.
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

load_dotenv(PROJECT_ROOT / ".env")

dataset_path = os.getenv("DATASET_PATH")
if not dataset_path:
    raise RuntimeError("DATASET_PATH not found in .env")

DATASET_ROOT = Path(dataset_path).expanduser().resolve()

IMAGE_SIZE = 256
BATCH_SIZE = 32
NUM_WORKERS = 8
EPOCHS = 30
PATIENCE = 7
LEARNING_RATE = 1e-4
RANDOM_SEED = 42

USE_AMP = True
AMP_DTYPE = "float16"

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 1 - TRAIN_SPLIT - VAL_SPLIT