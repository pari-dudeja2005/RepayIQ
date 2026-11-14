
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports"

RAW_DATA_PATH = RAW_DATA_DIR / "credit_default.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "repayiq_processed.csv"

MODEL_PATH = MODEL_DIR / "repayiq_model.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.20
