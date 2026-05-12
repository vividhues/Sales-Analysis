from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
REPORT_DIR = ROOT_DIR / "reports"
MODEL_DIR = ROOT_DIR / "models"

SALES_CSV = DATA_DIR / "sample_sales.csv"
DATE_COLUMN = "date"
TARGET_COLUMN = "sales"

# ML model parameters go here, alter based on dataset
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 200

# plotting configs, like image dpi
FIGURE_DPI = 150
COLOR_PALETTE = "Set2"


class Settings:
    root_dir = ROOT_DIR
    data_dir = DATA_DIR
    report_dir = REPORT_DIR
    model_dir = MODEL_DIR
    sales_csv = SALES_CSV
    date_column = DATE_COLUMN
    target_column = TARGET_COLUMN
    test_size = TEST_SIZE
    random_state = RANDOM_STATE
    n_estimators = N_ESTIMATORS
    figure_dpi = FIGURE_DPI
    color_palette = COLOR_PALETTE