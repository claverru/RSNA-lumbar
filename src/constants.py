from pathlib import Path

ROOT = Path("data")

TRAIN_IMG_DIR = ROOT / "train_images"
TEST_IMG_DIR = ROOT / "train_images"

COOR_PATH = ROOT / "train_label_coordinates.csv"
TRAIN_PATH = ROOT / "train.csv"
DESC_PATH = ROOT / "train_series_descriptions.csv"
T2_COOR_PATH = ROOT / "t2_coordinates.csv"

FEATS_PATH = ROOT / "preds/feats.parquet"
META_PATH = ROOT / "metadata.csv"
KEYPOINTS_PATH = ROOT / "preds_keypoints.parquet"

LEVELS_PATH = ROOT / "preds_levels.parquet"

VAL_PREDS_NAME = "val_preds.csv"

BASIC_COLS = ["study_id", "series_id", "instance_number"]

CONDITIONS = ["spinal_canal_stenosis", "neural_foraminal_narrowing", "subarticular_stenosis"]

SEVERITY2LABEL = {"Normal/Mild": 0, "Moderate": 1, "Severe": 2}

DESCRIPTIONS = ["Sagittal T2/STIR", "Sagittal T1", "Axial T2"]

LEVELS = ["l1_l2", "l2_l3", "l3_l4", "l4_l5", "l5_s1"]

SIDES = ["right", "left"]

LEVELS_SIDES = [f"{lvl}_{side}" for lvl in LEVELS for side in SIDES]

CONDITIONS_COMPLETE = [
    "left_neural_foraminal_narrowing",
    "left_subarticular_stenosis",
    "right_neural_foraminal_narrowing",
    "right_subarticular_stenosis",
    "spinal_canal_stenosis",
]

CONDITION_LEVEL = [f"{c}_{lvl}" for c in CONDITIONS_COMPLETE for lvl in LEVELS]
