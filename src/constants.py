
from pathlib import Path

ROOT = Path("data")

TRAIN_IMG_DIR = ROOT / "train_images"
TEST_IMG_DIR = ROOT / "train_images"

COOR_PATH = ROOT / "train_label_coordinates.csv"
TRAIN_PATH = ROOT / "train.csv"


CONDITION_LEVEL = [
    'left_neural_foraminal_narrowing_l1_l2',
    'left_neural_foraminal_narrowing_l2_l3',
    'left_neural_foraminal_narrowing_l3_l4',
    'left_neural_foraminal_narrowing_l4_l5',
    'left_neural_foraminal_narrowing_l5_s1',
    'left_subarticular_stenosis_l1_l2',
    'left_subarticular_stenosis_l2_l3',
    'left_subarticular_stenosis_l3_l4',
    'left_subarticular_stenosis_l4_l5',
    'left_subarticular_stenosis_l5_s1',
    'right_neural_foraminal_narrowing_l1_l2',
    'right_neural_foraminal_narrowing_l2_l3',
    'right_neural_foraminal_narrowing_l3_l4',
    'right_neural_foraminal_narrowing_l4_l5',
    'right_neural_foraminal_narrowing_l5_s1',
    'right_subarticular_stenosis_l1_l2',
    'right_subarticular_stenosis_l2_l3',
    'right_subarticular_stenosis_l3_l4',
    'right_subarticular_stenosis_l4_l5',
    'right_subarticular_stenosis_l5_s1',
    'spinal_canal_stenosis_l1_l2',
    'spinal_canal_stenosis_l2_l3',
    'spinal_canal_stenosis_l3_l4',
    'spinal_canal_stenosis_l4_l5',
    'spinal_canal_stenosis_l5_s1'
]
