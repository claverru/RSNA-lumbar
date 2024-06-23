from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pydicom
import torch
import cv2

from src import constants


def cat_preds(preds: List[Dict[str, torch.Tensor]], f=lambda x: torch.concat(x, dim=0)) -> Dict[str, torch.Tensor]:
    result = {}
    for k in preds[0]:
        result[k] = f([pred[k] for pred in preds])
    return result


def load_dcm_img(path: Path, add_channels: bool = True, size: Optional[int] = None) -> np.ndarray:
    dicom = pydicom.read_file(path)
    data: np.ndarray = dicom.pixel_array
    if dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data
    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    if size is not None:
        data = cv2.resize(data, (size, size), interpolation=cv2.INTER_CUBIC)

    if add_channels:
        data = data[..., None].repeat(3, -1)
    return data


def load_train(train_path: Path = constants.TRAIN_PATH) -> pd.DataFrame:
    train = pd.read_csv(train_path)
    col_map = {0: "severity", "level_1": "condition_level"}
    train = train.set_index("study_id").stack().reset_index().rename(columns=col_map)
    train.name = "train"
    return train


def get_images_df(img_dir: Path = constants.TRAIN_IMG_DIR) -> pd.DataFrame:
    def get_record(img_path):
        return {
            "study_id": int(img_path.parent.parent.stem),
            "series_id": int(img_path.parent.stem),
            "instance_number": int(img_path.stem),
        }
    records = [get_record(path) for path in img_dir.rglob("*.dcm")]
    return pd.DataFrame.from_dict(records).sort_values(["study_id", "series_id", "instance_number"])
