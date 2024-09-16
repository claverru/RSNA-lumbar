from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import albumentations as A
import cv2
import numpy as np

from src import constants, utils


class Spacing:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y


class Image:
    def __init__(self, array: np.ndarray, spacing: Optional[Spacing] = None):
        self.array = array
        self.spacing = spacing

    @property
    def h(self):
        return self.array.shape[0]

    @property
    def w(self):
        return self.array.shape[1]

    def rotate(self, angle: float) -> Image:
        angle = int(angle)
        if angle == 0:
            return self
        array = A.augmentations.geometric.functional.rotate(self.array, angle, cv2.INTER_CUBIC, cv2.BORDER_CONSTANT, 0)
        return Image(array, self.spacing)

    def resize_spacing(self, target_spacing: Spacing) -> Image:
        rescale_x = int(self.spacing.x / target_spacing.x * self.w)
        rescale_y = int(self.spacing.y / target_spacing.y * self.h)
        new_array = cv2.resize(self.array, (rescale_x, rescale_y), interpolation=cv2.INTER_CUBIC)
        return Image(new_array, target_spacing)

    @classmethod
    def from_path(cls, path: Union[str, Path], **kwargs) -> Image:
        path = Path(path)
        suffix = path.suffix
        array = cls.loader()[suffix](path)
        return Image(array, **kwargs)

    @classmethod
    def from_params(
        cls,
        study_id: int,
        series_id: int,
        instance_number: int,
        img_dir: Path = constants.TRAIN_IMG_DIR,
        suffix: str = ".dcm",
        **kwargs,
    ) -> Image:
        path = utils.get_image_path(study_id, series_id, instance_number, img_dir, suffix)
        return cls.from_path(path, **kwargs)

    @classmethod
    def loader(cls) -> dict:
        return {
            ".png": lambda x: cv2.imread(x, cv2.IMREAD_GRAYSCALE),
            ".dcm": lambda x: utils.load_dcm_img(x, False),
        }

    def get(self):
        return self.array


class Keypoint:
    def __init__(self, x: Union[int, float], y: Union[int, float]):
        self.x = x
        self.y = y

    def scale_to_img(self, img: Image) -> Keypoint:
        x = int(self.x * img.w)
        y = int(self.y * img.h)
        return Keypoint(x, y)

    def rotate(self, img: Image, angle: float) -> Keypoint:
        angle = int(angle)
        if angle == 0:
            return self
        kp_A = (self.x, self.y, 0, 1)
        x, y = A.augmentations.geometric.functional.keypoint_rotate(kp_A, angle, img.w, img.h)[:2]
        return Keypoint(x, y)

    def get(self):
        return self.x, self.y


# 96 patch size
PLANE2SPACING = {"Axial T2": 0.35, "Sagittal T1": 0.72, "Sagittal T2/STIR": 0.72}


def angle_crop_size(img: Image, kp: Keypoint, angle: float, size: int, plane: str):
    angle = angle
    center = kp.scale_to_img(img).get()
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    if "Axial" in plane:
        size = (size, size)
    else:
        size = (size, size // 2)

    # Determine the new dimensions
    abs_cos = abs(rotation_matrix[0, 0])
    abs_sin = abs(rotation_matrix[0, 1])
    bound_w = int(size[1] * abs_sin + size[0] * abs_cos)
    bound_h = int(size[1] * abs_cos + size[0] * abs_sin)

    # Update the rotation matrix
    rotation_matrix[0, 2] += bound_w / 2 - center[0]
    rotation_matrix[1, 2] += bound_h / 2 - center[1]

    # Rotate the img
    rotated = cv2.warpAffine(img.get(), rotation_matrix, (bound_w, bound_h))

    # Crop the rotated img
    cropped = cv2.getRectSubPix(rotated, size, (bound_w / 2, bound_h / 2))

    return cropped
