from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

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

    def rotate(self, angle: int) -> Image:
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

    def rotate(self, img: Image, angle: int) -> Keypoint:
        if angle == 0:
            return self
        kp_A = (self.x, self.y, 0, 1)
        x, y = A.augmentations.geometric.functional.keypoint_rotate(kp_A, angle, img.w, img.h)[:2]
        return Keypoint(x, y)

    def get(self):
        return self.x, self.y


def angle_crop_ratio(img: Image, kp: Keypoint, angle: int, size_ratio: int):
    img = img.rotate(angle)
    kp = kp.scale_to_img(img)
    kp = kp.rotate(img, angle)
    patch = crop_ratio(img, kp, size_ratio)
    return patch


def crop_ratio(img: Image, kp: Keypoint, size_ratio: int) -> np.ndarray:
    half_side = max(img.h, img.w) // size_ratio
    xmin, ymin, xmax, ymax = kp.x - half_side, kp.y - half_side, kp.x + half_side, kp.y + half_side
    xmin, ymin, xmax, ymax = max(xmin, 0), max(ymin, 0), min(xmax, img.w), min(ymax, img.h)
    patch = img.get()[ymin:ymax, xmin:xmax]
    return patch


PLANE2SPACING = {"Axial T2": 0.35, "Sagittal T1": 0.70, "Sagittal T2/STIR": 0.72}


def angle_crop_size(img: Image, kp: Keypoint, angle: int, size: int, plane: str = None):
    img = img.rotate(angle)
    kp = kp.scale_to_img(img)
    kp = kp.rotate(img, angle)
    patch = crop_size(img, kp, size, plane)
    return patch


def default_limits(kp: Keypoint, size: int) -> Tuple[int, int, int, int]:
    xmin = kp.x - int(size * 0.5)
    ymin = kp.y - int(size * 0.5)
    xmax = kp.x + int(size * 0.5)
    ymax = kp.y + int(size * 0.5)
    return xmin, ymin, xmax, ymax


def sagittal_t1_limits(kp: Keypoint, size: int) -> Tuple[int, int, int, int]:
    xmin = kp.x - int(size * 0.60)
    xmax = kp.x + int(size * 0.40)
    ymin = kp.y - int(size * 0.23)
    ymax = kp.y + int(size * 0.30)
    return xmin, ymin, xmax, ymax


def sagittal_t2_limits(kp: Keypoint, size: int) -> Tuple[int, int, int, int]:
    xmin = kp.x - int(size * 0.7)
    xmax = kp.x + int(size * 0.3)
    ymin = kp.y - int(size * 0.3)
    ymax = kp.y + int(size * 0.3)
    return xmin, ymin, xmax, ymax


def axial_t2_limits(kp: Keypoint, size: int) -> Tuple[int, int, int, int]:
    xmin = kp.x - int(size * 0.5)
    xmax = kp.x + int(size * 0.5)
    ymin = kp.y - int(size * 0.3)
    ymax = kp.y + int(size * 0.7)
    return xmin, ymin, xmax, ymax


LIMITS = {
    None: default_limits,
    "Sagittal T1": sagittal_t1_limits,
    "Sagittal T2/STIR": sagittal_t2_limits,
    "Axial T2": axial_t2_limits,
}


def crop_size(img: Image, kp: Keypoint, size: int = 96, plane: str = None):
    xmin, ymin, xmax, ymax = LIMITS[plane](kp, size)
    xmin, ymin, xmax, ymax = max(xmin, 0), max(ymin, 0), min(xmax, img.w), min(ymax, img.h)
    patch = img.get()[ymin:ymax, xmin:xmax]
    return patch
