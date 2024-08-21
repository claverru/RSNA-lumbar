from pathlib import Path
from typing import List, Tuple

import albumentations as A
import cv2
import lightning as L
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2

from src import constants, utils
from src.patch.utils import PLANE2SPACING, Image, Keypoint, Spacing, angle_crop_size


class Dataset(torch.utils.data.Dataset):
    def __init__(self, train: pd.DataFrame, df: pd.DataFrame, img_dir: Path, img_size: int, transforms: A.Compose):
        self.train = train
        self.df = df
        self.img_dir = img_dir
        self.index = self.train.index if train is not None else df.index
        self.transforms = transforms
        self.img_size = img_size

    def __len__(self):
        return len(self.index)

    def get_target(self, idx):
        if self.train is None:
            return None
        return self.train.loc[idx].apply(torch.tensor).to_dict()

    def get_patch_params(self, idx):
        return self.df.loc[idx].values

    def __getitem__(self, index):
        idx = self.index[index]
        study_id, series_id, instance_number, plane, *_ = idx
        x, y, angle, spacing_x, spacing_y = self.get_patch_params(idx)
        spacing = Spacing(spacing_x, spacing_y)
        img = Image.from_params(study_id, series_id, instance_number, self.img_dir, ".png", spacing=spacing)
        kp = Keypoint(x, y)

        target_spacing = Spacing(PLANE2SPACING[plane], PLANE2SPACING[plane])
        img = img.resize_spacing(target_spacing)
        patch = angle_crop_size(img, kp, angle, self.img_size, plane)
        X = self.transforms(image=patch)["image"]
        target = self.get_target(idx)

        if target is None:
            return X

        return X, target


def get_transforms(img_size):
    return A.Compose(
        [
            A.LongestMaxSize(img_size, interpolation=cv2.INTER_CUBIC),
            A.PadIfNeeded(img_size, img_size, position="center", border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize((0.485,), (0.229,)),
            ToTensorV2(),
        ]
    )


def get_aug_transforms(img_size):
    return A.Compose(
        [
            A.LongestMaxSize(img_size, interpolation=cv2.INTER_CUBIC),
            A.PadIfNeeded(img_size, img_size, position="random", border_mode=cv2.BORDER_CONSTANT, value=0),
            A.HorizontalFlip(p=0.5),
            A.Affine(
                rotate=(-10, 10),
                shear=(-10, 10),
                translate_percent=0.1,
                scale=(0.8, 1.2),
                p=0.3,
                interpolation=cv2.INTER_CUBIC,
                mode=cv2.BORDER_CONSTANT,
            ),
            A.Perspective(0.2, p=0.5, interpolation=cv2.INTER_CUBIC, pad_mode=cv2.BORDER_CONSTANT, pad_val=0),
            A.GaussNoise(var_limit=20, noise_scale_factor=0.90, mean=0, p=0.2),
            A.MotionBlur(blur_limit=(3, 7), p=0.2),
            A.Normalize((0.485,), (0.229,)),
            ToTensorV2(),
        ]
    )


LEVEL2ANGLE = {
    "l1_l2": 9.492165632503335,
    "l2_l3": 6.48974669112818,
    "l3_l4": 0.2562395236298992,
    "l4_l5": -10.442638805859872,
    "l5_s1": -28.4725950244635,
    "right": 0,
    "left": 0,
}


def load_keypoints(
    keypoints_path: Path = constants.KEYPOINTS_PATH, desc_path: Path = constants.DESC_PATH
) -> pd.DataFrame:
    # reshape keypoints
    df = pd.read_parquet(keypoints_path)
    levels_sides = constants.LEVELS + ["left", "right"]
    levels_sides_cols = [f"{side}_{x}" for side in levels_sides for x in ("x", "y")]
    feature_cols = [c for c in df.columns if c.startswith("_f")]
    new_cols = dict(zip(feature_cols, levels_sides_cols))
    df = df.rename(columns=new_cols)
    df = df.melt(id_vars=constants.BASIC_COLS)
    df[["type", "xy"]] = df.pop("variable").str.rsplit("_", n=1, expand=True)
    df = df.pivot(values="value", index=constants.BASIC_COLS + ["type"], columns="xy").reset_index()

    # merge desc filter invalid keypoints
    desc = utils.load_desc(desc_path)
    df = df.merge(desc, how="inner", on=constants.BASIC_COLS[:2])
    is_axial = df["series_description"].eq("Axial T2")
    is_sagittal = ~is_axial
    is_levels = df["type"].isin(constants.LEVELS)
    is_sides = ~is_levels
    df = df[(is_axial & is_sides) | (is_sagittal & is_levels)]
    df["angle"] = df["type"].map(LEVEL2ANGLE).astype(int)

    # final cleaning
    df = df.reset_index(drop=True)
    return df


def load_this_train(train_path: Path = constants.TRAIN_PATH) -> pd.DataFrame:
    df = utils.load_train(train_path)
    df = df.melt(ignore_index=False, var_name="condition_level", value_name="severity").reset_index()
    return df


def load_df(
    keypoints_path: Path = constants.KEYPOINTS_PATH,
    desc_path: Path = constants.DESC_PATH,
    coor_path: Path = constants.COOR_PATH,
    train_path: Path = constants.TRAIN_PATH,
    meta_path: Path = constants.META_PATH,
):
    keypoints = load_keypoints(keypoints_path, desc_path)
    coor = utils.load_coor(coor_path).drop(columns=["level", "x", "y"])
    train = load_this_train(train_path)
    meta = utils.load_meta(meta_path, 0, False, False, False)[
        constants.BASIC_COLS + ["PixelSpacing_0", "PixelSpacing_1"]
    ]

    df = coor.merge(keypoints, how="inner", on=constants.BASIC_COLS + ["type"])
    df = df.merge(train, how="inner", on=["study_id", "condition_level"])
    df = df.merge(meta, how="inner", on=constants.BASIC_COLS)

    common_index = constants.BASIC_COLS + ["series_description", "type"]

    # set X
    x = df[common_index + ["x", "y", "angle", "PixelSpacing_0", "PixelSpacing_1"]].copy().drop_duplicates()
    x = x.set_index(common_index)

    # set Y
    y = df[common_index + ["condition", "severity"]].copy().drop_duplicates()
    y = y.set_index(common_index)

    # Remove laterality
    y["condition"] = y["condition"].str.extract("(" + "|".join(constants.CONDITIONS) + ")")[0]
    y = y.groupby(level=[0, 1, 2, 3, 4]).agg({"severity": "max", "condition": "first"})

    y = y.pivot(columns="condition", values="severity").fillna(-1).astype(int)

    return x, y


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        keypoints_path: Path = constants.KEYPOINTS_PATH,
        coor_path: Path = constants.COOR_PATH,
        train_path: Path = constants.TRAIN_PATH,
        desc_path: Path = constants.DESC_PATH,
        meta_path: Path = constants.META_PATH,
        img_size: int = 224,
        img_dir: Path = constants.TRAIN_IMG_DIR,
        n_splits: int = 5,
        this_split: int = 0,
        batch_size: int = 64,
        num_workers: int = 8,
    ):
        super().__init__()
        self.img_size = img_size
        self.n_splits = n_splits
        self.this_split = this_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_dir = Path(img_dir)
        self.df, self.train = load_df(Path(keypoints_path), Path(desc_path), Path(coor_path), Path(train_path))
        self.train_path = train_path
        self.keypoints_path = Path(keypoints_path)

    def split(self) -> Tuple[List[int], List[int]]:
        return utils.split(self.train, self.n_splits, self.this_split)

    def setup(self, stage: str):
        if stage == "fit":
            train_df, val_df = self.split()
            self.train_ds = Dataset(train_df, self.df, self.img_dir, self.img_size, get_aug_transforms(self.img_size))
            self.val_ds = Dataset(val_df, self.df, self.img_dir, self.img_size, get_transforms(self.img_size))

        if stage == "test":
            pass

        if stage == "predict":
            df = load_keypoints(self.keypoints_path).set_index(constants.BASIC_COLS + ["type"])
            self.predict_ds = Dataset(None, df, self.img_dir, self.img_size, get_transforms(self.img_size))

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=lambda wid: np.random.seed(np.random.get_state()[1][0] + wid),
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=lambda wid: np.random.seed(np.random.get_state()[1][0] + wid),
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.predict_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=lambda wid: np.random.seed(np.random.get_state()[1][0] + wid),
        )


if __name__ == "__main__":
    import yaml

    config = yaml.load(open("configs/patch.yaml"), Loader=yaml.FullLoader)
    dm = DataModule(**config["data"]["init_args"])
    dm.setup("fit")
    for X, target in dm.train_dataloader():
        print()
        print(X.shape, X.dtype)
        print({k: (v.shape, v.dtype) for k, v in target.items()})
        print()
        input()
