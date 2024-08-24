import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import albumentations as A
import cv2
import lightning as L
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms.functional
from albumentations.pytorch import ToTensorV2

from src import constants, utils

KEYPOINT_KEYS = [
    "axial_right",
    "axial_left",
    "sagittal_l1_l2_left",
    "sagittal_l1_l2_right",
    "sagittal_l2_l3_left",
    "sagittal_l2_l3_right",
    "sagittal_l3_l4_left",
    "sagittal_l3_l4_right",
    "sagittal_l4_l5_left",
    "sagittal_l4_l5_right",
    "sagittal_l5_s1_left",
    "sagittal_l5_s1_right",
]


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: Path = constants.TRAIN_IMG_DIR,
        train: bool = False,
        transforms: Optional[A.Compose] = None,
        hflip_p: float = 0.0,
    ):
        self.df = df
        self.img_dir = img_dir
        self.train = train
        self.transforms = transforms
        self.index = df.droplevel(-1).index.unique()
        self.hflip_p = hflip_p

    def __len__(self):
        return len(self.index)

    def get_keypoints(self, chunk: pd.DataFrame):
        return chunk.values, chunk.index.values

    def keypoints2xy(self, keypoints, shape):
        C, H, W = shape
        denominator = torch.tensor([[W, H]])
        return torch.tensor(keypoints) / denominator

    def hflip(self, x: torch.Tensor, target: Dict[str, torch.Tensor], plane: str):
        if random.random() < self.hflip_p and plane == "axial":
            x = torchvision.transforms.functional.hflip(x)
            prev_left = target["axial_left"]
            prev_right = target["axial_right"]
            prev_left[0] = 1 - prev_left[0]
            prev_right[0] = 1 - prev_right[0]
            target["axial_left"], target["axial_right"] = prev_right, prev_left
        return x, target

    def __getitem__(self, index):
        idx = self.index[index]
        study_id, series_id, instance_number, plane = idx
        chunk: pd.DataFrame = self.df.loc[idx]
        img_path = utils.get_image_path(study_id, series_id, instance_number, self.img_dir, suffix=".png")
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        keypoints, keys = self.get_keypoints(chunk)

        transformed = self.transforms(image=img, keypoints=keypoints)
        x = transformed["image"]
        keypoints = np.array(transformed["keypoints"])

        keypoints = self.keypoints2xy(keypoints, x.shape)
        target = dict(zip(keys, keypoints))

        x, target = self.hflip(x, target, plane)

        return x, target


class PredictDataset(torch.utils.data.Dataset):
    def __init__(
        self, df: pd.DataFrame, img_dir: Path = constants.TRAIN_IMG_DIR, transforms: Optional[A.Compose] = None
    ):
        self.df = df
        self.img_dir = img_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        chunk = self.df.loc[index]
        img_path = utils.get_image_path(
            chunk["study_id"], chunk["series_id"], chunk["instance_number"], self.img_dir, suffix=".png"
        )
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        transformed = self.transforms(image=img)
        x = transformed["image"]
        return x


def get_transforms(img_size):
    return A.Compose(
        [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
            A.Normalize((0.485,), (0.229,)),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def get_train_transforms(img_size):
    return A.Compose(
        [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
            A.Affine(
                rotate=(-15, 15),
                shear=(-15, 15),
                translate_percent=0.2,
                scale=(0.8, 1.2),
                interpolation=cv2.INTER_CUBIC,
                p=0.5,
            ),
            A.Perspective(0.2, interpolation=cv2.INTER_CUBIC, p=0.5),
            A.GaussNoise(var_limit=30, noise_scale_factor=0.90, mean=0, p=0.5),
            A.MotionBlur(blur_limit=(3, 7), p=0.5),
            A.Normalize((0.485,), (0.229,)),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def get_predict_transforms(img_size):
    return A.Compose(
        [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
            A.Normalize((0.485,), (0.229,)),
            ToTensorV2(),
        ]
    )


def load_df(coor_path: Path = constants.COOR_PATH, t2_coor_path: Path = constants.T2_COOR_PATH) -> pd.DataFrame:
    df = utils.load_coor(coor_path)
    df = df[constants.BASIC_COLS + ["x", "y", "type"]]
    df["type"] = df["type"].map(lambda x: {"right": "left", "left": "right"}.get(x, x))
    df["type_side"] = df.pop("type").apply(lambda x: f"axial_{x}" if x in ("right", "left") else f"sagittal_{x}_right")

    t2 = pd.read_csv(t2_coor_path)
    t2["type_side"] = "sagittal" + "_" + t2.pop("level").str.lower().str.replace("/", "_") + "_" + t2.pop("side")

    df = df.merge(t2, how="outer", on=constants.BASIC_COLS + ["type_side"], suffixes=["1", "2"])
    df["plane"] = df["type_side"].str.split("_", n=1, expand=True)[0]

    df["x"] = df["x2"].where(~df.pop("x2").isna(), df.pop("x1"))
    df["y"] = df["y2"].where(~df.pop("y2").isna(), df.pop("y1"))

    df = df.pivot_table(index=constants.BASIC_COLS + ["plane"], columns=["type_side"], values=["x", "y"]).fillna(-1000)
    df = df.stack(future_stack=True)

    df = df.sort_index()

    return df


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        coor_path: Path = constants.COOR_PATH,
        t2_coor_path: Path = constants.T2_COOR_PATH,
        img_dir: Path = constants.TRAIN_IMG_DIR,
        n_splits: int = 5,
        this_split: int = 0,
        img_size: int = 256,
        batch_size: int = 64,
        num_workers: int = 8,
    ):
        super().__init__()
        self.img_dir = Path(img_dir)
        self.n_splits = n_splits
        self.this_split = this_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_size = img_size
        self.coor_path = coor_path
        self.t2_coor_path = t2_coor_path

    def prepare_data(self):
        pass

    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        return utils.split(self.df, self.n_splits, self.this_split)

    def setup(self, stage: str):
        if stage == "fit":
            self.df = load_df(self.coor_path, self.t2_coor_path)
            train_df, val_df = self.split()
            self.train_ds = Dataset(
                train_df, self.img_dir, train=True, transforms=get_train_transforms(self.img_size), hflip_p=0.5
            )
            self.val_ds = Dataset(
                val_df, self.img_dir, train=False, transforms=get_transforms(self.img_size), hflip_p=0.0
            )

        if stage == "test":
            pass

        if stage == "predict":
            img_df = utils.get_images_df(self.img_dir)
            self.predict_ds = PredictDataset(img_df, self.img_dir, get_predict_transforms(self.img_size))

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

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            dataset=self.predict_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


if __name__ == "__main__":
    import yaml

    config = yaml.load(open("configs/keypoints.yaml"), Loader=yaml.FullLoader)
    dm = DataModule(**config["data"]["init_args"])
    dm.setup("fit")
    for x, target in dm.train_dataloader():
        print()
        utils.print_tensor(x)
        utils.print_tensor(target)

        torch.save(target, "target.pt")
        input()
