from pathlib import Path
from typing import Optional, Tuple

import cv2
from sklearn.model_selection import StratifiedGroupKFold
import albumentations as A
import lightning as L
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2

from src import constants, utils
from src.image.data_loading import load_df
from src.segmentation import utils as segmentation_utils


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        img_dir: Path = constants.TRAIN_IMG_DIR,
        train: bool = False,
        transforms: Optional[A.Compose] = None
    ):
        self.df = df
        self.img_dir = img_dir
        self.train = train
        self.transforms = transforms
        self.index2id = {i: idx for i, idx in enumerate(self.df.index.unique())}

    def __len__(self):
        return len(self.index2id)

    def get_levels(self, chunk: pd.DataFrame):
        return chunk["condition_level"].str.split("_").apply(lambda x: "_".join(x[-2:])).values

    def get_keypoints(self, chunk: pd.DataFrame):
        return chunk[["x", "y"]].values

    def get_mask(self, img, levels, keypoints):
        _, h, w = img.shape
        mask = torch.zeros(5, h, w)
        sigma = max(h, w) / 40
        filter = segmentation_utils.gaussian_filter2d(sigma)
        fh, fw = filter.shape
        for level, (x, y) in zip(levels, keypoints):
            i = constants.LEVELS.index(level)
            x, y = int(x), int(y)
            half = filter.shape[0] // 2

            left = max(x-half, 0)
            right = min(x+half+1, w)
            up = max(y-half, 0)
            down = min(y+half+1, h)

            fleft = max(half-x, 0)
            fright = fw - max((x+half+1)-w, 0)
            fup = max(half-y, 0)
            fdown = fh - max((y+half+1)-h, 0)

            mask[i, up:down, left:right] = filter[fup:fdown, fleft:fright]
        background = (1 - mask.sum(0, keepdim=True)).clip(0)
        mask = torch.concat([background, mask], 0)
        mask = mask / mask.sum(0, keepdim=True)
        return mask

    def __getitem__(self, index):
        idx = self.index2id[index]
        chunk: pd.DataFrame = self.df.loc[idx]
        img_path = utils.get_image_path(idx[0], idx[1], idx[2], self.img_dir, suffix=".png")
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

        levels = self.get_levels(chunk)
        keypoints = self.get_keypoints(chunk)
        transformed = self.transforms(image=img, keypoints=keypoints, levels=levels)
        x = transformed["image"]
        levels = transformed["levels"]
        keypoints = transformed["keypoints"]
        mask = self.get_mask(x, levels, keypoints)
        return x, mask


def get_transforms(img_size):
    return A.Compose(
        [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
            A.Normalize((0.485, ), (0.229, )),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(
            format="xy",
            label_fields=["levels"], remove_invisible=True
        )
    )


def get_train_transforms(img_size):
    return A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=(-0.3, 0.3),
                rotate_limit=(-10, 10),
                interpolation=cv2.INTER_CUBIC,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=0.5
            ),
            A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
            A.RandomBrightnessContrast(p=0.1),
            A.MotionBlur(p=0.1),
            A.GaussNoise(p=0.1),
            A.Normalize((0.485, ), (0.229, )),
            ToTensorV2()
        ],
        keypoint_params=A.KeypointParams(
            format="xy",
            label_fields=["levels"], remove_invisible=True
        )
    )



def get_predict_transforms(img_size):
    return A.Compose(
        [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
            A.Normalize((0.485, ), (0.229, )),
            ToTensorV2(),
        ]
    )


class DataModule(L.LightningDataModule):
    def __init__(
            self,
            coor_path: Path = constants.COOR_PATH,
            train_path: Path = constants.TRAIN_PATH,
            desc_path: Path = constants.DESC_PATH,
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
        self.df = load_df(coor_path, train_path, desc_path)

    def prepare_data(self):
        pass

    def split(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        strats = self.df["condition_level"] + "_" + self.df["severity"]
        groups = self.df.reset_index()["study_id"]
        skf = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True)
        for i, (train_ids, val_ids) in enumerate(skf.split(strats, strats, groups)):
            if i == self.this_split:
                break
        return self.df.iloc[train_ids], self.df.iloc[val_ids]

    def setup(self, stage: str):
        if stage == "fit":
            train_df, val_df = self.split()
            self.train_ds = Dataset(train_df, self.img_dir, train=True, transforms=get_train_transforms(self.img_size))
            self.val_ds = Dataset(val_df, self.img_dir, train=False, transforms=get_transforms(self.img_size))

        if stage == "test":
            pass

        if stage == "predict":
            pass

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


if __name__ == "__main__":
    import yaml

    config = yaml.load(open("configs/image.yaml"), Loader=yaml.FullLoader)
    dm = DataModule(**config["data"]["init_args"])
    dm.setup("fit")
    for x, mask in dm.train_dataloader():
        print(x.shape, x.dtype)
        print(mask.shape, mask.dtype)
        break
