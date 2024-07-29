from typing import List, Optional, Tuple

from pathlib import Path
import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
import lightning as L

from src import constants, utils
from src.sequence.data_loading import load_this_train


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        train: pd.DataFrame,
        df: pd.DataFrame,
        img_dir: Path,
        img_size: int,
        size_ratio: int,
        transforms: A.Compose
    ):
        self.df = df
        self.img_dir = img_dir
        self.train_index = train.index
        self.train = train
        self.img_size = img_size
        self.transforms = transforms
        self.df_index = set(list(df.index))
        self.kp_cols = [c for c in df.columns if c.startswith("f")]
        self.size_ratio = size_ratio

    def __len__(self):
        return len(self.train_index)

    def get_target(self, idx):
        return self.train.loc[idx].apply(torch.tensor).to_dict()

    def get_patch(self, img, keypoint):
        h, w = img.shape
        half_side = max(h, w) // self.size_ratio
        x, y = keypoint
        x, y = (int(x * img.shape[1]), int(y * img.shape[0]))
        xmin, ymin, xmax, ymax = x - half_side, y - half_side, x + half_side, y + half_side
        xmin, ymin, xmax, ymax = max(xmin, 0), max(ymin, 0), min(xmax, w) , min(ymax, h)
        patch = img[ymin:ymax, xmin:xmax]
        patch = self.transforms(image=patch)["image"]
        return patch


    def get_patches(self, study_id, level) -> torch.Tensor:
        chunk = self.df.loc[(study_id, level)]
        patches = []
        for (series_id, instance_number), gdf in chunk.groupby(["series_id", "instance_number"]):
            img_path = utils.get_image_path(study_id, series_id, instance_number, self.img_dir, suffix=".png")
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            for keypoint in gdf[["x", "y"]].values:
                patch = self.get_patch(img, keypoint)
                patches.append(patch)

        patches = torch.stack(patches, 0)

        return patches

    def __getitem__(self, index):
        study_id, level = self.train_index[index]
        target = self.get_target((study_id, level))
        X = self.get_patches(study_id, level)
        mask = torch.tensor([False] * len(X))
        return X, mask, target


def get_transforms(img_size):
    return A.Compose(
        [
            A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
            A.Normalize((0.485, ), (0.229, )),
            ToTensorV2(),
        ]
    )


def get_aug_transforms(img_size):
    return A.Compose(
        [
            A.ShiftScaleRotate(
                shift_limit=(-0.1, 0.1),
                rotate_limit=(-10, 10),
                interpolation=cv2.INTER_CUBIC,
                scale_limit=0.1,
                # border_mode=cv2.BORDER_CONSTANT,
                # value=0,
                p=0.5
            ),
            A.Resize(img_size, img_size, interpolation=cv2.INTER_CUBIC),
            # A.HorizontalFlip(p=0.5),
            # A.VerticalFlip(p=0.5),
            A.MotionBlur(p=0.1),
            A.GaussNoise(p=0.1),
            A.Normalize((0.485, ), (0.229, )),
            ToTensorV2()
        ]
    )


def collate_fn(data):
    X, mask, labels = zip(*data)
    X = utils.pad_sequences(X, padding_value=0)
    mask = utils.pad_sequences(mask, padding_value=True)
    labels = utils.cat_dict_tensor(labels, torch.stack)
    return X, mask, labels


def load_keypoints(keypoints_path: Path = constants.KEYPOINTS_PATH) -> pd.DataFrame:
    df = pd.read_parquet(keypoints_path)
    levels_sides = constants.LEVELS + ["left", "right"]
    levels_sides_cols = [f"{side}_{x}" for side in levels_sides for x in ("x", "y")]
    df = df.set_index(constants.BASIC_COLS)
    df.columns = levels_sides_cols
    df = df.melt(ignore_index=False)
    df[["level", "coor"]] = df.pop("variable").str.rsplit("_", n=1, expand=True)
    df = df.set_index("level", append=True)
    df = df.pivot_table(values="value", index=df.index.names, columns="coor").reset_index()
    return df


def load_levels(levels_path: Path = constants.LEVELS_PATH):
    df = pd.read_parquet(levels_path)
    basic_cols = ["study_id", "series_id", "instance_number"]
    probas_cols = [c for c in df.columns if c.startswith("probas")]
    df = df[basic_cols + probas_cols]
    df["level_id"] = df[probas_cols].values.argmax(1)
    df["level"] = df["level_id"].map(lambda x: constants.LEVELS[x])
    df = df.sort_values(basic_cols)
    df = df.drop(columns=probas_cols + ["level_id"])
    return df


def load_df(
    keypoints_path: Path = constants.KEYPOINTS_PATH,
    levels_path: Path = constants.LEVELS_PATH,
    desc_path: Path = constants.DESC_PATH
):
    keypoints = load_keypoints(keypoints_path)
    desc = utils.load_desc(desc_path)
    levels = load_levels(levels_path)

    # merge desc and filter
    df = pd.merge(keypoints, desc, how="inner", on=constants.BASIC_COLS[:2])
    is_axial = df["series_description"].eq("Axial T2")
    is_sagittal = ~is_axial
    is_levels = df["level"].isin(constants.LEVELS)
    is_sides = ~is_levels
    df = df[(is_axial & is_sides) | (is_sagittal & is_levels)]

    # merge levels and filter
    df = pd.merge(df, levels, how="left", on=constants.BASIC_COLS)
    df["level"] = df.pop("level_x").where(df["level_y"].isna(), df.pop("level_y"))

    # add path
    df = df.sort_values(["study_id", "series_id", "instance_number"])
    df = df.drop(columns=["series_description"]).set_index(["study_id", "level"]).sort_index()
    return df


class DataModule(L.LightningDataModule):
    def __init__(
            self,
            keypoints_path: Path = constants.KEYPOINTS_PATH,
            desc_path: Path = constants.DESC_PATH,
            train_path: Path = constants.TRAIN_PATH,
            levels_path: Path = constants.LEVELS_PATH,
            img_size: int = 224,
            size_ratio: int = 10,
            img_dir: Path = constants.TRAIN_IMG_DIR,
            n_splits: int = 5,
            this_split: int = 0,
            batch_size: int = 64,
            num_workers: int = 8,
        ):
        super().__init__()
        self.img_size = img_size
        self.size_ratio = size_ratio
        self.n_splits = n_splits
        self.this_split = this_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.img_dir = Path(img_dir)
        self.df = load_df(Path(keypoints_path), Path(levels_path), Path(desc_path))
        self.train = load_this_train(train_path)
        self.train_path = train_path

    def split(self) -> Tuple[List[int], List[int]]:
        return utils.split(self.train, self.n_splits, self.this_split, self.train_path)

    def setup(self, stage: str):
        if stage == "fit":
            train_df, val_df = self.split()
            self.train_ds = Dataset(
                train_df, self.df, self.img_dir, self.img_size, self.size_ratio, get_aug_transforms(self.img_size)
            )
            self.val_ds = Dataset(
                val_df, self.df, self.img_dir, self.img_size, self.size_ratio, get_transforms(self.img_size)
            )

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
            collate_fn=collate_fn,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=lambda wid: np.random.seed(np.random.get_state()[1][0] + wid),
            collate_fn=collate_fn,
        )


if __name__ == "__main__":
    import yaml

    config = yaml.load(open("configs/patchseq.yaml"), Loader=yaml.FullLoader)
    dm = DataModule(**config["data"]["init_args"])
    dm.setup("fit")
    for X, mask, labels in dm.train_dataloader():
        print()
        print(X.shape, X.dtype)
        print(mask.shape, mask.dtype)
        print({k: (v.shape, v.dtype) for k, v in labels.items()})
        print()
        input()
