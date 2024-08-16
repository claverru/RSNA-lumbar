from pathlib import Path
from typing import List, Tuple

import albumentations as A
import cv2
import lightning as L
import numpy as np
import pandas as pd
import torch

from src import constants, utils
from src.patch.data_loading import get_aug_transforms, get_patch, get_transforms, load_keypoints, resize_spacing


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        train: pd.DataFrame,
        df: pd.DataFrame,
        meta: pd.DataFrame,
        size_ratio: int,
        transforms: A.Compose,
    ):
        self.train = train
        self.df = df
        self.meta = meta
        self.index = train.index.unique()
        self.transforms = transforms
        self.size_ratio = size_ratio
        self.study_level_side_index = set(meta.index)

    def __len__(self):
        return len(self.index)

    def get_target(self, idx):
        if self.train.shape[1] == 0:
            return None
        return self.train.loc[idx].apply(torch.tensor).to_dict()

    def get_patches(self, study_id, level, side, imgs) -> torch.Tensor:
        if (study_id, level, side) not in self.study_level_side_index:
            img = np.zeros((200, 200), dtype=np.uint8)
            patch = self.transforms(image=img)["image"][None, ...]
            return patch

        chunk = self.df.loc[(study_id, level, side)]
        patches = []
        for img_path, gdf in chunk.groupby(level=0)[["x", "y"]]:
            img = imgs[img_path]
            patches += [get_patch(img, x, y, self.size_ratio) for x, y in gdf.values]

        patches = [self.transforms(image=patch)["image"] for patch in patches]
        patches = torch.stack(patches, 0)
        return patches

    def get_meta(self, study_id, level, side):
        if (study_id, level, side) not in self.study_level_side_index:
            return torch.zeros((1, self.meta.shape[1]), dtype=torch.float)

        meta = self.meta.loc[(study_id, level, side)].values
        meta = torch.tensor(meta, dtype=torch.float)
        return meta

    def get_images(self, idx):
        chunk = self.df.loc[idx].reset_index()[
            ["img_path", "PixelSpacing_0", "PixelSpacing_1", "TargetSpacing_0", "TargetSpacing_1"]
        ]
        imgs = {}
        for img_path, spacing_x, spacing_y, target_spacing_x, target_spacing_y in set(chunk.itertuples(index=False)):
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            imgs[img_path] = resize_spacing(img, spacing_x, spacing_y, target_spacing_x, target_spacing_y)
        return imgs

    def __getitem__(self, index):
        idx = self.index[index]
        target = self.get_target(idx)
        imgs = self.get_images(idx)

        if isinstance(idx, tuple):
            study_id, level, side = idx
            X = {"any": {"any": self.get_patches(study_id, level, side, imgs)}}
            meta = {"any": {"any": self.get_meta(study_id, level, side)}}
            mask = {"any": {"any": torch.tensor([False] * len(X["any"]["any"]))}}
        else:
            study_id = idx
            X = {}
            meta = {}
            mask = {}
            for level in constants.LEVELS:
                X_level = {}
                meta_level = {}
                mask_level = {}
                for side in constants.SIDES:
                    X_level[side] = self.get_patches(study_id, level, side, imgs)
                    meta_level[side] = self.get_meta(study_id, level, side)
                    mask_level[side] = torch.tensor([False] * len(X_level[side]))
                X[level] = X_level
                meta[level] = meta_level
                mask[level] = mask_level

        if target is None:
            return X, meta, mask

        return X, meta, mask, target


def collate_fn(data):
    data = zip(*data)  # X, meta, mask, (target)
    functions = [
        lambda x: utils.pad_sequences(x, padding_value=0),
        lambda x: utils.pad_sequences(x, padding_value=0),
        lambda x: utils.pad_sequences(x, padding_value=True),
        torch.stack,
    ]
    return [utils.cat_tensors(x, f) for x, f in zip(data, functions)]


def load_levels(levels_path: Path = constants.LEVELS_PATH):
    df = pd.read_parquet(levels_path)
    target = np.linspace(0, 4, 5)
    df["level_id"] = df["pred_f0"].apply(lambda x: np.where(np.abs(x * 4 - target) <= 0.6)[0])
    df = df.explode("level_id")
    df["level_distance"] = df.pop("pred_f0") - df["level_id"] / 4
    df["level_distance"] = df["level_distance"].astype(float)
    df["level"] = df.pop("level_id").map(lambda x: constants.LEVELS[x])
    df = df.sort_values(constants.BASIC_COLS)
    return df


META_COLS = [
    "level_distance",
    "x",
    "y",
    "xx",
    "yy",
    "zz",
    "xx_center",
    "yy_center",
    "zz_center",
    "pos_x",
    "pos_y",
    "pos_z",
    "nx",
    "ny",
    "nz",
    "center_x",
    "center_y",
    "SliceLocation",
    "SliceThickness",
    # "ImageOrientationPatient_0",
    # "ImageOrientationPatient_1",
    # "ImageOrientationPatient_2",
    # "ImageOrientationPatient_3",
    # "ImageOrientationPatient_4",
    # "ImageOrientationPatient_5",
    # "PixelSpacing_0",
    # "PixelSpacing_1",
    # "ImagePositionPatient_0",
    # "ImagePositionPatient_1",
    # "ImagePositionPatient_2",
    # "BitsStored",
    # "Columns",
    # "Rows",
]


def load_df(
    keypoints_path: Path = constants.KEYPOINTS_PATH,
    levels_path: Path = constants.LEVELS_PATH,
    desc_path: Path = constants.DESC_PATH,
    meta_path: Path = constants.META_PATH,
    img_dir: Path = constants.TRAIN_IMG_DIR,
):
    keypoints = load_keypoints(keypoints_path, desc_path, meta_path)
    levels = load_levels(levels_path)
    meta = utils.load_meta(meta_path).drop(columns=["PixelSpacing_0", "PixelSpacing_1"])

    # merge levels and filter
    df = pd.merge(keypoints, levels, how="left", on=constants.BASIC_COLS)
    df["level"] = df["type"].where(df["level"].isna(), df.pop("level"))
    df["level_distance"] = df["level_distance"].fillna(0.0)

    # add img path
    df["img_path"] = df[constants.BASIC_COLS].apply(
        lambda x: str(utils.get_image_path(*x, img_dir, suffix=".png")), axis=1
    )

    # merge meta
    df = df.merge(meta, how="inner", on=constants.BASIC_COLS)
    df = utils.add_xyz_world(df)
    df = utils.add_xyz_world(df, x_col="center_x", y_col="center_y", suffix="_center")
    df = utils.add_normal(df)
    df = utils.add_relative_position(df)
    df = utils.add_sides(df)

    # sort, clean and index
    df = df.sort_values(constants.BASIC_COLS)

    df = df.drop(columns=constants.BASIC_COLS[1:])
    df = df.set_index(["study_id", "level", "side", "img_path"]).sort_index()

    x = df[["x", "y", "PixelSpacing_0", "PixelSpacing_1", "TargetSpacing_0", "TargetSpacing_1"]]
    new_meta = df[META_COLS].droplevel(-1)

    print("Data loaded")

    return x, new_meta


class DataModule(L.LightningDataModule):
    def __init__(
        self,
        train_level_side: bool = True,
        keypoints_path: Path = constants.KEYPOINTS_PATH,
        desc_path: Path = constants.DESC_PATH,
        train_path: Path = constants.TRAIN_PATH,
        levels_path: Path = constants.LEVELS_PATH,
        meta_path: Path = constants.META_PATH,
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
        self.df, self.meta = load_df(
            Path(keypoints_path), Path(levels_path), Path(desc_path), Path(meta_path), self.img_dir
        )
        self.train = utils.load_train(train_path)
        self.train_path = train_path
        self.train_level_side = train_level_side

    def split(self) -> Tuple[List[int], List[int]]:
        return utils.split(self.train, self.n_splits, self.this_split)

    def setup(self, stage: str):
        if stage == "fit":
            train_df, val_df = self.split()
            if self.train_level_side:
                train_df = utils.train_study2levelside(train_df)
            self.train_ds = Dataset(train_df, self.df, self.meta, self.size_ratio, get_aug_transforms(self.img_size))
            self.val_ds = Dataset(val_df, self.df, self.meta, self.size_ratio, get_transforms(self.img_size))

        if stage == "test":
            _, val_df = self.split()
            self.test_ds = Dataset(val_df, self.df, self.meta, self.size_ratio, get_transforms(self.img_size))

        if stage == "predict":
            fake_train = self.df[[]].droplevel([1, 2, 3])
            self.predict_ds = Dataset(fake_train, self.df, self.meta, self.size_ratio, get_transforms(self.img_size))

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

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=lambda wid: np.random.seed(np.random.get_state()[1][0] + wid),
            collate_fn=collate_fn,
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            dataset=self.predict_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=lambda wid: np.random.seed(np.random.get_state()[1][0] + wid),
            collate_fn=collate_fn,
        )


if __name__ == "__main__":
    import yaml

    config = yaml.load(open("configs/sequence_1.yaml"), Loader=yaml.FullLoader)
    dm = DataModule(**config["data"]["init_args"])

    dm.setup("fit")

    X, meta, mask, labels = dm.train_ds[0]
    utils.print_tensor(X)
    utils.print_tensor(meta)
    utils.print_tensor(mask)
    utils.print_tensor(labels)

    for X, meta, mask, labels in dm.train_dataloader():
        print()
        utils.print_tensor(X)
        utils.print_tensor(meta)
        utils.print_tensor(mask)
        utils.print_tensor(labels)
        print()
        break

    for X, meta, mask, labels in dm.val_dataloader():
        print()
        utils.print_tensor(X)
        utils.print_tensor(meta)
        utils.print_tensor(mask)
        utils.print_tensor(labels)
        print()
        break
