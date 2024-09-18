from pathlib import Path
from typing import List, Tuple

import albumentations as A
import lightning as L
import numpy as np
import pandas as pd
import torch

from src import constants, utils
from src.patch.data_loading import get_transforms, load_keypoints
from src.patch.utils import PLANE2SPACING, Image, Keypoint, Spacing, angle_crop_size


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        train: pd.DataFrame,
        df: pd.DataFrame,
        meta: pd.DataFrame,
        img_size: int,
        transforms: A.Compose,
    ):
        self.train = train
        self.df = df
        self.meta = meta
        self.index = train.index.unique()
        self.transforms = transforms
        self.img_size = img_size
        self.study_level_side_index = set(meta.index)

    def __len__(self):
        return len(self.index)

    def get_target(self, idx):
        if self.train.shape[1] == 0:
            return None
        return self.train.loc[idx].apply(torch.tensor).to_dict()

    def get_patches(self, study_id, level, side, imgs) -> torch.Tensor:
        if (study_id, level, side) not in self.study_level_side_index:
            img = np.zeros((128, 128), dtype=np.uint8)
            patches = self.transforms(image=img)["image"][None, ...]
            return patches

        chunk = self.df.loc[(study_id, level, side), ["img_path", "series_description", "x", "y", "angle"]]
        patches = []
        for _, (img_path, plane, x, y, angle) in chunk.iterrows():
            img = imgs[img_path]
            kp = Keypoint(x, y)
            patch = angle_crop_size(img, kp, angle, self.img_size, plane)
            if 0 in patch.shape:  # I don't know
                patch = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
            patch = self.transforms(image=patch)["image"]
            patches.append(patch)

        patches = torch.stack(patches, 0)
        return patches

    def get_meta(self, study_id, level, side):
        if (study_id, level, side) not in self.study_level_side_index:
            return torch.zeros((1, self.meta.shape[1]), dtype=torch.float)

        meta = self.meta.loc[(study_id, level, side)].values
        meta = torch.tensor(meta, dtype=torch.float)
        return meta

    def get_images(self, idx):
        chunk = self.df.loc[idx][
            ["series_description", "img_path", "PixelSpacing_0", "PixelSpacing_1"]
        ].drop_duplicates()
        imgs = {}
        for _, (plane, img_path, spacing_x, spacing_y) in chunk.iterrows():
            spacing = Spacing(spacing_x, spacing_y)
            img = Image.from_path(img_path, spacing=spacing)
            target_spacing = Spacing(PLANE2SPACING[plane], PLANE2SPACING[plane])
            img = img.resize_spacing(target_spacing)
            imgs[img_path] = img
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


def get_level_ids(x):
    target = np.linspace(0, 4, 5)
    x_4 = target - x * 4
    mask = (-0.65 <= x_4) & (x_4 <= 0.55)
    return np.where(mask)[0]


def load_levels(levels_path: Path = constants.LEVELS_PATH):
    df = pd.read_parquet(levels_path)
    df["level_id"] = df["pred_f0"].apply(get_level_ids)
    df = df.explode("level_id")
    df["level_distance"] = df["level_id"] - df.pop("pred_f0") * 4
    df["level_distance"] = df["level_distance"].astype(float)
    df["level"] = df.pop("level_id").map(lambda x: constants.LEVELS[x])
    df = df.sort_values(constants.BASIC_COLS)
    return df


META_COLS = ["level_distance", "xx", "yy", "zz", "nx", "ny", "nz"]


def mirror_world(meta, suf=""):
    meta["xx" + suf] = meta.groupby(level=[0, 1, 2])["xx" + suf].transform(lambda x: x if x.mean() > 0 else -x) / 10
    meta["yy" + suf] = meta["yy" + suf] / 10
    meta["zz" + suf] = meta.groupby(level=[0, 1, 2])["zz" + suf].transform(lambda x: x - x.min()) / 10
    return meta


def add_lumbar_distances(df: pd.DataFrame) -> pd.DataFrame:
    def distance(a, b):
        a = np.stack(a.values)
        b = np.stack(b.values)
        return np.linalg.norm(a - b, axis=1)

    mean_world = df.groupby(["study_id", "level"], as_index=False)[["xx", "yy", "zz"]].mean()
    mean_world["xxyyzz"] = mean_world[["xx", "yy", "zz"]].apply(np.array, axis=1)
    mean_world = mean_world.pivot(index="study_id", columns="level", values="xxyyzz").reset_index()
    pair_levels = zip(constants.LEVELS[:-1], constants.LEVELS[1:])
    mean_world["lumbar_distance"] = distance(mean_world["l1_l2"], mean_world["l5_s1"]) / 10
    mean_world["lumbar_path_distance"] = sum(distance(mean_world[a], mean_world[b]) for a, b in pair_levels) / 10
    distances = mean_world[["study_id", "lumbar_distance", "lumbar_path_distance"]]
    df = df.merge(distances, how="inner", on="study_id")
    return df


def load_df(
    keypoints_path: Path = constants.KEYPOINTS_PATH,
    levels_path: Path = constants.LEVELS_PATH,
    desc_path: Path = constants.DESC_PATH,
    meta_path: Path = constants.META_PATH,
    img_dir: Path = constants.TRAIN_IMG_DIR,
):
    keypoints = load_keypoints(keypoints_path, desc_path, meta_path)
    levels = load_levels(levels_path)
    meta = utils.load_meta(meta_path, with_center=False, with_normal=True, with_relative_position=True)
    meta["img_path"] = meta[constants.BASIC_COLS].apply(
        lambda x: str(utils.get_image_path(*x, img_dir, suffix=".png")), axis=1
    )

    # merge levels and filter
    df = pd.merge(keypoints, levels, how="left", on=constants.BASIC_COLS)
    df["level"] = df["type"].where(df["level"].isna(), df.pop("level"))
    df["level_distance"] = df["level_distance"].fillna(0.0)

    # merge meta
    df = df.merge(meta, how="inner", on=constants.BASIC_COLS)
    df = utils.add_xyz_world(df)
    df = utils.add_sides(df)

    common_index = ["study_id", "level", "side"]

    # sort, clean and index
    df = df.set_index(common_index).sort_index()
    df = mirror_world(df)

    x = df[["series_description", "img_path", "PixelSpacing_0", "PixelSpacing_1", "x", "y", "angle"]]
    new_meta = df[META_COLS]

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
            self.train_ds = Dataset(train_df, self.df, self.meta, self.img_size, get_transforms(self.img_size))
            self.val_ds = Dataset(val_df, self.df, self.meta, self.img_size, get_transforms(self.img_size))

        if stage == "test":
            _, val_df = self.split()
            self.test_ds = Dataset(val_df, self.df, self.meta, self.img_size, get_transforms(self.img_size))

        if stage == "predict":
            fake_train = self.df[[]].droplevel([1, 2])
            transforms = get_transforms(self.img_size)
            self.predict_ds = Dataset(fake_train, self.df, self.meta, self.img_size, transforms)

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

    config = yaml.load(open("configs/sequence.yaml"), Loader=yaml.FullLoader)
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
