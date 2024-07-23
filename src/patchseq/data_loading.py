from typing import List, Optional, Tuple

from pathlib import Path
import albumentations as A
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import torch
from albumentations.pytorch import ToTensorV2
import lightning as L

from src import constants, utils


class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        study_ids: List[int],
        df: pd.DataFrame,
        train: pd.DataFrame,
        img_size: int,
        transforms: A.Compose
    ):
        self.study_ids = study_ids
        self.df = df
        self.train = train
        self.img_size = img_size
        self.transforms = transforms
        self.df_index = set(list(df.index))
        self.kp_cols = [c for c in df.columns if c.startswith("f")]

    def __len__(self):
        return len(self.study_ids)

    def get_labels(self, study_id):
        d: dict = self.train.loc[study_id].set_index("condition_level")["severity"].to_dict()
        return {k: torch.tensor(constants.SEVERITY2LABEL.get(d.get(k), -1)) for k in constants.CONDITION_LEVEL}

    def get_chunk(self, study_id, description) -> pd.DataFrame:
        index = (study_id, description)
        if index in self.df_index:
            return self.df.loc[index]

    def get_crops(self, chunk: Optional[pd.DataFrame], description: str) -> torch.Tensor:
        if chunk is None:
            n = 2 if description == "Axial T2" else 5
            return torch.zeros((1, n, 1, self.img_size, self.img_size), dtype=torch.float32)

        study_crops = []
        for _, row in chunk.iterrows():
            img_path = str(row["image_path"])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            h, w = img.shape
            half_side = max(h, w) // 10
            keypoints = row[self.kp_cols].to_numpy().reshape(-1, 2)
            if description == "Axial T2":
                keypoints = keypoints[-2:]
            else:
                keypoints = keypoints[:-2]
            img_crops = []
            for (x, y) in keypoints:
                x, y = (int(x * img.shape[1]), int(y * img.shape[0]))
                xmin, ymin, xmax, ymax = x - half_side, y - half_side, x + half_side, y + half_side
                xmin, ymin, xmax, ymax = max(xmin, 0), max(ymin, 0), min(xmax, w) , min(ymax, h)
                crop = img[ymin:ymax, xmin:xmax]
                crop = self.transforms(image=crop)["image"]
                img_crops.append(crop)
            img_crops = torch.stack(img_crops, 0)
            study_crops.append(img_crops)
        study_crops = torch.stack(study_crops, 0)

        return study_crops

    def __getitem__(self, index):
        study_id = self.study_ids[index]
        labels = self.get_labels(study_id)
        X = {}
        masks = {}
        z = {}
        for description in constants.DESCRIPTIONS:
            chunk = self.get_chunk(study_id, description)
            x = self.get_crops(chunk, description)
            X[description] = x
            masks[description] = torch.tensor([False] * len(x))
            z[description] = torch.linspace(0, 1, len(x))
        return X, masks, z, labels


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
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.MotionBlur(p=0.1),
            A.GaussNoise(p=0.1),
            A.Normalize((0.485, ), (0.229, )),
            ToTensorV2()
        ]
    )


def collate_fn(data):
    X, masks, z, labels = zip(*data)
    X = utils.cat_dict_tensor(X, lambda x: utils.pad_sequences(x, padding_value=0))
    z = utils.cat_dict_tensor(z, lambda x: utils.pad_sequences(x, padding_value=0))
    masks = utils.cat_dict_tensor(masks, lambda x: utils.pad_sequences(x, padding_value=True))
    labels = utils.cat_dict_tensor(labels, torch.stack)
    return X, masks, z, labels


def load_df(
    keypoints_path = Path("data/preds_keypoints.parquet"),
    desc_path = constants.DESC_PATH,
    img_dir = Path("data/train_images_clip")
):
    keypoints = pd.read_parquet(keypoints_path)
    desc = utils.load_desc(desc_path)
    df = pd.merge(desc, keypoints, how="inner", on=["study_id", "series_id"])
    df = df.sort_values(["study_id", "series_id", "instance_number"])
    df["image_path"] = df.apply(
        lambda x: utils.get_image_path(x["study_id"], x["series_id"], x["instance_number"], img_dir, suffix=".png"),
        axis=1
    )
    df = df.drop(columns=["instance_number", "series_id"]).set_index(["study_id", "series_description"]).sort_index()
    return df


class DataModule(L.LightningDataModule):
    def __init__(
            self,
            keypoints_path: Path = Path("data/preds_keypoints.parquet"),
            desc_path: Path = constants.DESC_PATH,
            train_path: Path = constants.TRAIN_PATH,
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
        self.df = load_df(Path(keypoints_path), Path(desc_path), self.img_dir)
        self.train = utils.load_train(train_path).set_index("study_id").sort_index()

    def split(self) -> Tuple[List[int], List[int]]:
        strats = self.train["condition_level"] + "_" + self.train["severity"]
        groups = self.train.reset_index()["study_id"]
        skf = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True)
        for i, (train_ids, val_ids) in enumerate(skf.split(strats, strats, groups)):
            if i == self.this_split:
                break
        return list(set(groups[train_ids])), list(set(groups[val_ids]))

    def setup(self, stage: str):
        if stage == "fit":
            train_study_ids, val_study_ids = self.split()
            self.train_ds = Dataset(train_study_ids, self.df, self.train, self.img_size, get_aug_transforms(self.img_size))
            self.val_ds = Dataset(val_study_ids, self.df, self.train, self.img_size, get_transforms(self.img_size))

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

    config = yaml.load(open("configs/classplusseq.yaml"), Loader=yaml.FullLoader)
    dm = DataModule(**config["data"]["init_args"])
    dm.setup("fit")
    for X, masks, z, y in dm.train_dataloader():
        print()
        print({k: (v.shape, v.dtype) for k, v in X.items()})
        print({k: (v.shape, v.dtype) for k, v in masks.items()})
        print({k: (v.shape, v.dtype) for k, v in z.items()})
        print({k: (v.shape, v.dtype) for k, v in y.items()})
        print()
        input()
