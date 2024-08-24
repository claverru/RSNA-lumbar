from typing import Literal

import cv2
import numpy as np
import tyro
import yaml

from src import utils
from src.patch import data_loading


def main(mode: Literal["train", "val"] = "val"):
    config = yaml.load(open("configs/patch.yaml"), Loader=yaml.FullLoader)
    config["data"]["init_args"]["num_workers"] = 0
    dm = data_loading.DataModule(**config["data"]["init_args"])
    dm.setup("fit")
    dl = dm.train_dataloader() if mode == "train" else dm.val_dataloader()
    for X, y in dl:
        patches = []
        for x in X:
            utils.print_tensor(y)
            img = x[0].numpy()
            img = img - img.min()
            img = img / img.max()
            img = (img * 255).astype(np.uint8)
            patches.append(img)

        patches = np.array(patches)
        B, H, W = patches.shape
        patches = patches.reshape(16, -1, H, W)
        for p in patches:
            draw = cv2.hconcat(p)
            cv2.imwrite("patches.png", draw)
            input("Press a key for another patch")


if __name__ == "__main__":
    tyro.cli(main)
