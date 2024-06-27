from multiprocessing import Pool
from pathlib import Path
from functools import partial

import tqdm
import tyro
import cv2

from src import utils
from src import constants



def f(ids, in_dir, out_dir, img_size):
    study_id, series_id, instance_number = ids
    in_path = utils.get_image_path(study_id, series_id, instance_number, in_dir)
    out_path = utils.get_image_path(study_id, series_id, instance_number, out_dir, suffix=".png")
    img = utils.load_dcm_img(in_path, add_channels=False, size=img_size)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)


def main(out_dir: Path, img_size: int, in_dir: Path = constants.TRAIN_IMG_DIR):
    df = utils.get_images_df()

    partial_f = partial(f, in_dir=in_dir, out_dir=out_dir, img_size=img_size)

    print("out_dir:",  out_dir)
    print("in_dir:", in_dir)
    print("img_size:", img_size)

    with Pool() as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(partial_f, df.values), total=len(df)):
            pass


if __name__ == "__main__":
    tyro.cli(main)
