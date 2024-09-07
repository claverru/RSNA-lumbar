import pandas as pd
import tyro

from src import constants, utils


def load_df(path):
    df = pd.read_csv(path, index_col=0)
    df = df[df.pop("condition").str.contains("Spinal")]
    df["instance_number"] = df["instance_number"].replace(-1, pd.NA).astype("Int64").bfill()
    df["side"] = df["side"].map({"L": "left", "R": "right"})
    return df


def main(in_path: str = "data/coords_rsna_improved.csv", out_path: str = "data/t2_coordinates.csv"):
    df = load_df(in_path)

    meta = utils.load_meta(with_center=False, with_normal=False, with_relative_position=False)[
        constants.BASIC_COLS + ["Rows", "Columns"]
    ]

    df = df.merge(meta, how="inner", on=constants.BASIC_COLS)

    df["x"] = df.pop("relative_x") * df.pop("Columns")
    df["y"] = df.pop("relative_y") * df.pop("Rows")

    df.to_csv(out_path, index=False)


if __name__ == "__main__":
    tyro.cli(main)
