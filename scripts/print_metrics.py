from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tyro


def main(metrics_path: Path = Path("lightning_logs/version_0/metrics.csv")):
    fig, axes = plt.subplots(nrows=2)
    df = pd.read_csv(metrics_path)
    train = df.train_loss.dropna().reset_index(drop=True)
    val = df.val_loss.dropna().reset_index(drop=True)
    lr = df["lr-Adam"].dropna().reset_index(drop=True)
    axes[0].plot(train)
    axes[0].plot(val)
    axes[0].hlines(val.min(), 0, len(val) -1, color="orange", linestyle=":")
    axes[1].plot(lr, color="red", linestyle="--")
    plt.savefig(f"{metrics_path.parent.stem}.png")


if __name__ == "__main__":
    tyro.cli(main)
