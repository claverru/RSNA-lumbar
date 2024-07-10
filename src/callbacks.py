from pathlib import Path

from lightning.pytorch.loggers import CSVLogger
import matplotlib.pyplot as plt

import pandas as pd


class CustomLogger(CSVLogger):
    def __init__(self, train_c: str = "train_loss", val_c: str = "val_loss", lr_c: str = "lr-Adam", **kwargs):
        super().__init__(**kwargs)
        self.train_c = train_c
        self.val_c = val_c
        self.lr_c = lr_c

    def save_plot(self, metrics_path: Path):
        if not metrics_path.exists():
            return

        df = pd.read_csv(metrics_path)
        if not (self.val_c in df.columns and self.train_c in df.columns and self.lr_c in df.columns):
            return

        train = df[self.train_c].dropna().reset_index(drop=True)
        val = df[self.val_c].dropna().reset_index(drop=True)
        lr = df[self.lr_c].dropna().reset_index(drop=True)
        fig, axes = plt.subplots(nrows=2)
        axes[0].plot(train)
        axes[0].plot(val)
        axes[0].hlines(val.min(), 0, len(val) -1, color="orange", linestyle=":")
        axes[1].plot(lr, color="red", linestyle="--")
        plt.savefig(metrics_path.with_suffix(".png"))
        plt.clf()
        plt.close(fig)

    def save(self):
        super().save()
        metrics_path = Path(self.experiment.metrics_file_path)
        self.save_plot(metrics_path)
