from pathlib import Path

from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import BasePredictionWriter
import matplotlib.pyplot as plt
import pandas as pd
import torch

from src import utils


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
        lr = df[self.lr_c].dropna().reset_index(drop=True).iloc[:len(val)]
        fig, axes = plt.subplots(nrows=2)
        axes[0].plot(train, label=self.train_c)
        axes[0].plot(val, label=self.val_c)
        axes[0].hlines(val.min(), 0, len(val) -1, color="orange", linestyle=":")
        axes[0].legend(loc="upper right")
        axes[1].plot(lr, color="red", linestyle="--", label=self.lr_c)
        axes[1].legend(loc="upper right")
        plt.savefig(metrics_path.with_suffix(".png"))
        plt.clf()
        plt.close(fig)

    def save(self):
        super().save()
        metrics_path = Path(self.experiment.metrics_file_path)
        self.save_plot(metrics_path)


class Writer(BasePredictionWriter):
    def __init__(self):
        super().__init__("epoch")

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        out_dir = Path(trainer.logger.log_dir)
        out_path = out_dir / "preds.parquet"

        preds = utils.cat_dict_tensor(predictions)
        cols = []
        for k in preds:
            for i in range(preds[k].shape[1]):
                cols.append(f"{k}_f{i}")
        data = torch.concat(list(preds.values()), 1)

        # NOTE: reduces significantly memory size, maybe try without it in the future
        preds_df = pd.DataFrame(data, columns=cols)#.round(2)
        print(preds_df.head())
        print(preds_df.shape)

        df = trainer.predict_dataloaders.dataset.df

        out_df = pd.concat([df, preds_df], axis=1)

        out_df.to_parquet(out_path, index=False)
