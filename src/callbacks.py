from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from lightning.pytorch.callbacks import BaseFinetuning, BasePredictionWriter
from lightning.pytorch.loggers import CSVLogger
from torch.nn.modules.batchnorm import _BatchNorm

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
        lr = df[self.lr_c].dropna().reset_index(drop=True).iloc[: len(val)]
        fig, axes = plt.subplots(nrows=2)
        axes[0].plot(train, label=self.train_c)
        axes[0].plot(val, label=self.val_c)
        axes[0].hlines(val.min(), 0, len(val) - 1, color="orange", linestyle=":")
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
    def __init__(self, out_dir=None, preds_name="preds.parquet"):
        super().__init__("epoch")
        self.out_dir = out_dir
        self.preds_name = preds_name

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        out_dir = self.out_dir if self.out_dir is not None else Path(trainer.logger.log_dir)
        out_path = out_dir / self.preds_name

        preds = utils.cat_dict_tensor(predictions)
        cols = []
        for k in preds:
            for i in range(preds[k].shape[1]):
                cols.append(f"{k}_f{i}")
        data = torch.concat(list(preds.values()), 1)

        df = trainer.predict_dataloaders.dataset.df

        try:
            preds_df = pd.DataFrame(data=data, columns=cols)
            out_df = pd.concat([df, preds_df], axis=1)
        except Exception as e:
            print(e)
            try:
                out_df = pd.DataFrame(index=df.index, data=data, columns=cols).reset_index()
            except Exception as e:
                print(e)
                print("Fallback to drop last index level")
                index = trainer.predict_dataloaders.dataset.index
                out_df = pd.DataFrame(index=index, data=data, columns=cols).reset_index()

        print(out_df.head())

        out_df.to_parquet(out_path)


class CustomBackboneFinetuning(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=10, unfreeze_bn=False):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch
        self._unfreeze_bn = unfreeze_bn

    def freeze_before_training(self, pl_module):
        self.freeze(pl_module.backbone, train_bn=False)

    @staticmethod
    def make_trainable(modules, unfreeze_backbone) -> None:
        modules = BaseFinetuning.flatten_modules(modules)
        for module in modules:
            if isinstance(module, _BatchNorm):
                if unfreeze_backbone:
                    module.track_running_stats = True
                    for param in module.parameters(recurse=False):
                        param.requires_grad = True
            else:
                for param in module.parameters(recurse=False):
                    param.requires_grad = True

    def finetune_function(self, pl_module, current_epoch, optimizer):
        if current_epoch == self._unfreeze_at_epoch:
            print(f"Unfreezing backbone... at epoch {current_epoch}")
            self.make_trainable(modules=pl_module.backbone, unfreeze_backbone=self._unfreeze_bn)
