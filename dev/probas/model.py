from typing import Dict

import torch

from src import constants, losses, model


def get_proj(in_dim, out_dim, dropout=0):
    return torch.nn.Sequential(
        torch.nn.Dropout(dropout) if dropout else torch.nn.Identity(),
        torch.nn.Linear(in_dim, out_dim) if in_dim is not None else torch.nn.LazyLinear(out_dim),
    )


class LightningModule(model.LightningModule):
    def __init__(self, linear_dropout, **kwargs):
        super().__init__(**kwargs)
        self.train_loss = losses.LumbarLoss()
        self.val_loss = losses.LumbarLoss()
        self.drop = torch.nn.Dropout1d(1 / 500)
        self.transformer = model.get_transformer(3, 3, 4, 0.1)
        self.heads = torch.nn.ModuleDict({k: get_proj(75, 3, linear_dropout) for k in constants.CONDITION_LEVEL})

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B = x.shape[0]
        x = x.reshape(-1, 25, 3)
        x = self.drop(x)
        x = self.transformer(x)
        x = x.reshape(B, -1)
        outs = {k: head(x) for k, head in self.heads.items()}
        return outs

    def augment(self, x, y):
        B = x.shape[0]
        x = x.reshape(-1, 25, 3)
        for i, k in enumerate(y):
            idx = torch.arange(B)
            if torch.rand(()) < 0.1:
                i1, i2 = torch.randint(B, ()), torch.randint(B, ())
                idx[i1], idx[i2] = idx[i2], idx[i1]
            x[:, i] = x[:, i][idx]
            y[k] = y[k][idx]
        x = x.reshape(B, -1)
        return x, y

    def training_step(self, batch, batch_idx):
        x, y = batch

        x, y = self.augment(x, y)

        pred = self.forward(x)
        batch_size = y[list(y)[0]].shape[0]

        losses: Dict[str, torch.Tensor] = self.train_loss.jit_loss(y, pred)

        for k, v in losses.items():
            self.log(f"train_{k}_loss", v, on_epoch=True, prog_bar=False, on_step=False, batch_size=batch_size)

        loss = sum(losses.values()) / len(losses)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, on_step=True, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self.forward(x)
        self.val_loss.update(y, pred)

    def on_validation_epoch_end(self, *args, **kwargs):
        losses = self.val_loss.compute()
        self.val_loss.reset()
        for k, v in losses.items():
            self.log(f"val_{k}_loss", v, on_epoch=True, prog_bar=False, on_step=False)

        loss = sum(losses.values()) / len(losses)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, on_step=False)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, meta, mask = batch
        pred = self.forward(x, meta, mask)
        return pred
