import timm
import torch
import torchmetrics

from src import constants, model
from src.levels.data_loading import META_COLS


class LightningModule(model.LightningModule):
    def __init__(
        self,
        arch: str = "resnet34",
        dropout: float = 0.2,
        eval: bool = True,
        pretrained: bool = True,
        emb_dim: int = 512,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = 1
        self.loss_f = torch.nn.MSELoss()
        self.acc = torchmetrics.Accuracy(task="multiclass", num_classes=len(constants.LEVELS))
        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0, in_chans=self.in_channels)
        if eval:
            self.backbone = self.backbone.eval()

        self.meta_norm = torch.nn.LayerNorm(len(META_COLS))
        self.emb = model.get_proj(None, emb_dim, 0.0, activation=torch.nn.GELU())
        self.head = model.get_proj(emb_dim, 1, dropout, activation=torch.nn.Sigmoid())
        self.norm = torch.nn.InstanceNorm2d(self.in_channels)

    def forward(self, x, meta):
        x_norm = self.norm(x)
        feats = self.backbone(x_norm)
        meta = self.meta_norm(meta)
        feats = torch.concat([feats, meta], -1)
        emb = self.emb(feats)
        out = self.head(emb)
        return out

    def do_step(self, batch, prefix="train"):
        x, meta, y = batch
        pred = self.forward(x, meta)

        loss = self.loss_f(pred, y)
        acc = self.acc((pred * 4).round().int(), (y * 4).int())
        self.log(f"{prefix}_loss", loss, on_epoch=True, prog_bar=True, on_step=False)
        self.log(f"{prefix}_acc", acc, on_epoch=True, prog_bar=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self.do_step(batch)

    def validation_step(self, batch, batch_idx):
        return self.do_step(batch, "val")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        X = batch
        pred = self.forward(*X)
        return {"pred": pred}
