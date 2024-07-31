import torch
import timm
import torchmetrics

from src import constants, model


def get_proj(in_dim, out_dim, dropout):
    return torch.nn.Sequential(
        torch.nn.Dropout(dropout),
        torch.nn.Linear(in_dim, out_dim)
    )


class LightningModule(model.LightningModule):
    def __init__(self, arch: str = "resnet34", emb_dim: int = 512, dropout: float = 0.2, pretrained: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = 1
        self.loss_f = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.acc = torchmetrics.Accuracy(task="multiclass", num_classes=len(constants.LEVELS))
        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0, in_chans=self.in_channels).eval()
        n_feats = self.backbone.num_features
        self.emb = get_proj(n_feats, emb_dim, dropout)
        self.head = get_proj(emb_dim, len(constants.LEVELS), dropout)
        self.norm = torch.nn.InstanceNorm2d(self.in_channels)

    def forward(self, x):
        x_norm = self.norm(x)
        feats = self.backbone(x_norm)
        emb = self.emb(feats)
        out = self.head(emb)
        return out, emb

    def do_step(self, batch, prefix="train"):
        x, y = batch
        pred, _ = self.forward(x)
        loss = self.loss_f(pred, y)
        acc = self.acc(pred, y)
        self.log(f"{prefix}_loss", loss, on_epoch=True, prog_bar=True, on_step=False)
        self.log(f"{prefix}_acc", acc, on_epoch=True, prog_bar=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self.do_step(batch)

    def validation_step(self, batch, batch_idx):
        return self.do_step(batch, "val")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        X = batch
        pred, emb = self.forward(X)
        return {"probas": pred.softmax(1), "emb": emb}
