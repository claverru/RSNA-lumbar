import torch
import timm
import torchmetrics

from src import model
from src.patch import constants as patch_constants


def get_head(in_features, out_features, dropout):
    return torch.nn.Sequential(
        # torch.nn.BatchNorm1d(in_features),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(in_features, out_features)
    )


def get_pool():
    return torch.nn.Sequential(
        torch.nn.Flatten(start_dim=2),
        torch.nn.LazyLinear(1),
        torch.nn.Flatten()
    )


class LightningModule(model.LightningModule):
    def __init__(
        self,
        arch: str = "resnet34",
        emb_dim: int = 512,
        dropout: float = 0.0,
        eval: bool = True,
        pretrained: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.in_channels = 1
        self.num_classes = 4

        if self.num_classes == 3:
            weight = torch.tensor([1., 2, 4])
        else:
            weight = torch.tensor([1., 2, 4, 0.1])

        self.loss = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=-1)
        self.acc = torchmetrics.Accuracy(task="multiclass", num_classes=self.num_classes)
        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0, in_chans=self.in_channels)
        if eval:
            self.backbone = self.backbone.eval()
        n_feats = self.backbone.num_features
        self.norm = torch.nn.InstanceNorm2d(self.in_channels)
        self.pool = get_pool()
        self.emb = get_head(n_feats, emb_dim, dropout)
        self.heads = torch.nn.ModuleDict({k: get_head(emb_dim, self.num_classes, dropout) for k in patch_constants.CONDITIONS})

    def forward(self, x):
        norm_x = self.norm(x)
        feats = self.backbone.forward_features(norm_x)
        pool = self.pool(feats)
        emb = self.emb(pool)
        outs = {k: head(emb) for k, head in self.heads.items()}
        return outs

    def do_step(self, batch, prefix="train"):
        x, y_dict = batch
        pred_dict = self.forward(x)

        losses = {k: self.loss(pred_dict[k], y) for k, y in y_dict.items()}
        for k, loss in losses.items():
            self.log(f"{prefix}_{k}_loss", loss, on_epoch=True, prog_bar=False, on_step=False)

        accs = {}
        for k in y_dict:
            y =  y_dict[k]
            pred =  pred_dict[k]
            mask = y != -1
            y = y[mask]
            pred = pred[mask]
            acc = self.acc(pred, y)
            accs[k] = acc

        total_acc = sum(accs.values()) / len(accs)
        self.log(f"{prefix}_acc", total_acc, on_epoch=True, prog_bar=True, on_step=False)

        total_loss = sum(losses.values()) / len(losses)
        self.log(f"{prefix}_loss", total_loss, on_epoch=True, prog_bar=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self.do_step(batch)

    def validation_step(self, batch, batch_idx):
        return self.do_step(batch, "val")

    # def predict_step(self, batch, batch_idx, dataloader_idx=0):
    #     x = batch
    #     B = x.shape[0]
    #     pred_xy_sagittal, pred_xy_axial = self.forward(x)
    #     result = torch.concat([pred_xy_sagittal.reshape(B, -1), pred_xy_axial.reshape(B, -1)], axis=1)
    #     return result
