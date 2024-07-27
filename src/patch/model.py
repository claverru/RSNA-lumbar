import torch
import timm

from src import model
from src.patch import constants as patch_constants


def get_head(in_features, out_features, dropout):
    return torch.nn.Sequential(
        # torch.nn.BatchNorm1d(in_features),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(in_features, out_features)
    )


class LightningModule(model.LightningModule):
    def __init__(self, arch: str = "resnet34", dropout: float = 0.0, eval: bool = True, pretrained: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = 1
        self.loss = torch.nn.CrossEntropyLoss(weight=torch.tensor([1, 2, 4, 0.2]), ignore_index=-1)
        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0, in_chans=self.in_channels)
        if eval:
            self.backbone = self.backbone.eval()
        n_feats = self.backbone.num_features
        self.norm = torch.nn.InstanceNorm2d(self.in_channels)
        self.heads = torch.nn.ModuleDict({k: get_head(n_feats, 4, dropout) for k in patch_constants.CONDITIONS})

    def forward(self, x):
        norm_x = self.norm(x)
        feats = self.backbone(norm_x)
        outs = {k: head(feats) for k, head in self.heads.items()}
        return outs

    def log_losses(self, losses, prefix="train"):
        for k, loss in losses.items():
            self.log(f"{prefix}_{k}", loss, on_epoch=True, prog_bar=False, on_step=False)

        total_loss = sum(losses.values()) / len(losses)
        self.log(f"{prefix}_loss", total_loss, on_epoch=True, prog_bar=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx):
        x, y_dict = batch
        pred_dict = self.forward(x)
        losses = {k: self.loss(pred_dict[k], y) for k, y in y_dict.items()}
        return self.log_losses(losses)

    def validation_step(self, batch, batch_idx):
        x, y_dict = batch
        pred_dict = self.forward(x)
        losses = {k: self.loss(pred_dict[k], y) for k, y in y_dict.items()}
        return self.log_losses(losses, "val")

    # def predict_step(self, batch, batch_idx, dataloader_idx=0):
    #     x = batch
    #     B = x.shape[0]
    #     pred_xy_sagittal, pred_xy_axial = self.forward(x)
    #     result = torch.concat([pred_xy_sagittal.reshape(B, -1), pred_xy_axial.reshape(B, -1)], axis=1)
    #     return result
