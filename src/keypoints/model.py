import timm
import torch

from src import model
from src.keypoints.data_loading import KEYPOINT_KEYS


class DistanceLoss(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        self.distance = torch.nn.PairwiseDistance()

    def forward(self, pred, target):
        mask = ((target >= 0) & (target <= 1)).all(-1)
        if ~mask.any():
            return -1.0

        result = self.distance(pred[mask], target[mask])
        if self.reduction == "mean":
            result = result.mean()
        return result


class PositionEncoding(torch.nn.Module):
    def __init__(self, n_feats: int, depth: int = 100):
        super().__init__()
        self.encoding = torch.nn.Parameter(torch.randn(1, n_feats, depth) / 100, requires_grad=True)

    def forward(self, x):
        return x + self.encoding


def get_pool(n_feats, size):
    return torch.nn.Sequential(
        torch.nn.AdaptiveMaxPool2d(size),
        torch.nn.Flatten(start_dim=2),
        PositionEncoding(n_feats, size * size),
        torch.nn.Linear(size * size, 1),
        torch.nn.Flatten(),
    )


class LightningModule(model.LightningModule):
    def __init__(self, arch: str = "resnet34", img_size: int = 448, pretrained: bool = True, **kwargs):
        super().__init__(**kwargs)

        self.in_channels = 1

        self.loss_xy = DistanceLoss(reduction="mean")

        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0, in_chans=self.in_channels).eval()
        n_feats = self.backbone.num_features

        depth = img_size // self.backbone.feature_info[-1]["reduction"]

        self.norm = torch.nn.InstanceNorm2d(self.in_channels)
        self.pool = get_pool(n_feats, depth)

        self.heads = torch.nn.ModuleDict({k: model.get_proj(None, 2, 0.0, torch.nn.Sigmoid()) for k in KEYPOINT_KEYS})

    def forward(self, x):
        norm_x = self.norm(x)
        feats = self.backbone.forward_features(norm_x)
        flatten_feats = self.pool(feats)
        outs = {k: head(flatten_feats) for k, head in self.heads.items()}
        return outs

    def do_step(self, batch, prefix="train"):
        x, target = batch
        preds = self.forward(x)

        losses = []
        for k in target:
            loss = self.loss_xy(preds[k], target[k])
            self.log(f"{prefix}_{k}_loss", loss, on_epoch=True, prog_bar=False, on_step=False)
            losses.append(loss)

        losses = [loss for loss in losses if loss >= 0]
        total_loss = sum(losses) / (len(losses) + 1e-16)
        self.log(f"{prefix}_loss", total_loss, on_epoch=True, prog_bar=True, on_step=prefix == "train")

        return total_loss

    def training_step(self, batch, batch_idx):
        return self.do_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.do_step(batch, "val")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch
        return self.forward(x)
