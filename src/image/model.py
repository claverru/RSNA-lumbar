import math
from typing import Dict, Tuple
import torch
import timm

from src import constants, model


def get_proj(in_dim, out_dim, dropout):
    return torch.nn.Sequential(
        torch.nn.Dropout(dropout),
        torch.nn.Linear(in_dim, out_dim)
    )


def get_block(in_dim, out_dim):
    return torch.nn.Sequential(
        torch.nn.Linear(in_dim, in_dim // 16),
        torch.nn.ReLU(),
        torch.nn.Linear(in_dim // 16, in_dim // 64),
        torch.nn.ReLU(),
        torch.nn.Linear(in_dim // 64, out_dim)
    )


class DistanceLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.distance = torch.nn.PairwiseDistance()

    def forward(self, pred, true):
        return self.distance(pred, true)[true[:, 0] != -1].mean()


class HierarchyHead(torch.nn.Module):
    def __init__(
        self,
        n_feats: int,
        emb_dim: int,
        dropout: float,
        from_levels: bool = False,
        with_unseen: bool = True,
        levels_and_conditions: bool = False
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_levels = len(constants.LEVELS)
        self.n_conditions = len(constants.CONDITIONS_COMPLETE)
        self.n_severities = len(constants.SEVERITY2LABEL)
        if with_unseen:
            self.n_severities += 1

        self.from_levels = from_levels
        if not from_levels:
            if levels_and_conditions:
                self.levels = get_proj(n_feats, self.n_levels * self.n_conditions * emb_dim, dropout)
            else:
                self.levels = get_proj(n_feats, self.n_levels * emb_dim, dropout)

        if levels_and_conditions:
            self.op = "BLCD,CSD->BLCS"
            self.reshape = (-1, self.n_levels, self.n_conditions, self.emb_dim)
        else:
            self.op = "BLD,CSD->BLCS"
            self.reshape = (-1, self.n_levels, self.emb_dim)

        self.condition_severity = torch.nn.Parameter(
            torch.randn(self.n_conditions, self.n_severities, emb_dim), requires_grad=True
        )
        self.loss_f = torch.nn.CosineEmbeddingLoss()

    def forward(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = feats.shape[0]
        if self.from_levels:
            levels = feats
        else:
            levels = self.levels(feats).reshape(self.reshape)
        out = torch.einsum(self.op, levels, self.condition_severity)
        return levels, out

    def do_loss(self):
        x = self.condition_severity.reshape(-1, self.emb_dim)
        ids = torch.combinations(torch.arange(x.shape[0]))
        xs = x[ids]
        x0 = xs[:, 0]
        x1 = xs[:, 1]
        # print(x.shape, xs.shape, x0.shape, x1.shape)
        w = -torch.ones(xs.shape[0], dtype=xs.dtype, device=xs.device)
        return self.loss_f(x0, x1, w)


class CoorsHead(torch.nn.Module):
    def __init__(self, n_feats: int, n_coors: int = 2):
        super().__init__()
        self.n_coors = n_coors
        self.block = torch.nn.Sequential(
            get_block(n_feats, len(constants.CONDITION_LEVEL) * n_coors),
            torch.nn.Sigmoid()
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        B = feats.shape[0]
        return self.block(feats).reshape(B, len(constants.CONDITION_LEVEL), self.n_coors) # (B, CL, XYZ)



class PositionEncoding(torch.nn.Module):
    def __init__(self, depth=100):
        super().__init__()
        position = torch.linspace(0, math.pi, depth, dtype=torch.float)
        encoding = torch.sin(position)
        encoding = encoding[None, None, :] / encoding.sum()
        self.register_buffer("encoding", encoding)

    def forward(self, x):
        return x + self.encoding


def get_pool(size):
    return torch.nn.Sequential(
        torch.nn.AdaptiveMaxPool2d(size),
        torch.nn.Flatten(start_dim=2),
        PositionEncoding(size * size),
        torch.nn.Linear(size * size, 1),
        torch.nn.Flatten()
    )


class LightningModule(model.LightningModule):
    def __init__(self, arch: str = "resnet34", emb_dim: int = 32, dropout: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = 1
        self.loss_f = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 2., 4., 0.2]))
        self.loss_xy_f = DistanceLoss()
        self.loss_z_f = torch.nn.MSELoss()
        self.loss_plane_f = torch.nn.CrossEntropyLoss()
        self.backbone = timm.create_model(arch, pretrained=True, num_classes=0, in_chans=self.in_channels).eval()
        n_feats = self.backbone.num_features

        self.norm = torch.nn.InstanceNorm2d(self.in_channels)
        self.pool = get_pool(8)

        self.head = HierarchyHead(n_feats, emb_dim, dropout)
        self.xy_head = CoorsHead(n_feats, n_coors=2)
        self.z_head = torch.nn.Sequential(torch.nn.Linear(emb_dim * len(constants.LEVELS), 1), torch.nn.Sigmoid())
        self.plane = torch.nn.Linear(emb_dim * len(constants.LEVELS), len(constants.DESCRIPTIONS))

    def forward_train(self, x):
        _, _, out, coor, z, plane = self.forward_features(x)
        outs = {}
        for i, level in enumerate(constants.LEVELS):
            for j, condition in enumerate(constants.CONDITIONS_COMPLETE):
                k = f"{condition}_{level}"
                outs[k] = out[:, i, j]
        xy = {}
        for i, cl in enumerate(constants.CONDITION_LEVEL):
            xy[cl] = coor[:, i]
        return outs, xy, z, plane

    def forward_features(self, x):
        B = x.shape[0]
        x = self.norm(x)
        feats = self.backbone.forward_features(x)
        feats = self.pool(feats)
        levels, out = self.head(feats)
        xy = self.xy_head(feats)
        z = self.z_head(levels.reshape(B, -1))
        plane = self.plane(levels.reshape(B, -1))
        return feats, levels, out, xy, z, plane

    def forward(self, x):
        feats, levels, out, xy, z, plane = self.forward_features(x)
        B = x.shape[0]
        levels = levels.reshape(B, -1)
        out = out.reshape(B, -1)
        xy = xy.reshape(B, -1)
        return torch.concat([feats, levels, out, xy, z, plane], 1)

    def do_cls_loss(self, y_true_dict, y_pred_dict):
        return sum(self.loss_f(y_pred_dict[k], y_true_dict[k]) for k in constants.CONDITION_LEVEL) / len(y_true_dict)

    def do_xy_loss(self, y_xy, pred_xy):
        y = torch.concat([y_xy[k] for k in constants.CONDITION_LEVEL], 0)
        pred = torch.concat([pred_xy[k] for k in constants.CONDITION_LEVEL], 0)
        loss = self.loss_xy_f(pred, y)
        return loss

    def do_loss(self, y_cls, y_xy, y_z, y_plane, pred_cls, pred_xy, pred_z, pred_plane, prefix="", log=True):
        cls_loss = self.do_cls_loss(y_cls, pred_cls)
        coor_loss = self.do_xy_loss(y_xy, pred_xy)
        z_loss = self.loss_z_f(pred_z, y_z)
        plane_loss = self.loss_plane_f(pred_plane, y_plane)
        sim_loss = self.head.do_loss()
        total_loss = cls_loss + coor_loss + z_loss + plane_loss + sim_loss
        if log:
            self.log(prefix + "cls_loss", cls_loss, on_epoch=True, prog_bar=True, on_step=False)
            self.log(prefix + "xy_loss", coor_loss, on_epoch=True, prog_bar=True, on_step=False)
            self.log(prefix + "z_loss", z_loss, on_epoch=True, prog_bar=True, on_step=False)
            self.log(prefix + "plane_loss", plane_loss, on_epoch=True, prog_bar=True, on_step=False)
            self.log(prefix + "sim_loss", sim_loss, on_epoch=True, prog_bar=True, on_step=False)
            self.log(prefix + "loss", total_loss, on_epoch=True, prog_bar=False, on_step=False)
        return total_loss

    def training_step(self, batch, batch_idx):
        x, y_cls, y_xy, y_z, y_plane = batch
        pred_cls, pred_xy, pred_z, pred_plane = self.forward_train(x)
        return self.do_loss(y_cls, y_xy, y_z, y_plane, pred_cls, pred_xy, pred_z, pred_plane)

    def validation_step(self, batch, batch_idx):
        x, y_cls, y_xy, y_z, y_plane = batch
        pred_cls, pred_xy, pred_z, pred_plane = self.forward_train(x)
        return self.do_loss(y_cls, y_xy, y_z, y_plane, pred_cls, pred_xy, pred_z, pred_plane, "val_")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        X = batch
        return self.forward(X)
