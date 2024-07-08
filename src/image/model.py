from typing import Dict, Tuple
import torch
import timm

from src import constants, model


def get_head(in_dim, out_dim, dropout):
    return torch.nn.Sequential(
        torch.nn.Dropout(dropout),
        torch.nn.Linear(in_dim, out_dim)
    )


class HierarchyHead(torch.nn.Module):
    def __init__(self, n_feats: int, emb_dim: int, dropout: float, from_levels: bool = False, with_unseen: bool = True):
        super().__init__()
        self.emb_dim = emb_dim
        self.n_levels = len(constants.LEVELS)
        self.n_conditions = len(constants.CONDITIONS_COMPLETE)
        self.n_severities = len(constants.SEVERITY2LABEL)
        if with_unseen:
            self.n_severities += 1

        self.from_levels = from_levels
        if not from_levels:
            self.levels = get_head(n_feats, self.n_conditions * self.n_levels * emb_dim, dropout)
        self.condition_severity = torch.nn.Parameter(
            torch.randn(self.n_conditions, self.n_severities, emb_dim), requires_grad=True
        )

    def forward(self, feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = feats.shape[0]
        if self.from_levels:
            levels = feats
        else:
            levels = self.levels(feats).reshape(B, self.n_conditions, self.n_levels, self.emb_dim)
        out = torch.einsum("BCLD,CSD->BCLS", levels, self.condition_severity)
        return levels, out

    def forward_train(self, feats: torch.Tensor) -> Dict[str, torch.Tensor]:
        _, out = self.forward(feats)
        outs = {}
        for i, c in enumerate(constants.CONDITIONS_COMPLETE):
            for j, l in enumerate(constants.LEVELS):
                k = f"{c}_{l}"
                outs[k] = out[:, i, j]
        return outs


class LightningModule(model.LightningModule):
    def __init__(self, arch: str = "resnet34", emb_dim: int = 32, dropout: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.loss_f = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 2., 4., 0.2]))
        backbone = timm.create_model(arch, pretrained=True, num_classes=0, in_chans=1).eval()
        n_feats = backbone.num_features
        self.backbone = torch.nn.Sequential(torch.nn.InstanceNorm2d(1), backbone)
        self.head = HierarchyHead(n_feats, emb_dim, dropout)

    def forward_train(self, x):
        feats = self.backbone(x)
        outs = self.head.forward_train(feats)
        return outs

    def forward(self, x):
        feats = self.backbone(x)
        levels, out = self.head.forward(feats)
        B = feats.shape[0]
        levels = levels.reshape(B, -1)
        out = out.reshape(B, -1)
        result = torch.concat([feats, levels, out], -1)
        return result

    def do_loss(self, y_true_dict, y_pred_dict):
        loss = 0.0
        n = 0
        for k, y_true in y_true_dict.items():
            if (y_true == -1).all():
                continue
            y_pred = y_pred_dict[k]
            loss += self.loss_f(y_pred, y_true)
            n += 1
        return loss / n

    def training_step(self, batch, batch_idx):
        x, y_true_dict = batch
        y_pred_dict = self.forward_train(x)
        loss = self.do_loss(y_true_dict, y_pred_dict)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true_dict = batch
        y_pred_dict = self.forward_train(x)
        loss = self.do_loss(y_true_dict, y_pred_dict)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, on_step=False)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        X = batch
        return self(X)
