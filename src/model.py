import torch
import lightning as L
import timm

from src import constants


class Model(torch.nn.Module):
    def __init__(self, arch: str):
        super().__init__()
        self.backbone = timm.create_model(arch, pretrained=True, num_classes=0)
        n_feats = self.backbone.num_features
        self.heads = torch.nn.ModuleDict({out: torch.nn.Linear(n_feats, 3) for out in constants.CONDITION_LEVEL})

    def forward(self, x):
        feats = self.backbone(x)
        outs = {k: head(feats) for k, head in self.heads.items()}
        return outs


class LightningModule(L.LightningModule):
    def __init__(self, arch: str = "resnet34"):
        super().__init__()
        self.loss_f = torch.nn.CrossEntropyLoss(ignore_index=-1, weight=torch.tensor([1., 2., 4.]))
        self.backbone = timm.create_model(arch, pretrained=True, num_classes=0)
        n_feats = self.backbone.num_features
        self.heads = torch.nn.ModuleDict({out: torch.nn.Linear(n_feats, 3) for out in constants.CONDITION_LEVEL})

    def forward(self, x):
        feats = self.backbone(x)
        outs = {k: head(feats) for k, head in self.heads.items()}
        return outs

    def training_step(self, batch, batch_idx):
        x, y_true_dict = batch
        y_pred_dict = self.forward(x)
        loss = 0.0
        for k, y_true in y_true_dict.items():
            if (y_true == -1).all():
                continue
            y_pred = y_pred_dict[k]
            loss += self.loss_f(y_pred, y_true)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true_dict = batch
        y_pred_dict = self.forward(x)
        loss = 0.0
        for k, y_true in y_true_dict.items():
            if (y_true == -1).all():
                continue
            y_pred = y_pred_dict[k]
            loss += self.loss_f(y_pred, y_true)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, on_step=False)
        return loss
