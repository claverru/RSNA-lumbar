import torch
import timm

from src import constants, model


def get_head(in_dim, out_dim, dropout):
    return torch.nn.Sequential(
        torch.nn.Dropout(dropout),
        torch.nn.Linear(in_dim, out_dim)
    )


class LightningModule(model.LightningModule):
    def __init__(self, arch: str = "resnet34", level_emb_dim: int = 32, dropout: float = 0.2, **kwargs):
        super().__init__(**kwargs)
        self.loss_f = torch.nn.CrossEntropyLoss(ignore_index=-1, weight=torch.tensor([1., 2., 4.]))
        self.backbone = timm.create_model(arch, pretrained=True, num_classes=0, in_chans=1).eval()
        n_feats = self.backbone.num_features

        self.levels = torch.nn.ModuleDict({out: get_head(n_feats, level_emb_dim, dropout) for out in constants.LEVELS})
        self.heads = torch.nn.ModuleDict({out: get_head(level_emb_dim, 3, dropout) for out in constants.CONDITION_LEVEL})

    def forward(self, x):
        feats = self.backbone(x)
        levels = {k: m(feats) for k, m in self.levels.items()}
        outs = {}
        for level in constants.LEVELS:
            for condition in constants.CONDITIONS_COMPLETE:
                k = f"{condition}_{level}"
                outs[k] = self.heads[k](levels[level])

        return outs

    def forward_features(self, x):
        feats = self.backbone(x)
        levels = {k: m(feats) for k, m in self.levels.items()}
        outs = {}
        for level in constants.LEVELS:
            for condition in constants.CONDITIONS_COMPLETE:
                k = f"{condition}_{level}"
                outs[k] = self.heads[k](levels[level])
        result = torch.concat([feats] + list(levels.values()) + list(outs.values()), -1)
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
        y_pred_dict = self.forward(x)
        loss = self.do_loss(y_true_dict, y_pred_dict)
        self.log('train_loss', loss, on_epoch=True, prog_bar=True, on_step=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true_dict = batch
        y_pred_dict = self.forward(x)
        loss = self.do_loss(y_true_dict, y_pred_dict)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, on_step=False)
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        X = batch
        return self.forward_features(X)
