from typing import Dict

import timm
import torch
import lightning as L

from src import constants

M = 18

def get_att(emb_dim, n_heads, n_layers, dropout):
    layer = torch.nn.TransformerEncoderLayer(
        emb_dim, n_heads, dropout=dropout, dim_feedforward=emb_dim * 2, batch_first=True
    )
    encoder = torch.nn.TransformerEncoder(layer, n_layers)
    return encoder


def get_proj(feats, emb_dim):
    return torch.nn.Sequential(
        torch.nn.Linear(feats, emb_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(emb_dim, emb_dim),
    )


def get_head(in_features, dropout):
    return torch.nn.Sequential(
        torch.nn.Dropout(dropout),
        torch.nn.Linear(in_features, 3)
    )


class LightningModule(L.LightningModule):
    def __init__(self, arch, emb_dim, n_heads, n_layers, att_dropout, linear_dropout, pretrained=True):
        super().__init__()
        self.loss_f = torch.nn.CrossEntropyLoss(ignore_index=-1, weight=torch.tensor([1., 2., 4.]))
        self.spinal_loss_f = torch.nn.NLLLoss(ignore_index=-1, reduction="none")

        self.backbone = timm.create_model(arch, num_classes=0, in_chans=1, pretrained=pretrained, global_pool="max")#.eval()
        self.F = self.backbone.num_features

        self.proj = torch.nn.ModuleDict(
            {c: get_proj(self.F + M, emb_dim) for c in constants.DESCRIPTIONS}
        )

        self.seq = torch.nn.ModuleDict(
            {c: get_att(emb_dim, n_heads, n_layers, att_dropout) for c in constants.DESCRIPTIONS}
        )
        self.heads = torch.nn.ModuleDict(
            {cl: get_head(emb_dim * 3, linear_dropout) for cl in constants.CONDITION_LEVEL}
        )


    def forward_one(self, x: torch.Tensor, metadata: torch.Tensor, description: str) -> torch.Tensor:
        # metadata: B, L, M
        B, L, C, H, W = x.shape

        padding_mask = metadata.mean(-1) == -1
        true_mask = padding_mask.logical_not()

        filter_x = x[true_mask]

        filter_feats: torch.Tensor = self.backbone(filter_x)

        feats = torch.zeros((B, L, self.F), device=filter_feats.device, dtype=filter_feats.dtype)
        feats[true_mask] = filter_feats


        feats = torch.concat([feats, metadata], -1)
        feats = self.proj[description](feats)

        o = self.seq[description](feats, src_key_padding_mask=padding_mask)
        o[padding_mask] = -100
        o = o.max(1)[0]
        return o

    def forward(self, x, metadata) -> Dict[str, torch.Tensor]:
        seqs = []
        for d in constants.DESCRIPTIONS:
            seq = self.forward_one(x[d], metadata[d], d)
            seqs.append(seq)

        cat = torch.concat(seqs, dim=1)
        outs = {k: head(cat) for k, head in self.heads.items()}
        return outs

    def compute_severe_spinal_loss(self, y_true_dict, y_pred_dict):
        severe_spinal_true = torch.stack(
            [y_true_dict[k] for k in constants.CONDITION_LEVEL if "spinal" in k], -1
        ).max(-1)[0]
        is_empty_spinal_batch = (severe_spinal_true == -1).all()

        if is_empty_spinal_batch:
            return -1

        severe_spinal_preds = torch.stack(
            [torch.softmax(y_pred_dict[k], -1)[:, -1] for k in constants.CONDITION_LEVEL if "spinal" in k],
            -1
        ).max(-1)[0]

        severe_spinal_preds = torch.stack([1 - severe_spinal_preds, severe_spinal_preds], -1)
        weight = torch.pow(2.0, severe_spinal_true)
        severe_spinal_binary_true = torch.where(
            severe_spinal_true > 0, severe_spinal_true - 1, severe_spinal_true
        )

        severe_spinal_losses = self.spinal_loss_f(
            torch.log(severe_spinal_preds), severe_spinal_binary_true
        ) * weight
        return severe_spinal_losses[severe_spinal_true != -1].mean()


    def compute_condition_loss(self, y_true_dict, y_pred_dict, condition):
        true_batch = [y_true for k, y_true in y_true_dict.items() if condition in k]
        true_batch = torch.concat(true_batch, 0)

        is_empty_batch = (true_batch == -1).all()
        if is_empty_batch:
            return -1

        pred_batch = [y_true for k, y_true in y_pred_dict.items() if condition in k]
        pred_batch = torch.concat(pred_batch, 0)

        return self.loss_f(pred_batch, true_batch)

    def do_loss(self, y_true_dict: Dict[str, torch.Tensor], y_pred_dict: Dict[str, torch.Tensor]):
        losses = {}
        for condition in constants.CONDITIONS:
            losses[condition] = self.compute_condition_loss(y_true_dict, y_pred_dict, condition)
        losses["severe_spinal"] = self.compute_severe_spinal_loss(y_true_dict, y_pred_dict)
        return losses

    def training_step(self, batch, batch_idx):
        x, metadata, y_true_dict = batch
        y_pred_dict = self.forward(x, metadata)
        losses = self.do_loss(y_true_dict, y_pred_dict)
        batch_size = y_pred_dict[constants.CONDITION_LEVEL[0]].shape[0]

        for k, v in losses.items():
            self.log(f"train_{k}_loss", v, on_epoch=True, prog_bar=True, on_step=False, batch_size=batch_size)

        loss = sum(losses.values()) / len(losses)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, on_step=False, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        x, metadata, y_true_dict = batch
        y_pred_dict = self.forward(x, metadata)
        losses = self.do_loss(y_true_dict, y_pred_dict)
        batch_size = y_pred_dict[constants.CONDITION_LEVEL[0]].shape[0]

        for k, v in losses.items():
            self.log(f"val_{k}_loss", v, on_epoch=True, prog_bar=True, on_step=False, batch_size=batch_size)

        loss = sum(losses.values()) / len(losses)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, on_step=False, batch_size=batch_size)

        return loss
