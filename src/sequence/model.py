from typing import Dict

import torch

from src import constants, model, losses


CONDS = [
    "left_neural_foraminal_narrowing",
    "right_neural_foraminal_narrowing",
    "spinal_canal_stenosis",
    "subarticular_stenosis"
]


PLANE2COND = dict(
    zip(
        constants.DESCRIPTIONS,
        (
            "spinal_canal_stenosis",
            "neural_foraminal_narrowing",
            "subarticular_stenosis"
        )
    )
)


def get_proj(in_features, out_features, dropout):
    return torch.nn.Sequential(
        torch.nn.Dropout(dropout),
        torch.nn.Linear(in_features, out_features) if in_features is not None else torch.nn.LazyLinear(out_features)
    )


class LightningModule(model.LightningModule):
    def __init__(
        self,
        emb_dim: int,
        n_heads: int,
        n_layers: int,
        att_dropout: float,
        linear_dropout: float,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.train_loss = losses.LumbarLoss()
        self.val_loss = losses.LumbarLoss()

        projs = {}
        transformers = {}
        for plane in constants.DESCRIPTIONS:
            projs[plane] = get_proj(None, emb_dim, linear_dropout)
            transformers[plane] = model.get_transformer(emb_dim, n_heads, n_layers, att_dropout)

        self.projs = torch.nn.ModuleDict(projs)
        self.transformers = torch.nn.ModuleDict(transformers)
        self.mid_transformer = model.get_transformer(emb_dim, n_heads, n_layers, att_dropout)
        # self.heads = torch.nn.ModuleDict({k: get_proj(emb_dim, 3, linear_dropout) for k in CONDS})
        self.heads = torch.nn.ModuleDict({k: get_proj(emb_dim, 3, linear_dropout) for k in constants.CONDITION_LEVEL})

    def forward_one(self, x, meta, mask, plane):
        x = x[plane]
        mask = mask[plane]
        _, _, P, _ = x.shape
        meta = meta[plane][:, :, None].repeat(1, 1, P, 1)
        x = torch.concat([x, meta], -1)
        x = self.projs[plane](x)
        outs = []
        for i in range(P):
            out = self.transformers[plane](x[:, :, i], src_key_padding_mask=mask)
            out[mask] = -100
            out = out.amax(1)
            outs.append(out)

        outs = torch.stack(outs, 1)

        return outs

    def get_out(self, plane, cond, seqs):
        sides = ("left", "right")
        levels = constants.LEVELS
        outs = {}
        match plane:

            case "Axial T2":
                for i, side in enumerate(sides):
                    seq = seqs[:, i]
                    for level in levels:
                        k = f"{side}_{cond}_{level}"
                        head = self.heads[k]
                        outs[k] = head(seq)

            case "Sagittal T1":
                for side in sides:
                    k = f"{side}_{cond}"
                    for i, level in enumerate(levels):
                        k1 = f"{k}_{level}"
                        head = self.heads[k1]
                        seq = seqs[:, i]
                        outs[k1] = head(seq)

            case "Sagittal T2/STIR":
                for i, level in enumerate(levels):
                    k = f"{cond}_{level}"
                    head = self.heads[k]
                    seq = seqs[:, i]
                    outs[k] = head(seq)

        return outs

    def forward(self, x, meta, mask) -> Dict[str, torch.Tensor]:
        seqs = []
        for plane in x:
            out = self.forward_one(x, meta, mask, plane)
            seqs.append(out)

        lens = [s.shape[1] for s in seqs]
        seqs = torch.concat(seqs, 1)

        seqs: torch.Tensor = self.mid_transformer(seqs)
        seqs = seqs.split(lens, 1)

        outs = {}
        for desc_seqs, (desc, cond) in zip(seqs, PLANE2COND.items()):
            outs.update(self.get_out(desc, cond, desc_seqs))

        return outs

    def training_step(self, batch, batch_idx):
        x, meta, mask, y_true_dict = batch
        y_pred_dict = self.forward(x, meta, mask)
        batch_size = y_pred_dict[constants.CONDITION_LEVEL[0]].shape[0]
        losses: Dict[str, torch.Tensor] = self.train_loss.jit_loss(y_true_dict, y_pred_dict)

        for k, v in losses.items():
            self.log(f"train_{k}_loss", v, on_epoch=True, prog_bar=False, on_step=False, batch_size=batch_size)

        loss = sum(losses.values()) / len(losses)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, on_step=False, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        x, meta, mask, y_true_dict = batch
        y_pred_dict = self.forward(x, meta, mask)
        self.val_loss.update(y_true_dict, y_pred_dict)

    def on_validation_epoch_end(self, *args, **kwargs):
        losses = self.val_loss.compute()
        self.val_loss.reset()
        for k, v in losses.items():
            self.log(f"val_{k}_loss", v, on_epoch=True, prog_bar=False, on_step=False)

        loss = sum(losses.values()) / len(losses)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, on_step=False)
