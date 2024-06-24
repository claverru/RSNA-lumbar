from typing import Dict

import torch
import lightning as L

from src import constants


def get_lstm(i, o, n, d):
    return torch.nn.GRU(i, o, n, dropout=d, bidirectional=True)


class LightningModule(L.LightningModule):
    def __init__(self, emb_dim, lstms, dropout):
        super().__init__()
        self.loss_f = torch.nn.CrossEntropyLoss(ignore_index=-1, weight=torch.tensor([1., 2., 4.]))
        self.spinal_loss_f = torch.nn.NLLLoss(ignore_index=-1, reduction="none")

        self.seq = torch.nn.ModuleDict({c: get_lstm(1792 * 5 + 1, emb_dim, lstms, dropout) for c in constants.CONDITIONS})
        self.heads = torch.nn.ModuleDict({cl: torch.nn.Linear(emb_dim * 2 * 3, 3) for cl in constants.CONDITION_LEVEL})

    @staticmethod
    def rnn_pool(module, packed_seqs):
        out = module(packed_seqs)[0]
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True, padding_value=-100)
        out = out.max(1)[0]
        return out

    def forward(self, x: Dict[str, torch.nn.utils.rnn.PackedSequence]) -> Dict[str, torch.Tensor]:
        seq = {k: self.rnn_pool(m, i) for (k, m), (_, i) in zip(self.seq.items(), x.items())}
        cat = torch.concat(list(seq.values()), dim=1)
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
        x, y_true_dict = batch
        y_pred_dict = self.forward(x)
        losses = self.do_loss(y_true_dict, y_pred_dict)
        batch_size = y_pred_dict[constants.CONDITION_LEVEL[0]].shape[0]

        for k, v in losses.items():
            self.log(f"train_{k}_loss", v, on_epoch=True, prog_bar=True, on_step=False, batch_size=batch_size)

        loss = sum(losses.values()) / len(losses)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True, on_step=False, batch_size=batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y_true_dict = batch
        y_pred_dict = self.forward(x)
        losses = self.do_loss(y_true_dict, y_pred_dict)
        batch_size = y_pred_dict[constants.CONDITION_LEVEL[0]].shape[0]

        for k, v in losses.items():
            self.log(f"val_{k}_loss", v, on_epoch=True, prog_bar=True, on_step=False, batch_size=batch_size)

        loss = sum(losses.values()) / len(losses)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, on_step=False, batch_size=batch_size)

        return loss
