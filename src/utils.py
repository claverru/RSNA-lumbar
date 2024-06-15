from typing import Dict, List

import torch


def cat_preds(preds: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    result = {}
    for k in preds[0]:
        result[k] = torch.concat([pred[k] for pred in preds], dim=0)
    return result
