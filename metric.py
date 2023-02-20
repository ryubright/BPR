import numpy as np
import torch


def ndcg(pred, pos_item_id):
    if pos_item_id in pred:
        idx = pred.index(pos_item_id)
        return np.reciprocal(np.log2(idx + 2))
    return 0
