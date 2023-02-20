import torch

def bpr_loss(pos_pred, neg_pred):
    loss = -(pos_pred - neg_pred).sigmoid().log().sum()
    return loss