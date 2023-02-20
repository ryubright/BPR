import torch
import torch.nn as nn


class BPR(nn.Module):
    def __init__(self, n_user, n_item, n_factor):
        super().__init__()
        self.user_embedding = nn.Embedding(n_user, n_factor)
        self.item_embedding = nn.Embedding(n_item, n_factor)

        self.__weight_init()

    def __weight_init(self):
        nn.init.xavier_normal_(self.user_embedding.weight)
        nn.init.xavier_normal_(self.item_embedding.weight)

    def forward(self, user_id, item_id, neg_item_id):
        user_emb = self.user_embedding(user_id)
        pos_item_emb = self.item_embedding(item_id)
        neg_item_emb = self.item_embedding(neg_item_id)

        pos_pred = torch.mul(user_emb, pos_item_emb).sum(dim=-1)
        neg_pred = torch.mul(user_emb, neg_item_emb).sum(dim=-1)

        return pos_pred, neg_pred
