from sklearn.preprocessing import LabelEncoder
from surprise import Dataset as SupriseDataset
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd
import numpy as np


def load_data(config):
    train_data = SupriseDataset.load_builtin("ml-100k")

    dtypes = {"user_id": int, "item_id": int}

    train_data = pd.DataFrame(train_data.raw_ratings, columns=["user_id", "item_id", "rating", "timestamp"])
    train_data = train_data.drop(["rating", "timestamp"], axis=1)
    train_data = train_data.astype(dtypes)

    encoder = LabelEncoder()
    train_data["user_id"] = encoder.fit_transform(train_data["user_id"])
    train_data["item_id"] = encoder.fit_transform(train_data["item_id"])

    n_user = train_data["user_id"].nunique()
    n_item = train_data["item_id"].nunique()

    test_file_path = config["test_file_path"]
    test_neg_n = config["test_neg_n"]

    interactions = list()
    with open(test_file_path, "r") as f:
        line = f.readline()
        while line != None and line != "":
            line = line.split(" ")
            positives = line[0][1:-1].split(",")
            user_id, pos_item_id = int(positives[0]) - 1, int(positives[1]) - 1

            interactions.append([user_id, pos_item_id])

            for neg_item_id in line[1:test_neg_n]:
                interactions.append([user_id, int(neg_item_id) - 1])
            line = f.readline()
    test_data = pd.DataFrame(interactions, columns=["user_id", "item_id"])

    return train_data, test_data, n_user, n_item


class BPRDataset(Dataset):
    def __init__(self, interactions, n_neg, is_train):
        super().__init__()
        self.is_train = is_train
        self.n_neg = n_neg
        self.n_item = interactions["item_id"].nunique()
        self.interactions = self.neg_sampling(interactions) if self.is_train else interactions
        # self.interactions = self.neg_sampling(interactions)

    def neg_sampling(self, interactions):
        neg_interactions = list()
        print("----------neg_sampling----------")
        for _, row in tqdm(interactions.iterrows()):
            user_id = row["user_id"]
            item_id = row["item_id"]
            for _ in range(self.n_neg):
                while True:
                    neg_item =np.random.randint(self.n_item)
                    if not ((interactions["user_id"] == user_id) & (interactions["item_id"] == neg_item)).any():
                        break

                neg_interactions.append([user_id, item_id, neg_item])

        return pd.DataFrame(neg_interactions, columns=["user_id", "item_id", "neg_item_id"])

    def __len__(self):
        return len(self.interactions)

    def __getitem__(self, idx):
        interaction = self.interactions.iloc[idx]

        user_id = interaction["user_id"]
        item_id = interaction["item_id"]
        neg_item_id = interaction["neg_item_id"] if self.is_train else 1

        return user_id, item_id, neg_item_id
