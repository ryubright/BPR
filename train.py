import argparse
import torch
import time
import os
import numpy as np

from bpr import BPR
from data import load_data, BPRDataset
from util import set_seed, read_json
from torch.utils.data import DataLoader
from torch.optim import SGD
from metric import ndcg
from loss import bpr_loss
from torcheval.metrics.functional import hit_rate


def main(config):
    device = config["device"]
    train_data, test_data, n_user, n_item = load_data(config)
    train_data = train_data
    train_dataset = BPRDataset(interactions=train_data, n_neg=config["n_neg"], is_train=True)
    test_dataset = BPRDataset(interactions=test_data, n_neg=0, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["test_neg_n"], shuffle=False)

    model = BPR(n_user=n_user, n_item=n_item, n_factor=config["n_factor"])

    optimizer = SGD(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    best_hit = 0
    top_k = config["top_k"]
    model = model.to(config["device"])

    for epoch in range(config["epochs"]):
        start_time = time.time()

        total_loss = list()
        model.train()
        for user_id, pos_item_id, neg_item_id in train_loader:
            user_id = user_id.to(device)
            pos_item_id = pos_item_id.to(device)
            neg_item_id = neg_item_id.to(device)

            pos_scores, neg_scores = model(user_id, pos_item_id, neg_item_id)
            loss = bpr_loss(pos_scores, neg_scores)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.append(loss.detach().cpu().numpy())

        train_loss = np.mean(total_loss)

        model.eval()
        hr = eval(model, test_loader, device, top_k)

        print(f"epoch {epoch} | train_loss {train_loss:.4f} | hit_rate {hr:.3f} | elapsed_time {time.time() - start_time}")

        if hr > best_hit:
            best_hit, best_epoch = hr, epoch
            if not os.path.exists(config["save_dir"]):
                os.makedirs(config["save_dir"], exist_ok=True)
            torch.save(model, f"{config['save_dir']}/bpr.pt")


def eval(model, test_loader, device, top_k):
    hr = list()
    for user_id, item_id, _ in test_loader:
        user_id = user_id.to(device)
        item_id = item_id.to(device)

        scores, _ = model(user_id, item_id, item_id)

        hit = hit_rate(scores.unsqueeze(0), torch.tensor([0]).to(device), k=top_k)

        hr.append(hit.detach().cpu().numpy())

    return np.mean(hr)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="BPR")
    args.add_argument(
        "-c",
        "--config",
        default="./config.json",
        type=str,
        help='config file path (default: "./config.json")',
    )
    args = args.parse_args()
    config = read_json(args.config)
    set_seed(seed=config["seed"])

    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    main(config)
