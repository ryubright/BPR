# BPR
PyTorch Implementation of ["BPR: Bayesian Personalized Ranking from Implicit Feedback"](https://arxiv.org/ftp/arxiv/papers/1205/1205.2618.pdf)

<br>

## 1. BPR optimization
BPR-OPT is an optimization technique for personalized ranking.

![image](https://user-images.githubusercontent.com/59256704/220161215-5f9b0c6b-4845-4c8e-be8a-11d6b404571b.png)

```python
def bpr_loss(pos_scores, neg_scores):
    loss = -(pos_scores - neg_scores).sigmoid().log().sum()
    return loss
```

<br>

## 2. Requirements

```text
numpy==1.24.1
pandas==1.5.2
scikit_learn==1.2.1
scikit_surprise==1.1.3
surprise==0.1
torch==1.13.1
torcheval==0.0.6
tqdm==4.64.1
```

<br>

## 3. Example run
- set config.json
```json
{
  "seed": 417,
  "n_neg": 4,
  "batch_size": 4096,
  "n_factor": 32,
  "learning_rate": 0.001,
  "weight_decay": 0,
  "epochs": 50,
  "top_k": 10,
  "test_file_path": "./data/ml-100k.test.negative",
  "test_neg_n": 100,
  "save_dir": "./save_models"
}
```

- run python code
```bash
python train.py --config [config.json file path]
```