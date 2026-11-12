"""Training and evaluation of one model.

AdamW (decoupled weight decay), mean cross-entropy, fixed number of epochs; the weights of the
epoch with the best validation accuracy are restored at the end (model selection uses the
validation split only).
"""

import copy
import random

import numpy as np
import torch
import torch.nn.functional as F

from imvbench.models import build_model

EPOCHS = 20
BATCH_SIZE = 32


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def evaluate(model, split, batch_size: int = 1024) -> dict:
    """Accuracy overall and on complete / incomplete samples (in %)."""
    model.eval()
    correct, incomplete = [], []
    for views, labels, mask in split.batches(batch_size, shuffle=False):
        correct.append(model(views, mask).argmax(dim=1) == labels)
        incomplete.append(mask.any(dim=1))
    correct, incomplete = torch.cat(correct), torch.cat(incomplete)
    out = {"acc": 100.0 * correct.float().mean().item(), "n": int(len(correct)),
           "n_incomplete": int(incomplete.sum())}
    for name, sel in (("complete", ~incomplete), ("incomplete", incomplete)):
        out[f"acc_{name}"] = 100.0 * correct[sel].float().mean().item() if sel.any() else float("nan")
    return out


def fit(model, data, lr: float, weight_decay: float, epochs: int = EPOCHS,
        batch_size: int = BATCH_SIZE) -> dict:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_acc, best_epoch, best_state = -1.0, -1, None
    for epoch in range(epochs):
        model.train()
        for views, labels, mask in data.train.batches(batch_size, shuffle=True):
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(views, mask), labels)
            loss.backward()
            opt.step()
        val_acc = evaluate(model, data.val)["acc"]
        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch + 1
            best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return {"val_acc": best_acc, "best_epoch": best_epoch}


def run(data, fusion: str, imputer: str, hidden: int, lr: float, weight_decay: float,
        epochs: int = EPOCHS, batch_size: int = BATCH_SIZE, test: bool = False, device=None) -> dict:
    """Train one model on prepared data. The seed of the data also seeds the initialisation
    (seed) and the batch order (seed + 1), so all configurations of a seed start alike."""
    set_seed(data.seed)
    model = build_model(data.spec, fusion, imputer, hidden)
    if device is not None:
        model = model.to(device)
    set_seed(data.seed + 1)
    out = fit(model, data, lr, weight_decay, epochs, batch_size)
    if test:
        out["test"] = evaluate(model, data.test)
    return out
