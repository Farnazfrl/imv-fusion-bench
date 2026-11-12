"""Fill the missing views of the train, validation and test splits.

All functions work on whole tensors: views is a list of [N, d_v] float tensors and mask a
[N, V] bool tensor (True = missing). The procedures follow the two-head repository
(core/data.py and centralized/data_setup.py, (c) 2026 Brecht Lauwers, MIT), and the CPM-Nets /
DCP weight files keep the same names, so weights cached by that repository are reused.
"""

import fcntl
import os
import socket
from contextlib import contextmanager
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from imvbench.imputers.cpm_nets import CPMNet_Works
from imvbench.imputers.dcp_nets import DCP_Net, dcp_loss

CHUNK = 8192    # rows decoded at once by CPM-Nets / DCP


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
class _IndexDataset(Dataset):
    """(idx, {"view_v": x}, label, {"v": 1 if observed else 0}): input format of CPM-Nets and DCP."""

    def __init__(self, views: List[torch.Tensor], labels: torch.Tensor, mask: torch.Tensor):
        self.views, self.labels = views, labels
        self.sn = (~mask).float()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = {f"view_{v}": x[idx] for v, x in enumerate(self.views)}
        sn = {str(v): self.sn[idx, v] for v in range(len(self.views))}
        return idx, data, self.labels[idx], sn


@contextmanager
def _locked(path: str):
    """Only one process trains a given imputer; the others wait and then load its weights."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".lock", "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _save(obj, path: str):
    """Atomic write through a temporary file unique to this process: file locks are not always
    honoured across the nodes of a cluster file system, so two processes may train the same
    imputer at once; each then writes its own temporary file and the last rename wins."""
    tmp = f"{path}.{socket.gethostname()}.{os.getpid()}.tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _fill(views, mask, recon: Dict[int, torch.Tensor]):
    """Copy of views with the missing rows of view v replaced by recon[v] (rows in mask order)."""
    out = []
    for v, x in enumerate(views):
        x = x.clone()
        miss = mask[:, v]
        if v in recon and miss.any():
            x[miss] = recon[v].to(x.dtype)
        out.append(x)
    return out


# --------------------------------------------------------------------------------------
# Zero and Gaussian noise
# --------------------------------------------------------------------------------------
def zero_fill(views, mask):
    return _fill(views, mask, {v: torch.zeros(int(mask[:, v].sum()), x.shape[1])
                               for v, x in enumerate(views)})


def gaus_fill(views, mask, intensity: float = 1.0):
    """Same noise as the original GausImputeWrapper: for sample idx a generator seeded with
    idx + 99999 draws one vector per missing view, in view order."""
    out = [x.clone() for x in views]
    rows = torch.nonzero(mask.any(dim=1)).flatten().tolist()
    for idx in rows:
        rng = torch.Generator()
        rng.manual_seed(idx + 99999)
        for v, x in enumerate(out):
            if mask[idx, v]:
                noise = torch.randn(x.shape[1], generator=rng)
                x[idx] = torch.nn.functional.normalize(noise, p=2, dim=0) * intensity
    return out


# --------------------------------------------------------------------------------------
# Mean and cross-view kNN (fitted on the training split only)
# --------------------------------------------------------------------------------------
def view_means(train):
    """Mean of each view over the training samples where that view is observed."""
    means = []
    for v, x in enumerate(train["views"]):
        obs = ~train["mask"][:, v]
        means.append(x[obs].mean(dim=0) if obs.any() else torch.zeros(x.shape[1]))
    return means


def mean_fill(views, mask, means):
    return _fill(views, mask, {v: means[v].expand(int(mask[:, v].sum()), -1) for v in range(len(views))})


KNN_K = 10
KNN_CHUNK = 1024


def knn_fill(views, mask, train, means, device, k: int = KNN_K):
    """Cross-view kNN. For a sample with observed views O and a missing view i, the donors are
    the training samples that observe O and i; the k donors closest to the sample on the
    concatenated views O (Euclidean; views are L2-normalised) give view i as their average.
    A sample never serves as its own donor (it does not observe i). Falls back to the training
    mean of view i if no donor exists."""
    V = len(views)
    tr_views, tr_mask = train["views"], train["mask"]
    recon_rows = {v: [] for v in range(V)}
    recon_idx = {v: [] for v in range(V)}
    patterns = {}
    for idx, row in enumerate(mask.tolist()):
        if any(row):
            patterns.setdefault(tuple(row), []).append(idx)
    with torch.no_grad():
        for pattern, rows in patterns.items():
            obs = [v for v in range(V) if not pattern[v]]
            rows = torch.tensor(rows)
            query = torch.cat([views[v][rows] for v in obs], dim=1).to(device)
            q_sq = (query ** 2).sum(dim=1, keepdim=True)
            for i in (v for v in range(V) if pattern[v]):
                donor_sel = ~tr_mask[:, obs + [i]].any(dim=1)
                if not donor_sel.any():
                    recon_rows[i].append(means[i].expand(len(rows), -1).clone())
                    recon_idx[i].append(rows)
                    continue
                donors = torch.cat([tr_views[v][donor_sel] for v in obs], dim=1).to(device)
                targets = tr_views[i][donor_sel].to(device)
                d_sq = (donors ** 2).sum(dim=1)
                kk = min(k, len(donors))
                out = []
                for s in range(0, len(rows), KNN_CHUNK):
                    dist = q_sq[s:s + KNN_CHUNK] + d_sq[None, :] - 2.0 * query[s:s + KNN_CHUNK] @ donors.T
                    nn_idx = dist.topk(kk, dim=1, largest=False).indices          # [chunk, kk]
                    out.append(targets[nn_idx].mean(dim=1).cpu())
                recon_rows[i].append(torch.cat(out))
                recon_idx[i].append(rows)
    recon = {}
    for v in range(V):
        if recon_rows[v]:
            idx, rec = torch.cat(recon_idx[v]), torch.cat(recon_rows[v])
            recon[v] = rec[torch.argsort(idx)]
    return _fill(views, mask, recon)


# --------------------------------------------------------------------------------------
# CPM-Nets
# --------------------------------------------------------------------------------------
def _cpm_decode(model, h_bank, mask):
    """Reconstruct only the missing rows (decoders in eval mode, in chunks)."""
    recon = {}
    with torch.no_grad():
        for v in range(mask.shape[1]):
            idx = torch.nonzero(mask[:, v]).flatten()
            parts = []
            for s in range(0, len(idx), CHUNK):
                h = h_bank[idx[s:s + CHUNK].to(h_bank.device)]
                parts.append(model.net[str(v)](h).cpu())
            if parts:
                recon[v] = torch.cat(parts)
    return recon


def cpm_impute(spec, splits, rate, seed, weights_dir, device):
    tr, va, te = splits["train"], splits["val"], splits["test"]
    view_dims = spec.view_dims
    weights = os.path.join(weights_dir, f"cpm_weights_{spec.name}_missing{rate}_seed{seed}_val.pth")
    val_h = os.path.join(weights_dir, f"cpm_val_h_{spec.name}_missing{rate}_seed{seed}.pth")

    model = CPMNet_Works(device=device, view_dims=view_dims, trainLen=len(tr["labels"]),
                         testLen=len(te["labels"]), lsd_dim=spec.imputer_hidden).to(device)
    with _locked(weights):
        if os.path.exists(weights):
            print(f"[{spec.name}] CPM-Nets: loading {weights}", flush=True)
            model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
        else:
            print(f"[{spec.name}] CPM-Nets: training ({spec.cpm_epochs} epochs) -> {weights}", flush=True)
            model.train_model(DataLoader(_IndexDataset(tr["views"], tr["labels"], tr["mask"]),
                                         batch_size=128, shuffle=True),
                              num_classes=spec.num_classes, epoch=spec.cpm_epochs)
            model.test_model(DataLoader(_IndexDataset(te["views"], te["labels"], te["mask"]),
                                        batch_size=128, shuffle=False), epoch=spec.cpm_epochs)
            _save(model.state_dict(), weights)
    model.eval()

    # validation latents: decoders fixed, latents fitted to the observed validation views
    model_val = CPMNet_Works(device=device, view_dims=view_dims, trainLen=1,
                             testLen=len(va["labels"]), lsd_dim=spec.imputer_hidden).to(device)
    model_val.net.load_state_dict(model.net.state_dict())
    with _locked(val_h):
        if os.path.exists(val_h):
            saved = torch.load(val_h, map_location=device, weights_only=True)
            model_val.h_test.data.copy_(saved["h_val"])
        else:
            print(f"[{spec.name}] CPM-Nets: fitting validation latents -> {val_h}", flush=True)
            model_val.test_model(DataLoader(_IndexDataset(va["views"], va["labels"], va["mask"]),
                                            batch_size=128, shuffle=False), epoch=spec.cpm_epochs)
            _save({"h_val": model_val.h_test.data}, val_h)
    model_val.eval()

    return {
        "train": _fill(tr["views"], tr["mask"], _cpm_decode(model, model.h_train, tr["mask"])),
        "val": _fill(va["views"], va["mask"], _cpm_decode(model_val, model_val.h_test, va["mask"])),
        "test": _fill(te["views"], te["mask"], _cpm_decode(model, model.h_test, te["mask"])),
    }


# --------------------------------------------------------------------------------------
# DCP
# --------------------------------------------------------------------------------------
def _dcp_recon(model, views, mask, device):
    """For every missing view: mean of the latents predicted from each observed view, decoded.
    Samples are processed in groups that share the same missing pattern."""
    V = len(views)
    recon_rows = {v: [] for v in range(V)}
    recon_idx = {v: [] for v in range(V)}
    patterns = {}
    for idx, row in enumerate(mask.tolist()):
        if any(row):
            patterns.setdefault(tuple(row), []).append(idx)
    with torch.no_grad():
        for pattern, rows in patterns.items():
            avail = [v for v in range(V) if not pattern[v]]
            missing = [v for v in range(V) if pattern[v]]
            rows = torch.tensor(rows)
            for s in range(0, len(rows), CHUNK):
                r = rows[s:s + CHUNK]
                z = {j: model.encoders[str(j)].encoder(views[j][r].to(device)) for j in avail}
                for i in missing:
                    preds = [model.predictors[f"{j}_to_{i}"](z[j])[0] for j in avail]
                    fused = torch.stack(preds).mean(dim=0)
                    recon_rows[i].append(model.encoders[str(i)].decoder(fused).cpu())
                    recon_idx[i].append(r)
    recon = {}
    for v in range(V):
        if recon_rows[v]:
            idx = torch.cat(recon_idx[v])
            rec = torch.cat(recon_rows[v])
            order = torch.argsort(idx)              # back to the order of mask[:, v]
            recon[v] = rec[order]
    return recon


def dcp_impute(spec, splits, rate, seed, weights_dir, device):
    tr = splits["train"]
    weights = os.path.join(weights_dir, f"dcp_weights_{spec.name}_missing{rate}_seed{seed}.pth")
    model = DCP_Net(view_dims=spec.view_dims, latent_dim=128, hidden_dim=spec.imputer_hidden).to(device)
    with _locked(weights):
        if os.path.exists(weights):
            print(f"[{spec.name}] DCP: loading {weights}", flush=True)
            model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
        else:
            print(f"[{spec.name}] DCP: training ({spec.dcp_epochs} epochs) -> {weights}", flush=True)
            loader = DataLoader(_IndexDataset(tr["views"], tr["labels"], tr["mask"]),
                                batch_size=128, shuffle=True)
            opt = torch.optim.Adam(model.parameters(), lr=0.001)
            model.train()
            for epoch in range(spec.dcp_epochs):
                total = 0.0
                for _, inputs, _, sn in loader:
                    inputs = {k: x.to(device) for k, x in inputs.items()}
                    sn = {k: x.to(device) for k, x in sn.items()}
                    opt.zero_grad()
                    loss = dcp_loss(model, inputs, sn)
                    loss.backward()
                    opt.step()
                    total += loss.item()
                if (epoch + 1) % 10 == 0:
                    print(f"DCP epoch {epoch + 1}/{spec.dcp_epochs} | loss {total / len(loader):.4f}", flush=True)
            _save(model.state_dict(), weights)
    model.eval()
    return {name: _fill(s["views"], s["mask"], _dcp_recon(model, s["views"], s["mask"], device))
            for name, s in splits.items()}


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------
def impute_splits(spec, imputer, splits, rate, seed, weights_dir, device):
    """splits: {"train"|"val"|"test": {"views", "labels", "mask"}} -> {split: imputed views}.
    Mean, kNN, CPM-Nets and DCP are fitted on the training split only."""
    if imputer in ("zero", "drop"):
        return {k: zero_fill(s["views"], s["mask"]) for k, s in splits.items()}
    if imputer == "gaus":
        return {k: gaus_fill(s["views"], s["mask"]) for k, s in splits.items()}
    if imputer == "mean":
        means = view_means(splits["train"])
        return {k: mean_fill(s["views"], s["mask"], means) for k, s in splits.items()}
    if imputer == "knn":
        means = view_means(splits["train"])
        return {k: knn_fill(s["views"], s["mask"], splits["train"], means, device) for k, s in splits.items()}
    if imputer == "cpm":
        return cpm_impute(spec, splits, rate, seed, weights_dir, device)
    if imputer == "dcp":
        return dcp_impute(spec, splits, rate, seed, weights_dir, device)
    raise ValueError(f"Unknown imputer '{imputer}'")
