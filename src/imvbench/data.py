"""Build the train / validation / test data of one experiment.

  * the dataset's training split is divided 85 / 15 into train and validation with a generator
    seeded by `seed` (the test split is never used for training or selection);
  * views go missing per view at rate p (missingness.py), with the seeds train = seed,
    test = seed + 1, validation = seed + 2, and every sample keeps at least one view;
  * missing views are then filled by the chosen imputer.

The splits, missing matrices and imputer weight files are the same as in the two-head
repository, so its cached CPM-Nets / DCP weights can be reused (see README).
"""

import functools
from dataclasses import dataclass
from typing import Iterator, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split

from imvbench import config
from imvbench.datasets import DatasetSpec, get_dataset
from imvbench.imputers import impute_splits
from imvbench.missingness import generate_missing_matrix

VAL_FRACTION = 0.15


# --------------------------------------------------------------------------------------
# Containers
# --------------------------------------------------------------------------------------
@dataclass
class Split:
    views: List[torch.Tensor]   # V tensors [N, d_v], missing views already filled
    labels: torch.Tensor        # [N] int64
    mask: torch.Tensor          # [N, V] bool, True = view missing

    def __len__(self):
        return len(self.labels)

    def to(self, device) -> "Split":
        return Split([x.to(device) for x in self.views], self.labels.to(device), self.mask.to(device))

    def batches(self, batch_size: int, shuffle: bool) -> Iterator[Tuple[list, torch.Tensor, torch.Tensor]]:
        """Mini-batches (views, labels, mask); shuffling uses the global torch generator."""
        n = len(self)
        order = torch.randperm(n).to(self.labels.device) if shuffle else None
        for s in range(0, n, batch_size):
            if order is None:
                sl = slice(s, s + batch_size)
                yield [x[sl] for x in self.views], self.labels[sl], self.mask[sl]
            else:
                idx = order[s:s + batch_size]
                yield [x[idx] for x in self.views], self.labels[idx], self.mask[idx]


@dataclass
class ExperimentData:
    spec: DatasetSpec
    imputer: str
    rate: float
    seed: int
    train: Split
    val: Split
    test: Split

    def to(self, device) -> "ExperimentData":
        return ExperimentData(self.spec, self.imputer, self.rate, self.seed,
                              self.train.to(device), self.val.to(device), self.test.to(device))


# --------------------------------------------------------------------------------------
# Raw data (loaded once per process)
# --------------------------------------------------------------------------------------
def _materialise(ds: Dataset, num_views: int):
    """Dataset -> (list of [N, d_v] L2-normalised tensors, labels)."""
    if hasattr(ds, "views") and hasattr(ds, "labels"):        # fast path for the built-in loaders
        views = [F.normalize(torch.as_tensor(np.asarray(ds.views[v], dtype=np.float32)), p=2, dim=1)
                 for v in range(num_views)]
        return views, torch.as_tensor(np.asarray(ds.labels), dtype=torch.long)
    items = [ds[i] for i in range(len(ds))]
    views = [torch.stack([torch.as_tensor(d[f"view_{v}"], dtype=torch.float32) for d, _ in items])
             for v in range(num_views)]
    return views, torch.as_tensor([int(y) for _, y in items], dtype=torch.long)


@functools.lru_cache(maxsize=None)
def load_raw(name: str):
    spec = get_dataset(name)
    train, test = spec.loader(config.DATA_DIR)
    V = len(spec.view_dims)
    tr_views, tr_labels = _materialise(train, V)
    te_views, te_labels = _materialise(test, V)
    for v, d in spec.view_dims.items():
        if tr_views[v].shape[1] != d:
            raise ValueError(f"{name}: view {v} has {tr_views[v].shape[1]} features, spec says {d}")
    return tr_views, tr_labels, te_views, te_labels


def train_val_indices(n: int, seed: int):
    """Same permutation as torch.utils.data.random_split(dataset, ..., Generator(seed))."""
    val_size = int(VAL_FRACTION * n)
    tr, va = random_split(range(n), [n - val_size, val_size],
                          generator=torch.Generator().manual_seed(seed))
    return torch.tensor(tr.indices, dtype=torch.long), torch.tensor(va.indices, dtype=torch.long)


# --------------------------------------------------------------------------------------
# One experiment
# --------------------------------------------------------------------------------------
def prepare_splits(dataset: str, rate: float, seed: int):
    """Train / validation / test views with their missing-view masks, before imputation."""
    spec = get_dataset(dataset)
    V = len(spec.view_dims)
    tr_views, tr_labels, te_views, te_labels = load_raw(dataset)

    tr_idx, va_idx = train_val_indices(len(tr_labels), seed)
    raw = {
        "train": ([x[tr_idx] for x in tr_views], tr_labels[tr_idx]),
        "val": ([x[va_idx] for x in tr_views], tr_labels[va_idx]),
        "test": (te_views, te_labels),
    }
    mask_seed = {"train": seed, "test": seed + 1, "val": seed + 2}
    splits = {}
    for name, (views, labels) in raw.items():
        m = generate_missing_matrix(len(labels), V, rate, mask_seed[name])     # [V, N]
        splits[name] = {"views": views, "labels": labels, "mask": torch.from_numpy(m.T.copy())}
    return spec, splits


def build_data(dataset: str, imputer: str, rate: float, seed: int, device=None) -> ExperimentData:
    from imvbench import base_imputer
    imputer = base_imputer(imputer)                  # the mask variant uses the same imputation
    spec, splits = prepare_splits(dataset, rate, seed)

    imputer_device = device if device is not None else config.DEVICE
    filled = impute_splits(spec, imputer, splits, rate, seed, config.WEIGHTS_DIR, imputer_device)

    out = {k: Split(filled[k], splits[k]["labels"], splits[k]["mask"]) for k in splits}
    data = ExperimentData(spec, imputer, rate, seed, out["train"], out["val"], out["test"])
    return data.to(device) if device is not None else data
