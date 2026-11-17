"""Datasets and the dataset registry.

Every dataset returns, per sample, a dict {"view_0": tensor, "view_1": tensor, ...} with each
view L2-normalised, and an integer label. The four loaders are ported unchanged from the
two-head repository (core/datasets.py, (c) 2026 Brecht Lauwers, MIT licence).

Adding a dataset
----------------
Write a function that takes the data folder and returns (train_dataset, test_dataset), then
register it:

    register_dataset(DatasetSpec(
        name="mydata", num_classes=10, view_dims={0: 300, 1: 128},
        loader=lambda root: (MyData(root, "train"), MyData(root, "test")),
        imputer_hidden=128, cpm_epochs=20, dcp_epochs=100))

Put the registration in this file, or in your own module and list that module in the
environment variable IMVB_PLUGINS (comma-separated module names on the Python path); the
dataset can then be used everywhere with --datasets mydata.
"""

import importlib
import os
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import scipy.io as scio
import torch
from torch.utils.data import Dataset


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DatasetSpec:
    name: str
    num_classes: int
    view_dims: Dict[int, int]                         # view index -> feature dimension
    loader: Callable[[str], Tuple[Dataset, Dataset]]  # data folder -> (train, test)
    imputer_hidden: int = 128    # CPM-Nets latent size and DCP hidden size
    cpm_epochs: int = 20
    dcp_epochs: int = 100
    mvae_epochs: int = 100
    description: str = ""


DATASETS: Dict[str, DatasetSpec] = {}


def register_dataset(spec: DatasetSpec) -> DatasetSpec:
    DATASETS[spec.name] = spec
    return spec


def get_dataset(name: str) -> DatasetSpec:
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Registered: {', '.join(sorted(DATASETS))}")
    return DATASETS[name]


def _normalised(views: Dict[int, np.ndarray], idx: int) -> Dict[str, torch.Tensor]:
    return {f"view_{v}": torch.nn.functional.normalize(torch.from_numpy(x[idx]), p=2, dim=0)
            for v, x in views.items()}


# --------------------------------------------------------------------------------------
# ModelNet40 (MVCNN 4096 + GVCNN 2048 features)
# --------------------------------------------------------------------------------------
class ModelNet40Features(Dataset):
    def __init__(self, features_path: str, split: str = "all"):
        try:
            data = scio.loadmat(features_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"features file not found at {features_path}")
        labels = data["Y"].astype(np.int64).squeeze()
        if labels.min() == 1:                      # MATLAB indexing
            labels = labels - 1
        indices = data["indices"].item().squeeze()
        view0 = data["X"][0].item().astype(np.float32)
        view1 = data["X"][1].item().astype(np.float32)
        if split == "train":
            mask = indices == 1
        elif split == "test":
            mask = indices == 0
        else:
            mask = np.ones_like(indices, dtype=bool)
        self.labels = labels[mask]
        self.views = {0: view0[mask], 1: view1[mask]}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return _normalised(self.views, idx), self.labels[idx]


# --------------------------------------------------------------------------------------
# SensIT Vehicle (acoustic 50 + seismic 50); classes AAV and DW (the noise class is dropped)
# --------------------------------------------------------------------------------------
class VehicleSensITFeatures(Dataset):
    """Expects the LIBSVM files acoustic, seismic, acoustic.t, seismic.t with the same sample on
    the same line in both views. scripts/split_sensit_combined.py can create them from LIBSVM's
    'combined' files (first 50 features acoustic, last 50 seismic)."""

    def __init__(self, root_dir: str, split: str = "all"):
        from sklearn.datasets import load_svmlight_file
        train_files = {"v0": os.path.join(root_dir, "acoustic"), "v1": os.path.join(root_dir, "seismic")}
        test_files = {"v0": os.path.join(root_dir, "acoustic.t"), "v1": os.path.join(root_dir, "seismic.t")}
        files = {"train": [train_files], "test": [test_files]}.get(split, [train_files, test_files])
        v0, v1, ys = [], [], []
        for f in files:
            if not os.path.exists(f["v0"]) or not os.path.exists(f["v1"]):
                raise FileNotFoundError(f"Missing data files in {root_dir}")
            x0, y0 = load_svmlight_file(f["v0"], n_features=50)
            x1, _ = load_svmlight_file(f["v1"], n_features=50)
            v0.append(x0.toarray().astype(np.float32))
            v1.append(x1.toarray().astype(np.float32))
            ys.append(y0)
        view0, view1 = np.concatenate(v0), np.concatenate(v1)
        labels = np.concatenate(ys).astype(np.int64)
        keep = labels != 3                               # keep AAV and DW
        labels = labels[keep]
        if labels.min() == 1:
            labels = labels - 1
        self.labels = labels
        self.views = {0: view0[keep], 1: view1[keep]}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return _normalised(self.views, idx), self.labels[idx]


# --------------------------------------------------------------------------------------
# UCI-HAR (accelerometer 348 + gyroscope 213 features)
# --------------------------------------------------------------------------------------
class UCIHARFeatures(Dataset):
    def __init__(self, root_dir: str, split: str = "all"):
        features_path = os.path.join(root_dir, "features.txt")
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"features not found in {root_dir}")
        with open(features_path) as f:
            names = [line.strip().split()[1] for line in f]
        acc_idx = [i for i, n in enumerate(names) if "Gyro" not in n]
        gyro_idx = [i for i, n in enumerate(names) if "Gyro" in n]
        splits = {"train": ["train"], "test": ["test"]}.get(split, ["train", "test"])
        xs, ys = [], []
        for sp in splits:
            x_path = os.path.join(root_dir, sp, f"X_{sp}.txt")
            y_path = os.path.join(root_dir, sp, f"y_{sp}.txt")
            if not os.path.exists(x_path) or not os.path.exists(y_path):
                raise FileNotFoundError(f"Missing data files for split '{sp}' in {root_dir}")
            xs.append(np.loadtxt(x_path, dtype=np.float32))
            ys.append(np.loadtxt(y_path, dtype=np.int64))
        X = np.concatenate(xs)
        labels = np.concatenate(ys).squeeze()
        if labels.min() == 1:
            labels = labels - 1
        self.labels = labels
        self.views = {0: X[:, acc_idx], 1: X[:, gyro_idx]}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return _normalised(self.views, idx), self.labels[idx]


# --------------------------------------------------------------------------------------
# YouTube Multiview Video Games (text 1000 + vision 512 + audio 2000)
# --------------------------------------------------------------------------------------
class YoutubeFeatures(Dataset):
    FILES = {0: "text_game_lda_1000.txt", 1: "vision_cuboids_histogram.txt", 2: "audio_mfcc.txt"}

    def __init__(self, root_dir: str, split: str, view_dims: Dict[int, int]):
        split_dir = os.path.join(root_dir, split)
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Split directory not found at {split_dir}")
        parsed = {v: {} for v in self.FILES}
        labels = {}
        for v, name in self.FILES.items():
            path = os.path.join(split_dir, name)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Feature file missing: {path}")
            dim, current = view_dims[v], None
            with open(path, "rt") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("#I"):
                        current = int(line.split()[1])
                    elif not line.startswith("#"):
                        parts = line.split()
                        labels[current] = int(parts[0]) - 1      # 1..31 -> 0..30
                        x = np.zeros(dim, dtype=np.float32)
                        for pair in parts[1:]:
                            fid, val = pair.split(":")
                            fid = int(fid) - 1
                            if fid < dim:
                                x[fid] = float(val)
                        parsed[v][current] = x
        ids = sorted(set(parsed[0]) & set(parsed[1]) & set(parsed[2]))   # videos with all views
        self.views = {v: np.stack([parsed[v][i] for i in ids]) for v in self.FILES}
        self.labels = np.array([labels[i] for i in ids], dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return _normalised(self.views, idx), self.labels[idx]


# --------------------------------------------------------------------------------------
# CMU-MOSEI (text 600 + audio 148 + vision 70: mean and std over time per modality)
# --------------------------------------------------------------------------------------
class ArrayViews(Dataset):
    """Views already stored as arrays: {v: [N, d_v]} and labels [N]."""

    def __init__(self, views: Dict[int, np.ndarray], labels: np.ndarray):
        self.views, self.labels = views, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return _normalised(self.views, idx), self.labels[idx]


def mosei_loader(root: str, classes: int = 2):
    """Reads mosei/mosei_pooled.npz (scripts/prepare_mosei.py). Each feature is standardised
    with the training statistics (and clipped to +-10) before the per-sample L2 normalisation.
    Labels: 2 classes = negative (< 0) vs non-negative sentiment; 7 classes = rounded score."""
    path = os.path.join(root, "mosei", "mosei_pooled.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found: run scripts/prepare_mosei.py first (see README)")
    z = np.load(path)
    out = []
    stats = {}
    for split in ("train", "test"):
        views = {}
        for v, m in enumerate(("text", "audio", "vision")):
            x = z[f"{m}_{split}"].astype(np.float32)
            if split == "train":
                stats[m] = (x.mean(axis=0), x.std(axis=0) + 1e-6)
            mu, sd = stats[m]
            views[v] = np.clip((x - mu) / sd, -10, 10).astype(np.float32)
        y = z[f"y_{split}"]
        labels = (y >= 0).astype(np.int64) if classes == 2 else (np.round(np.clip(y, -3, 3)) + 3).astype(np.int64)
        out.append(ArrayViews(views, labels))
    return tuple(out)


# --------------------------------------------------------------------------------------
# Handwritten digits (UCI Multiple Features, "mfeat"): 2000 digits, 6 views
# --------------------------------------------------------------------------------------
MFEAT_VIEWS = [("fou", 76), ("fac", 216), ("kar", 64), ("pix", 240), ("zer", 47), ("mor", 6)]


def handwritten_loader(root: str, test_fraction: float = 0.2, split_seed: int = 0):
    """Reads the six files mfeat-fou, -fac, -kar, -pix, -zer, -mor from <root>/handwritten (or
    <root>/handwritten/mfeat). Rows are ordered by digit, 200 per digit 0..9. There is no official
    test set: a stratified 20 % test split is drawn once with a fixed seed. Each feature is
    standardised with the training statistics (and clipped to +-10) before the per-sample L2
    normalisation, because the features of a view have very different scales."""
    folder = os.path.join(root, "handwritten")
    if os.path.isdir(os.path.join(folder, "mfeat")):
        folder = os.path.join(folder, "mfeat")
    raw = {}
    for v, (name, dim) in enumerate(MFEAT_VIEWS):
        path = os.path.join(folder, f"mfeat-{name}")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found: unzip the UCI 'Multiple Features' data into {folder}")
        x = np.loadtxt(path, dtype=np.float32).reshape(-1, dim)
        if x.shape != (2000, dim):
            raise ValueError(f"{path}: expected 2000 x {dim}, got {x.shape}")
        raw[v] = x
    labels = np.repeat(np.arange(10, dtype=np.int64), 200)
    rng = np.random.default_rng(split_seed)
    test = np.zeros(len(labels), dtype=bool)
    for c in range(10):
        idx = np.where(labels == c)[0]
        test[rng.choice(idx, size=int(round(test_fraction * len(idx))), replace=False)] = True
    out = []
    stats = {v: (x[~test].mean(axis=0), x[~test].std(axis=0) + 1e-6) for v, x in raw.items()}
    for sel in (~test, test):
        views = {v: np.clip((x[sel] - stats[v][0]) / stats[v][1], -10, 10).astype(np.float32)
                 for v, x in raw.items()}
        out.append(ArrayViews(views, labels[sel]))
    return tuple(out)


# --------------------------------------------------------------------------------------
# The four multi-view datasets (settings as in the two-head repository)
# --------------------------------------------------------------------------------------
register_dataset(DatasetSpec(
    name="modelnet", num_classes=40, view_dims={0: 4096, 1: 2048},
    loader=lambda root: (ModelNet40Features(os.path.join(root, "ModelNet40_mvcnn_gvcnn.mat"), "train"),
                         ModelNet40Features(os.path.join(root, "ModelNet40_mvcnn_gvcnn.mat"), "test")),
    imputer_hidden=256, cpm_epochs=50, dcp_epochs=100, description="ModelNet40, MVCNN + GVCNN"))

register_dataset(DatasetSpec(
    name="sensit", num_classes=2, view_dims={0: 50, 1: 50},
    loader=lambda root: (VehicleSensITFeatures(os.path.join(root, "sensit_vehicle_features"), "train"),
                         VehicleSensITFeatures(os.path.join(root, "sensit_vehicle_features"), "test")),
    imputer_hidden=128, cpm_epochs=20, dcp_epochs=100, description="SensIT Vehicle, acoustic + seismic"))

register_dataset(DatasetSpec(
    name="har", num_classes=6, view_dims={0: 348, 1: 213},
    loader=lambda root: (UCIHARFeatures(os.path.join(root, "uci_har_dataset"), "train"),
                         UCIHARFeatures(os.path.join(root, "uci_har_dataset"), "test")),
    imputer_hidden=256, cpm_epochs=20, dcp_epochs=100, description="UCI-HAR, accelerometer + gyroscope"))

register_dataset(DatasetSpec(
    name="youtube", num_classes=31, view_dims={0: 1000, 1: 512, 2: 2000},
    loader=lambda root: (YoutubeFeatures(os.path.join(root, "youtube"), "train", {0: 1000, 1: 512, 2: 2000}),
                         YoutubeFeatures(os.path.join(root, "youtube"), "test", {0: 1000, 1: 512, 2: 2000})),
    imputer_hidden=128, cpm_epochs=50, dcp_epochs=100, description="YouTube Multiview Video Games"))


register_dataset(DatasetSpec(
    name="mosei", num_classes=2, view_dims={0: 600, 1: 148, 2: 70},
    loader=lambda root: mosei_loader(root, classes=2),
    imputer_hidden=128, cpm_epochs=20, dcp_epochs=100,
    description="CMU-MOSEI sentiment, negative vs non-negative; text + audio + vision"))

register_dataset(DatasetSpec(
    name="mosei7", num_classes=7, view_dims={0: 600, 1: 148, 2: 70},
    loader=lambda root: mosei_loader(root, classes=7),
    imputer_hidden=128, cpm_epochs=20, dcp_epochs=100,
    description="CMU-MOSEI sentiment, 7 classes (rounded score); text + audio + vision"))


register_dataset(DatasetSpec(
    name="handwritten", num_classes=10, view_dims={v: d for v, (_, d) in enumerate(MFEAT_VIEWS)},
    loader=handwritten_loader, imputer_hidden=128, cpm_epochs=50, dcp_epochs=100,
    description="UCI Multiple Features (handwritten digits), 6 views"))


# Extra datasets registered by user modules (IMVB_PLUGINS=module_a,module_b)
for _module in filter(None, (m.strip() for m in os.environ.get("IMVB_PLUGINS", "").split(","))):
    importlib.import_module(_module)
