"""A small synthetic two-view dataset used by the tests (registered via IMVB_PLUGINS)."""

import numpy as np
from torch.utils.data import Dataset

from imvbench.datasets import DatasetSpec, register_dataset


class Synthetic(Dataset):
    """Two views, three classes; each view alone is informative."""

    def __init__(self, n, seed):
        rng = np.random.default_rng(seed)
        self.labels = rng.integers(0, 3, n)
        centres = [np.random.default_rng(99 + v).normal(size=(3, d)) for v, d in enumerate((12, 8))]
        self.views = {v: (centres[v][self.labels] + 1.5 * rng.normal(size=(n, d))).astype(np.float32)
                      for v, d in enumerate((12, 8))}

    def __len__(self):
        return len(self.labels)


register_dataset(DatasetSpec(
    name="synthetic", num_classes=3, view_dims={0: 12, 1: 8},
    loader=lambda root: (Synthetic(400, 0), Synthetic(150, 1)),
    imputer_hidden=16, cpm_epochs=2, dcp_epochs=2, mvae_epochs=2))
