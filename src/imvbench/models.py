"""Fusion networks.

  early  concat(x_1..x_V) -> Linear(sum d_v, h) -> ReLU -> Linear(h, C)
  mid    per view: Linear(d_v, h) -> ReLU;  concat -> Linear(V*h, h) -> ReLU -> Linear(h, C)
  late   per view: Linear(d_v, h) -> ReLU -> Linear(h, C);  logits averaged over views
         * with an imputer: mean over all V views (imputed views included)
         * with "drop":     mean over the observed views of each sample only

With use_mask ("<imputer>+mask") the network also receives the missing-view indicator
m_n in {0,1}^V (1 = missing): appended to the input (early), to the concatenation before the
fusion layer (mid), or as one extra input of each view's branch (late).

Every model is trained with the plain mean cross-entropy over the batch.
"""

from typing import Dict, List

import torch
import torch.nn as nn


class FusionNet(nn.Module):
    def __init__(self, view_dims: Dict[int, int], num_classes: int, hidden: int,
                 fusion: str, drop_missing: bool = False, use_mask: bool = False):
        super().__init__()
        if fusion not in ("early", "mid", "late"):
            raise ValueError(f"Unknown fusion '{fusion}'")
        if drop_missing and fusion != "late":
            raise ValueError("drop_missing is only defined for late fusion")
        if drop_missing and use_mask:
            raise ValueError("drop already uses the mask")
        self.fusion, self.drop_missing, self.use_mask = fusion, drop_missing, use_mask
        dims = [view_dims[v] for v in sorted(view_dims)]
        V = len(dims)
        m = 1 if use_mask else 0          # extra inputs per view for the mask bit

        if fusion == "early":
            self.net = nn.Sequential(nn.Linear(sum(dims) + m * V, hidden), nn.ReLU(), nn.Linear(hidden, num_classes))
        elif fusion == "mid":
            self.encoders = nn.ModuleList([nn.Sequential(nn.Linear(d, hidden), nn.ReLU()) for d in dims])
            self.fuse = nn.Sequential(nn.Linear(V * hidden + m * V, hidden), nn.ReLU())
            self.classifier = nn.Linear(hidden, num_classes)
        else:
            self.branches = nn.ModuleList([
                nn.Sequential(nn.Linear(d + m, hidden), nn.ReLU(), nn.Linear(hidden, num_classes)) for d in dims])

    def forward(self, views: List[torch.Tensor], mask: torch.Tensor = None) -> torch.Tensor:
        """views: V tensors [B, d_v]; mask: [B, V] bool, True = missing (used by +mask and drop)."""
        mf = mask.to(views[0].dtype) if self.use_mask else None
        if self.fusion == "early":
            x = torch.cat(views + ([mf] if self.use_mask else []), dim=1)
            return self.net(x)
        if self.fusion == "mid":
            h = [enc(x) for enc, x in zip(self.encoders, views)]
            if self.use_mask:
                h.append(mf)
            return self.classifier(self.fuse(torch.cat(h, dim=1)))
        if self.use_mask:
            views = [torch.cat([x, mf[:, v:v + 1]], dim=1) for v, x in enumerate(views)]
        logits = torch.stack([b(x) for b, x in zip(self.branches, views)], dim=1)   # [B, V, C]
        if not self.drop_missing:
            return logits.mean(dim=1)
        w = (~mask).to(logits.dtype).unsqueeze(-1)                                 # [B, V, 1]
        return (logits * w).sum(dim=1) / w.sum(dim=1).clamp_min(1.0)


def build_model(spec, fusion: str, imputer: str, hidden: int) -> FusionNet:
    from imvbench import uses_mask
    return FusionNet(spec.view_dims, spec.num_classes, hidden, fusion,
                     drop_missing=(imputer == "drop"), use_mask=uses_mask(imputer))
