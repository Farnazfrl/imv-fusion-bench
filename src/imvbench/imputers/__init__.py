"""Imputation methods.

  zero  missing views are set to 0
  mean  missing views are set to the view's mean over the observed training samples
  gaus  missing views are filled with L2-normalised Gaussian noise (fixed per sample)
  knn   cross-view kNN: average of the view in the 10 training samples closest on the
        observed views
  cpm   CPM-Nets (Zhang et al., NeurIPS 2019) reconstructs missing views from a learned latent
  dcp   DCP (Lin et al., TPAMI 2022) predicts missing views from the observed views
  mvae  MVAE (Wu & Goodman, NeurIPS 2018): multimodal VAE with a product-of-experts
        posterior; missing views are decoded from the posterior mean given the observed views
  drop  late fusion only: missing views are ignored when the per-view decisions are averaged
        (their inputs are zero-filled, but the model never uses them)
"""

from imvbench.imputers.methods import impute_splits  # noqa: F401
