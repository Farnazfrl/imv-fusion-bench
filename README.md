# imv-fusion-bench

Fusion and imputation for neural classifiers on incomplete multi-view data. The benchmark compares

* **fusion**: early, mid (concatenation), late (mean of per-view logits);
* **imputation**: zero, mean, Gaussian noise, cross-view kNN, CPM-Nets, DCP, and for late fusion
  also **drop** (each sample averages only the decisions of its observed views);
* **mask indicator**: every imputer also as `<imputer>+mask`, where the network additionally
  receives the missing-view mask m_n ∈ {0,1}^V (early: appended to the input; mid: appended to
  the concatenation before the fusion layer; late: one extra input per view branch);
* **missing rate**: each view is missing independently with probability p ∈ {0.1, 0.3, 0.5, 0.7},
  and every sample keeps at least one view.

Every (dataset, fusion, imputer) configuration has its **own** hyperparameter search:
37 configurations per dataset (3 fusions × 6 imputers × with/without mask, plus late drop).

## Protocol

| | |
|---|---|
| Models | early: concat → Linear(Σd, h) → ReLU → Linear(h, C) |
| | mid: per view Linear(d_v, h) → ReLU; concat → Linear(V·h, h) → ReLU → Linear(h, C) |
| | late: per view Linear(d_v, h) → ReLU → Linear(h, C); mean of logits (all views, or observed views only for `drop`) |
| Training | AdamW, mean cross-entropy, batch 32, 20 epochs, weights of the best validation epoch kept |
| Splits | 15 % of the training split is the validation set (seeded); the test split is used only for the final evaluation |
| Missingness | train / test / validation masks use seeds s, s+1, s+2 |
| Search space | h ∈ {32, 64, 128, 256, 512}; lr ∈ [1e-4, 1e-2] log; weight decay ∈ [1e-6, 1e-1] log |
| Tuning | Optuna TPE (seed 0), 60 trials, objective = mean validation accuracy over p ∈ {0.3, 0.7} × seeds {7, 11, 13}; median pruning after p = 0.3 |
| Evaluation | tuned configuration retrained for seeds {50, 121, 893, 17, 2024} × p ∈ {0.1, 0.3, 0.5, 0.7}; test accuracy overall and on complete / incomplete samples |
| Report | per-dataset tables, averages, average ranks, paired Wilcoxon tests with Holm correction (mask vs no mask, imputers vs zero, fusions, late drop vs late imputed) |

## Setup

```bash
git clone <this repo> imv-fusion-bench && cd imv-fusion-bench
uv sync                      # creates .venv from uv.lock
uv run python -m pytest -q   # optional: synthetic end-to-end test, under a minute
```

Data, imputer weights and results live in folders set by environment variables:

| variable | default | meaning |
|---|---|---|
| `IMVB_DATA_DIR` | `data` | raw features (layout below) |
| `IMVB_WEIGHTS_DIR` | `weights` | cached CPM-Nets / DCP models |
| `IMVB_RESULTS_DIR` | `results` | Optuna studies, evaluation runs, report |
| `TORCH_DEVICE` | auto | force `cpu` or `cuda:0` |

Missing imputer weights are trained on first use (a file lock makes sure only one process trains
a given model); `precompute` trains them all in advance. The splits, missing matrices and weight
file names are the same as in the two-head repository, so its cached weights can be reused.

Expected layout of `IMVB_DATA_DIR`:

```
ModelNet40_mvcnn_gvcnn.mat
sensit_vehicle_features/{acoustic, seismic, acoustic.t, seismic.t}
uci_har_dataset/{features.txt, train/X_train.txt, train/y_train.txt, test/X_test.txt, test/y_test.txt}
youtube/{train,test}/{text_game_lda_1000.txt, vision_cuboids_histogram.txt, audio_mfcc.txt}
mosei/mosei_pooled.npz                     (made by scripts/prepare_mosei.py, see below)
handwritten/mfeat-{fou,fac,kar,pix,zer,mor}
```

### Handwritten (UCI Multiple Features)

Download the "Multiple Features" data set from the UCI Machine Learning Repository
(archive.ics.uci.edu, data set 72) and unzip it into `data/handwritten/` (the six files
`mfeat-fou`, `mfeat-fac`, `mfeat-kar`, `mfeat-pix`, `mfeat-zer`, `mfeat-mor`, directly or in a
`mfeat/` subfolder). Dataset `handwritten`: 2,000 digits, 10 classes, 6 views (76 Fourier, 216
profile-correlation, 64 Karhunen-Loève, 240 pixel, 47 Zernike and 6 morphological features). As
there is no official test set, a stratified 20 % test split (400 digits) is drawn once with a
fixed seed.

### CMU-MOSEI

1. Download `mosei_senti_data.pkl` (word-aligned CMU-MOSEI) from the data link in the README of
   the Multimodal Transformer repository (github.com/yaohungt/Multimodal-Transformer, section
   "Datasets").
2. Pool each modality over time (mean and std over the real time steps):
   `python scripts/prepare_mosei.py data/mosei/mosei_senti_data.pkl data/mosei`
   (needs a few GB of RAM; the pickle is not needed afterwards).
3. Datasets `mosei7` (7 sentiment classes) and `mosei` (negative vs non-negative) are then
   available: views text (600), audio (148), vision (70); official train + valid are used for
   training, official test for testing.

### SensIT Vehicle

`scripts/split_sensit_combined.py` creates the row-aligned `acoustic` / `seismic` files from
LIBSVM's `combined` files.

## Imputers

| name | what fills a missing view |
|---|---|
| zero | 0 |
| mean | mean of that view over the observed training samples |
| gaus | L2-normalised Gaussian noise, fixed per sample |
| knn | mean of the view in the 10 training samples closest on the sample's observed views (donors observe all of them) |
| cpm | CPM-Nets reconstruction |
| dcp | DCP cross-view prediction |
| drop | nothing: late fusion averages the observed views only |

Append `+mask` (e.g. `cpm+mask`) to also give the network the missing-view mask.

## Running

```bash
uv run python -m imvbench tasks --datasets har                 # the 37 configurations and their indices
uv run python -m imvbench run --datasets har --workers 3        # tune + evaluate all of them
uv run python -m imvbench run --datasets har --task-index 7     # one configuration (e.g. one job of an array)
uv run python -m imvbench status --datasets har                 # progress
uv run python -m imvbench report --datasets har sensit          # -> results/report.md, results/cells.csv
```

`tune` and `evaluate` can also be run separately; `precompute --datasets har --imputers cpm dcp`
trains the CPM-Nets / DCP imputers for all rates and seeds. Selection options: `--datasets`,
`--fusions`, `--imputers` (e.g. `zero cpm+mask drop`), `--task-index`. Other options:
`--trials`, `--tune-rates`, `--tune-seeds`, `--eval-rates`, `--eval-seeds`, `--epochs`,
`--batch-size`, `--workers` (parallel processes, one per seed), `--threads`, `--cache-size`.

Everything resumes: a restarted study continues from its finished trials, and evaluation skips
finished (rate, seed) runs. If a study is re-tuned, its old evaluation file is renamed `*.stale`.

Imputer cost (fit + fill, CPM-Nets and DCP trained from scratch) and one classifier training run:

```bash
uv run python scripts/time_imputers.py --datasets har sensit modelnet mosei7 --out results/timing.csv
```

## Adding a dataset

Write a `Dataset` whose items are `({"view_0": x0, "view_1": x1, ...}, label)` (or that has
`views` = {v: array [N, d_v]} and `labels` attributes, which is faster), then register it:

```python
from imvbench.datasets import DatasetSpec, register_dataset

register_dataset(DatasetSpec(
    name="mydata", num_classes=10, view_dims={0: 300, 1: 128},
    loader=lambda root: (MyData(root, "train"), MyData(root, "test")),
    imputer_hidden=128, cpm_epochs=20, dcp_epochs=100))
```

Put this in `src/imvbench/datasets.py`, or in your own module and set
`IMVB_PLUGINS=my_module`. Views are L2-normalised per sample. Then use `--datasets mydata`
(37 new configurations, `--task-index 0..36`).

## Layout

```
src/imvbench/
  datasets.py      dataset loaders + registry
  missingness.py   per-view MCAR missing matrices
  imputers/        zero, mean, Gaussian, kNN, CPM-Nets, DCP
  data.py          splits, masks, imputation -> tensors
  models.py        early / mid / late fusion networks (+ mask input)
  train.py         AdamW training, best-validation selection, evaluation
  runner.py        parallel worker processes + data cache
  tune.py          Optuna search per configuration
  evaluate.py      final runs
  report.py        tables and tests
  cli.py           command line
scripts/           data preparation (MOSEI, SensIT), imputer timing
tests/             synthetic end-to-end tests
```

## Credits

The dataset loaders, the missingness generator and the CPM-Nets / DCP imputers are ported from
Brecht Lauwers' two-head repository (MIT). CPM-Nets: Zhang et al., NeurIPS 2019. DCP: Lin et
al., TPAMI 2022.
