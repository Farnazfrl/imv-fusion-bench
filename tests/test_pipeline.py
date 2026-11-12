"""End-to-end tests on a small synthetic dataset (no real data needed).

    uv run python -m pytest -q tests          (or: uv run python tests/test_pipeline.py)
"""

import os
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
_TMP = tempfile.mkdtemp(prefix="imvb_test_")
os.environ["IMVB_RESULTS_DIR"] = os.path.join(_TMP, "results")
os.environ["IMVB_WEIGHTS_DIR"] = os.path.join(_TMP, "weights")
os.environ["TORCH_DEVICE"] = "cpu"
os.environ["IMVB_DATA_DIR"] = os.path.join(_TMP, "data")

os.environ["IMVB_PLUGINS"] = "synthetic_data"          # registers the "synthetic" dataset
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from imvbench import configurations                                  # noqa: E402
from imvbench.missingness import generate_missing_matrix             # noqa: E402


def test_missing_matrix():
    for rate in (0.1, 0.5, 0.9):
        m = generate_missing_matrix(5000, 3, rate, seed=3)
        assert m.shape == (3, 5000)
        assert not np.all(m, axis=0).any()                      # every sample keeps a view
        assert abs(m.mean() - (rate - rate ** 3 / 3)) < 0.02     # restored views lower the rate
    assert generate_missing_matrix(10, 2, 0.0, 1).sum() == 0


def test_configurations():
    confs = configurations(["a"])
    assert len(confs) == 37
    assert ("a", "late", "drop") in confs and ("a", "mid", "drop") not in confs
    assert ("a", "mid", "cpm+mask") in confs and ("a", "late", "drop+mask") not in confs


def test_mask_input_changes_only_masked_models():
    from imvbench.models import FusionNet
    torch.manual_seed(0)
    x = [torch.randn(4, 5), torch.randn(4, 3)]
    m0 = torch.zeros(4, 2, dtype=torch.bool)
    m1 = m0.clone(); m1[:, 1] = True
    for fusion in ("early", "mid", "late"):
        net = FusionNet({0: 5, 1: 3}, 3, 8, fusion, use_mask=True)
        assert not torch.allclose(net(x, m0), net(x, m1))         # the mask reaches the output
        plain = FusionNet({0: 5, 1: 3}, 3, 8, fusion)
        assert torch.allclose(plain(x, m0), plain(x, m1))          # without +mask it is ignored


def test_mean_and_knn_fill():
    from imvbench.imputers.methods import knn_fill, mean_fill, view_means
    torch.manual_seed(0)
    a = torch.nn.functional.normalize(torch.randn(6, 4), dim=1)
    b = torch.nn.functional.normalize(torch.randn(6, 3), dim=1)
    tr_mask = torch.tensor([[False, False]] * 5 + [[False, True]])      # last train row misses view 1
    train = {"views": [a, b], "mask": tr_mask}
    means = view_means(train)
    assert torch.allclose(means[1], b[:5].mean(dim=0))                  # only observed rows count
    # a query identical to train row 2 on view 0 gets row 2's view 1 with k = 1
    q = [a[2:3].clone(), torch.zeros(1, 3)]
    qm = torch.tensor([[False, True]])
    out = knn_fill(q, qm, train, means, "cpu", k=1)
    assert torch.allclose(out[1][0], b[2]) and torch.allclose(out[0][0], a[2])
    out3 = knn_fill(q, qm, train, means, "cpu", k=3)
    d = ((a[:5] - a[2]) ** 2).sum(dim=1)
    assert torch.allclose(out3[1][0], b[:5][d.topk(3, largest=False).indices].mean(dim=0))
    mf = mean_fill(q, qm, means)
    assert torch.allclose(mf[1][0], means[1])


def test_late_drop_ignores_missing_views():
    from imvbench.models import FusionNet
    torch.manual_seed(0)
    net = FusionNet({0: 4, 1: 3}, 5, 8, "late", drop_missing=True)
    x0, x1 = torch.randn(2, 4), torch.randn(2, 3)
    mask = torch.tensor([[False, True], [False, False]])
    out = net([x0, x1], mask)
    out_garbage = net([x0, x1 * 100], mask)
    assert torch.allclose(out[0], out_garbage[0])                # view 1 missing -> not used
    assert torch.allclose(out[0], net.branches[0](x0)[0])
    assert torch.allclose(out[1], (net.branches[0](x0)[1] + net.branches[1](x1)[1]) / 2)


def test_gaus_fill_matches_original():
    from imvbench.imputers.methods import gaus_fill
    views = [torch.zeros(5, 4), torch.zeros(5, 3)]
    mask = torch.tensor([[True, False], [False, True], [False, False], [True, False], [False, True]])
    out = gaus_fill(views, mask)
    for idx in range(5):                                       # reference: core/data.py GausImputeWrapper
        rng = torch.Generator()
        rng.manual_seed(idx + 99999)
        for v in range(2):
            if mask[idx, v]:
                ref = torch.nn.functional.normalize(torch.randn(views[v].shape[1], generator=rng), p=2, dim=0)
                assert torch.allclose(out[v][idx], ref)
            else:
                assert torch.all(out[v][idx] == 0)


def test_split_matches_random_split():
    from torch.utils.data import random_split
    from imvbench.data import train_val_indices
    tr, va = train_val_indices(100, 7)
    ref_tr, ref_va = random_split(list(range(100)), [85, 15], generator=torch.Generator().manual_seed(7))
    assert tr.tolist() == list(ref_tr.indices) and va.tolist() == list(ref_va.indices)


def test_build_data_all_imputers():
    from imvbench.data import build_data
    for imp in ("zero", "mean", "gaus", "knn", "cpm", "dcp", "drop", "knn+mask"):
        d = build_data("synthetic", imp, 0.5, 7)
        assert len(d.train) + len(d.val) == 400 and len(d.test) == 150
        for s in (d.train, d.val, d.test):
            assert s.mask.shape == (len(s), 2) and not s.mask.all(dim=1).any()
            for v, x in enumerate(s.views):
                assert torch.isfinite(x).all()
                if imp in ("zero", "drop"):
                    assert torch.all(x[s.mask[:, v]] == 0)
                else:
                    assert torch.all(x[s.mask[:, v]].norm(dim=1) > 0)
    # cached imputer weights are reused
    names = sorted(os.listdir(os.environ["IMVB_WEIGHTS_DIR"]))
    assert any(n.startswith("cpm_weights_synthetic_missing0.5_seed7_val") for n in names)
    assert any(n.startswith("dcp_weights_synthetic_missing0.5_seed7") for n in names)


def test_training_is_deterministic():
    from imvbench.runner import Runner
    r = Runner(epochs=3, batch_size=32)
    kw = dict(dataset="synthetic", fusion="mid", imputer="zero", rate=0.3, seed=11,
              hidden=16, lr=3e-3, weight_decay=1e-4, test=True)
    a, b = r.run(**kw), r.run(**kw)
    assert a == b
    assert a["test"]["acc"] > 50                                # well above chance (33 %)


def test_tune_evaluate_report():
    from imvbench.cli import main
    common = ["--datasets", "synthetic", "--trials", "3", "--epochs", "2",
              "--tune-seeds", "7", "11", "--eval-seeds", "50", "121", "--eval-rates", "0.3", "0.7"]
    main(["run", *common, "--fusions", "early", "late", "--imputers", "zero", "cpm", "drop", "knn+mask", "mean"])
    main(["run", *common, "--task-index", "5", "--workers", "2"])          # mid / gaus, parallel
    main(["status", "--datasets", "synthetic", "--trials", "3",
          "--eval-seeds", "50", "121", "--eval-rates", "0.3", "0.7"])
    main(["report", "--datasets", "synthetic"])
    report = open(os.path.join(os.environ["IMVB_RESULTS_DIR"], "report.md")).read()
    assert "late/drop" in report and "## synthetic" in report and "late/knn+mask" in report
    # resuming does not redo finished work
    main(["run", *common, "--fusions", "late", "--imputers", "drop"])


def test_prepare_and_load_mosei():
    """Fake MulT-style pickle -> scripts/prepare_mosei.py -> the registered 'mosei' dataset."""
    import pickle
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import prepare_mosei
    rng = np.random.default_rng(0)

    def split(n):
        T = 50
        d = {"text": rng.normal(size=(n, T, 300)), "audio": rng.normal(size=(n, T, 74)),
             "vision": rng.normal(size=(n, T, 35)), "labels": rng.uniform(-3, 3, size=(n, 1, 1))}
        for i in range(n):                                   # pre-padding with zeros
            pad = rng.integers(0, 40)
            for m in ("text", "audio", "vision"):
                d[m][i, :pad] = 0
        d["audio"][0, -1, 0] = -np.inf
        return d
    pkl = os.path.join(_TMP, "mosei_senti_data.pkl")
    with open(pkl, "wb") as fh:
        pickle.dump({"train": split(120), "valid": split(30), "test": split(50)}, fh)
    prepare_mosei.main(pkl, os.path.join(os.environ["IMVB_DATA_DIR"], "mosei"))
    z = np.load(os.path.join(os.environ["IMVB_DATA_DIR"], "mosei", "mosei_pooled.npz"))
    assert z["text_train"].shape == (150, 600) and z["vision_test"].shape == (50, 70)
    assert np.isfinite(z["audio_train"]).all()

    from imvbench.data import build_data
    d = build_data("mosei", "knn", 0.5, 7)
    assert [x.shape[1] for x in d.train.views] == [600, 148, 70]
    assert len(d.train) + len(d.val) == 150 and len(d.test) == 50
    assert set(d.test.labels.tolist()) <= {0, 1}


def test_time_imputers():
    import csv
    import subprocess
    out = os.path.join(_TMP, "timing.csv")
    env = dict(os.environ, PYTHONPATH=os.path.dirname(os.path.abspath(__file__)))
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "time_imputers.py"),
                    "--datasets", "synthetic", "--out", out, "--hidden", "16"], check=True, env=env)
    rows = list(csv.DictReader(open(out)))
    assert {r["method"] for r in rows} == {"zero", "mean", "gaus", "knn", "cpm", "dcp", "classifier"}
    assert all(float(r["seconds"]) >= 0 for r in rows)


def test_handwritten_loader_six_views():
    """Fake mfeat files -> the registered 'handwritten' dataset with all imputers (6 views)."""
    from dataclasses import replace
    from imvbench.data import build_data
    from imvbench.datasets import MFEAT_VIEWS, get_dataset, register_dataset
    register_dataset(replace(get_dataset("handwritten"), cpm_epochs=2, dcp_epochs=2))   # fast test
    rng = np.random.default_rng(0)
    folder = os.path.join(os.environ["IMVB_DATA_DIR"], "handwritten", "mfeat")
    os.makedirs(folder, exist_ok=True)
    labels = np.repeat(np.arange(10), 200)
    for name, dim in MFEAT_VIEWS:
        x = rng.normal(size=(2000, dim)) + labels[:, None] * 0.3 + rng.uniform(0, 100, size=dim)
        np.savetxt(os.path.join(folder, f"mfeat-{name}"), x, fmt="%.5f")
    for imp in ("knn", "cpm", "dcp"):
        d = build_data("handwritten", imp, 0.7, 7)
        assert len(d.train) + len(d.val) == 1600 and len(d.test) == 400
        assert [x.shape[1] for x in d.test.views] == [76, 216, 64, 240, 47, 6]
        assert not d.train.mask.all(dim=1).any()
        assert sorted(np.bincount(d.test.labels.numpy()).tolist()) == [40] * 10      # stratified


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
