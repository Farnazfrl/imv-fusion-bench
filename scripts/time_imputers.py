"""Measure the cost of each imputer: fitting it on the training split and filling the missing
views of the train, validation and test splits (one dataset, missing rate and seed), plus the
training of one classifier for reference.

CPM-Nets and DCP are trained from scratch in a temporary folder (cached weights are not used).
Run it on the hardware you want to report:

    uv run python scripts/time_imputers.py --datasets har sensit modelnet mosei7 \
        --out results/timing.csv
"""

import argparse
import csv
import os
import sys
import tempfile
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from imvbench import config                                   # noqa: E402
from imvbench.data import ExperimentData, Split, prepare_splits  # noqa: E402
from imvbench.imputers import impute_splits                   # noqa: E402
from imvbench.train import run, set_seed                      # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--imputers", nargs="+", default=["zero", "mean", "gaus", "knn", "cpm", "dcp"])
    ap.add_argument("--rate", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--hidden", type=int, default=256, help="classifier width for the reference run")
    ap.add_argument("--out", default=os.path.join(config.RESULTS_DIR, "timing.csv"))
    a = ap.parse_args()

    threads = config.available_cpus()
    torch.set_num_threads(threads)
    device = config.DEVICE
    hardware = torch.cuda.get_device_name(0) if str(device).startswith("cuda") else f"cpu x{threads}"
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    new = not os.path.exists(a.out)
    fh = open(a.out, "a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(["dataset", "method", "seconds", "rate", "seed", "n_train", "hardware"])

    for ds in a.datasets:
        spec, splits = prepare_splits(ds, a.rate, a.seed)
        n_train = len(splits["train"]["labels"])
        with tempfile.TemporaryDirectory() as weights:          # no cached CPM-Nets / DCP weights
            for imp in a.imputers:
                set_seed(a.seed)
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                filled = impute_splits(spec, imp, splits, a.rate, a.seed, weights, device)
                if str(device).startswith("cuda"):
                    torch.cuda.synchronize()
                sec = time.perf_counter() - t0
                w.writerow([ds, imp, f"{sec:.3f}", a.rate, a.seed, n_train, hardware])
                fh.flush()
                print(f"[{ds}] {imp:5s} {sec:9.2f} s", flush=True)
                if imp == "zero":
                    data = ExperimentData(spec, "zero", a.rate, a.seed,
                                          *[Split(filled[k], splits[k]["labels"], splits[k]["mask"]).to(device)
                                            for k in ("train", "val", "test")])
                    t0 = time.perf_counter()
                    run(data, "early", "zero", a.hidden, 1e-3, 1e-4, device=device)
                    sec = time.perf_counter() - t0
                    w.writerow([ds, "classifier", f"{sec:.3f}", a.rate, a.seed, n_train, hardware])
                    fh.flush()
                    print(f"[{ds}] classifier (early, h={a.hidden}, 20 epochs) {sec:9.2f} s", flush=True)
    fh.close()
    print("written", a.out)


if __name__ == "__main__":
    main()
