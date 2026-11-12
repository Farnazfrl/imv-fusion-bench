"""Final evaluation with the tuned hyperparameters.

Every configuration is retrained with its own best hyperparameters for each evaluation seed
(50, 121, 893, 17, 2024) and missing rate (0.1, 0.3, 0.5, 0.7); the restored best-validation
model is then tested once. One JSON line per run goes to
results/eval/<dataset>/<fusion>_<imputer>.jsonl. Finished runs are skipped on a restart; if the
study was re-tuned since, the old file is renamed *.stale and the runs are redone.
"""

import json
import os

from imvbench import config
from imvbench.tune import best_params

EVAL_SEEDS = [50, 121, 893, 17, 2024]
EVAL_RATES = [0.1, 0.3, 0.5, 0.7]
HP_KEYS = ("hidden", "lr", "weight_decay")


def eval_path(dataset, fusion, imputer):
    return os.path.join(config.RESULTS_DIR, "eval", dataset, f"{fusion}_{imputer}.jsonl")


def read_runs(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def evaluate(runner, dataset, fusion, imputer, rates=EVAL_RATES, seeds=EVAL_SEEDS):
    params = best_params(dataset, fusion, imputer)
    if params is None:
        raise SystemExit(f"No finished study for {dataset}/{fusion}/{imputer}: run `tune` first.")
    hp = {k: params[k] for k in HP_KEYS}
    path = eval_path(dataset, fusion, imputer)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    runs = read_runs(path)
    if any(any(r[k] != hp[k] for k in HP_KEYS) for r in runs):
        os.replace(path, path + ".stale")
        print(f"{path}: hyperparameters changed since the last evaluation -> moved to .stale", flush=True)
        runs = []
    done = {(r["rate"], r["seed"]) for r in runs}

    for rate in rates:
        todo = [s for s in seeds if (rate, s) not in done]
        if not todo:
            continue
        jobs = [dict(dataset=dataset, fusion=fusion, imputer=imputer, rate=rate, seed=s, test=True, **hp)
                for s in todo]
        for seed, res in zip(todo, runner.map(jobs)):
            row = dict(dataset=dataset, fusion=fusion, imputer=imputer, rate=rate, seed=seed, **hp,
                       val_acc=res["val_acc"], best_epoch=res["best_epoch"], **res["test"])
            with open(path, "a") as fh:
                fh.write(json.dumps(row) + "\n")
        accs = [r["acc"] for r in read_runs(path) if r["rate"] == rate]
        print(f"[{dataset}/{fusion}/{imputer}] p={rate:g}: test {sum(accs) / len(accs):.2f}% "
              f"({len(accs)} seeds)", flush=True)
    return read_runs(path)
