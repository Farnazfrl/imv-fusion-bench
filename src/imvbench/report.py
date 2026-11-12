"""Tables and paired tests from the evaluation results.

Writes results/report.md (also printed) and results/cells.csv. All accuracies are test
accuracies in %, mean +- std over the evaluation seeds. Paired tests pair runs that share the
dataset, missing rate and seed (hence the same split and missing matrix): two-sided Wilcoxon
signed-rank tests, Holm-corrected within each family.
"""

import glob
import io
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from imvbench import ALL_IMPUTERS, FUSIONS, IMPUTERS, MASK_SUFFIX, config, imputers_for
from imvbench.tune import best_params

IMP_ORDER = ALL_IMPUTERS
CONFIG_ORDER = [(f, i) for f in FUSIONS for i in imputers_for(f)]


def load_runs(datasets=None) -> pd.DataFrame:
    rows = []
    for path in sorted(glob.glob(os.path.join(config.RESULTS_DIR, "eval", "*", "*.jsonl"))):
        with open(path) as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit(f"No evaluation results under {config.RESULTS_DIR}/eval")
    if datasets:
        df = df[df.dataset.isin(datasets)]
    return df


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    idx = np.argsort(p[ok])
    m = ok.sum()
    adj = np.minimum(1.0, np.maximum.accumulate((m - np.arange(m)) * p[ok][idx]))
    tmp = np.empty(m)
    tmp[idx] = adj
    out[ok] = tmp
    return out


def paired(df, a, b, by=("dataset", "rate", "seed")):
    """Differences a - b over the units both have; a and b are (fusion, imputer) pairs."""
    by = list(by)
    sa = df[(df.fusion == a[0]) & (df.imputer == a[1])].set_index(by)["acc"]
    sb = df[(df.fusion == b[0]) & (df.imputer == b[1])].set_index(by)["acc"]
    d = (sa - sb).dropna()
    if len(d) == 0:
        return dict(n=0, diff=np.nan, wins=0, losses=0, p=np.nan)
    try:
        p = wilcoxon(d.values).pvalue if np.any(d.values != 0) else 1.0
    except ValueError:
        p = np.nan
    return dict(n=len(d), diff=d.mean(), wins=int((d > 0).sum()), losses=int((d < 0).sum()), p=p)


def fmt(m, s):
    return "" if pd.isna(m) else f"{m:.2f} ± {s:.2f}"


def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r.values) + " |")
    return "\n".join(lines)


def build_report(datasets=None) -> str:
    df = load_runs(datasets)
    out = io.StringIO()
    w = lambda s="": out.write(s + "\n")     # noqa: E731
    dsets = [d for d in dict.fromkeys(df.dataset)]

    cells = (df.groupby(["dataset", "fusion", "imputer", "rate"])
               .agg(acc=("acc", "mean"), std=("acc", "std"), n=("acc", "size"),
                    acc_complete=("acc_complete", "mean"), acc_incomplete=("acc_incomplete", "mean"))
               .reset_index())
    cells.to_csv(os.path.join(config.RESULTS_DIR, "cells.csv"), index=False)

    w("# Fusion and imputation benchmark: test accuracy (%)")
    w()
    w("Mean ± std over the evaluation seeds. Every (dataset, fusion, imputer) has its own tuned hyperparameters.")
    w("`drop` = late fusion that averages only the decisions of the observed views; "
      "`+mask` = the missing-view indicator is also given to the network.")

    # ---- 0. overview: imputer rows x fusion columns, mean over datasets and rates -----------
    w("\n## Overview (mean over datasets and missing rates)\n")
    ov = cells.groupby(["fusion", "imputer"]).acc.mean()
    rows = []
    for imp in IMP_ORDER:
        row = {"imputer": imp}
        for fu in FUSIONS:
            row[fu] = f"{ov[(fu, imp)]:.2f}" if (fu, imp) in ov.index else ""
        if any(row[fu] for fu in FUSIONS):
            rows.append(row)
    w(md_table(pd.DataFrame(rows)))
    w(f"\n(datasets: {', '.join(dsets)})")

    # ---- 1. per dataset: imputer x rate rows, fusion columns ---------------------------
    for ds in dsets:
        sub = cells[cells.dataset == ds]
        rows = []
        for imp in IMP_ORDER:
            for rate in sorted(sub.rate.unique()):
                row = {"imputer": imp, "p": f"{rate:g}"}
                for fu in FUSIONS:
                    c = sub[(sub.fusion == fu) & (sub.imputer == imp) & (sub.rate == rate)]
                    row[fu] = fmt(c.acc.iloc[0], c["std"].iloc[0]) if len(c) else ""
                if any(row[fu] for fu in FUSIONS):
                    rows.append(row)
        w(f"\n## {ds}\n")
        w(md_table(pd.DataFrame(rows)))

    # ---- 2. averaged over datasets -----------------------------------------------------
    w("\n## Average over datasets\n")
    full = cells.groupby(["fusion", "imputer", "rate"]).filter(lambda g: g.dataset.nunique() == len(dsets))
    avg = full.groupby(["fusion", "imputer", "rate"]).acc.mean().reset_index()
    rows = []
    for imp in IMP_ORDER:
        for rate in sorted(avg.rate.unique()):
            row = {"imputer": imp, "p": f"{rate:g}"}
            for fu in FUSIONS:
                c = avg[(avg.fusion == fu) & (avg.imputer == imp) & (avg.rate == rate)]
                row[fu] = f"{c.acc.iloc[0]:.2f}" if len(c) else ""
            if any(row[fu] for fu in FUSIONS):
                rows.append(row)
    w(md_table(pd.DataFrame(rows)))

    # ---- 3. average rank of each configuration over (dataset, rate) cells ------------------
    w("\n## Average rank (1 = best) over dataset × missing-rate cells\n")
    piv = cells.pivot_table(index=["dataset", "rate"], columns=["fusion", "imputer"], values="acc").dropna(axis=1)
    ranks = piv.rank(axis=1, ascending=False).mean().sort_values()
    mean_acc = piv.mean()
    rows = [{"fusion": f, "imputer": i, "avg rank": f"{r:.2f}", "mean acc": f"{mean_acc[(f, i)]:.2f}"}
            for (f, i), r in ranks.items()]
    w(md_table(pd.DataFrame(rows)))
    w(f"\n({len(piv)} cells)")

    # ---- 4. complete vs incomplete samples -------------------------------------------------
    w("\n## Complete vs incomplete test samples (mean over datasets, rates and seeds)\n")
    ci = (df.groupby(["fusion", "imputer"])[["acc", "acc_complete", "acc_incomplete"]].mean().reset_index())
    ci["order"] = [CONFIG_ORDER.index((f, i)) if (f, i) in CONFIG_ORDER else 99 for f, i in zip(ci.fusion, ci.imputer)]
    ci = ci.sort_values("order").drop(columns="order")
    for c in ("acc", "acc_complete", "acc_incomplete"):
        ci[c] = ci[c].map(lambda v: f"{v:.2f}")
    w(md_table(ci.rename(columns={"acc": "all", "acc_complete": "complete", "acc_incomplete": "incomplete"})))

    # ---- 5. paired comparisons -------------------------------------------------------------
    w("\n## Paired comparisons (A − B, pairs = dataset × rate × seed; Wilcoxon, Holm within each family)\n")
    M = MASK_SUFFIX
    families = {
        "Mask indicator (imputer+mask vs imputer), per fusion":
            [((f, i + M), (f, i)) for f in FUSIONS for i in IMPUTERS],
        "Imputer vs zero, per fusion": [((f, i), (f, "zero")) for f in FUSIONS for i in IMPUTERS if i != "zero"],
        "Imputer+mask vs zero+mask, per fusion":
            [((f, i + M), (f, "zero" + M)) for f in FUSIONS for i in IMPUTERS if i != "zero"],
        "Fusion, per imputer": [((b, i), (a, i)) for i in ALL_IMPUTERS if i != "drop"
                                for a, b in (("early", "mid"), ("early", "late"), ("mid", "late"))],
        "Late drop vs late with imputation": [(("late", "drop"), ("late", i)) for i in ALL_IMPUTERS if i != "drop"],
    }
    for fam, pairs in families.items():
        res = [dict(A=f"{a[0]}/{a[1]}", B=f"{b[0]}/{b[1]}", **paired(df, a, b)) for a, b in pairs]
        res = [r for r in res if r["n"] > 0]
        if not res:
            continue
        adj = holm([r["p"] for r in res])
        rows = [{"A": r["A"], "B": r["B"], "n": r["n"], "A − B": f"{r['diff']:+.2f}",
                 "A wins/losses": f"{r['wins']}/{r['losses']}",
                 "p (Holm)": f"{p:.3g}" + (" *" if p < 0.05 else "")} for r, p in zip(res, adj)]
        w(f"\n### {fam}\n")
        w(md_table(pd.DataFrame(rows)))

    # ---- 6. tuned hyperparameters ----------------------------------------------------------
    w("\n## Tuned hyperparameters\n")
    rows = []
    for ds in dsets:
        for fu, imp in CONFIG_ORDER:
            bp = best_params(ds, fu, imp)
            if bp:
                rows.append({"dataset": ds, "fusion": fu, "imputer": imp, "h": bp["hidden"],
                             "lr": f"{bp['lr']:.2e}", "weight decay": f"{bp['weight_decay']:.2e}",
                             "val acc": f"{bp['val_acc']:.2f}", "trials": bp["trials"]})
    if rows:
        w(md_table(pd.DataFrame(rows)))

    text = out.getvalue()
    with open(os.path.join(config.RESULTS_DIR, "report.md"), "w") as fh:
        fh.write(text)
    return text
