"""imvbench: fusion and imputation for neural classifiers on incomplete multi-view data.

Compares fusion types (early, mid, late), imputers (zero, mean, Gaussian noise, cross-view kNN,
CPM-Nets, DCP, and dropping the missing views in late fusion), with or without a missing-view
indicator given to the network, and missing rates, with every configuration tuned independently.

Imputer names: "zero", "mean", ...; "<imputer>+mask" also feeds the missing-view mask m_n to the
network; "drop" (late fusion only) averages the decisions of the observed views only.
"""

FUSIONS = ["early", "mid", "late"]
IMPUTERS = ["zero", "mean", "gaus", "knn", "cpm", "dcp"]
MASK_SUFFIX = "+mask"
LATE_ONLY_IMPUTERS = ["drop"]          # late fusion: average only the observed views
ALL_IMPUTERS = [i + s for i in IMPUTERS for s in ("", MASK_SUFFIX)] + LATE_ONLY_IMPUTERS


def base_imputer(imputer: str) -> str:
    """'cpm+mask' -> 'cpm' (the imputation itself; the mask only changes the network input)."""
    return imputer[:-len(MASK_SUFFIX)] if imputer.endswith(MASK_SUFFIX) else imputer


def uses_mask(imputer: str) -> bool:
    return imputer.endswith(MASK_SUFFIX)


def imputers_for(fusion: str) -> list:
    """Imputers that apply to a fusion type ('drop' only makes sense for late fusion)."""
    return [i + s for i in IMPUTERS for s in ("", MASK_SUFFIX)] + \
        (LATE_ONLY_IMPUTERS if fusion == "late" else [])


def configurations(datasets) -> list:
    """All (dataset, fusion, imputer) combinations, in a fixed order (used for task indices)."""
    return [(d, f, i) for d in datasets for f in FUSIONS for i in imputers_for(f)]
