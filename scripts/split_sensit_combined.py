"""Create the SensIT Vehicle view files from LIBSVM's 'combined' files.

LIBSVM's combined / combined.t hold 100 features per sample: 50 acoustic followed by 50 seismic.
This writes acoustic, seismic, acoustic.t and seismic.t (same sample on the same line).

    python scripts/split_sensit_combined.py <folder with combined, combined.t> <output folder>
"""

import os
import sys

from sklearn.datasets import dump_svmlight_file, load_svmlight_file


def main(src, dst):
    os.makedirs(dst, exist_ok=True)
    for suffix in ("", ".t"):
        X, y = load_svmlight_file(os.path.join(src, "combined" + suffix), n_features=100)
        dump_svmlight_file(X[:, :50], y, os.path.join(dst, "acoustic" + suffix), zero_based=False)
        dump_svmlight_file(X[:, 50:], y, os.path.join(dst, "seismic" + suffix), zero_based=False)
        print(f"combined{suffix}: {X.shape[0]} samples -> acoustic{suffix}, seismic{suffix}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
