"""Device and folder settings. Every folder can be changed with an environment variable."""

import os
import warnings

import torch


def _select_device() -> torch.device:
    forced = os.environ.get("TORCH_DEVICE")          # e.g. TORCH_DEVICE=cpu
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        if major >= 7:
            return torch.device("cuda:0")
        warnings.warn(f"GPU with compute capability {major}.{minor} is not supported by this "
                      f"PyTorch build; using the CPU.")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = _select_device()


def available_cpus() -> int:
    """CPUs this process may use: IMVB_CPUS if set, else the CPU affinity of the process (which
    job schedulers restrict to the allocation on Linux), else all CPUs of the machine."""
    if os.environ.get("IMVB_CPUS"):
        return max(1, int(os.environ["IMVB_CPUS"]))
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:                      # macOS / Windows
        return os.cpu_count() or 1

# Raw feature files of the datasets (see README for the expected layout)
DATA_DIR = os.environ.get("IMVB_DATA_DIR", "data")
# Cached CPM-Nets / DCP models; same file names as the two-head repository, so its
# weights folder can be reused directly
WEIGHTS_DIR = os.environ.get("IMVB_WEIGHTS_DIR", "weights")
# Optuna studies, evaluation results and reports
RESULTS_DIR = os.environ.get("IMVB_RESULTS_DIR", "results")
