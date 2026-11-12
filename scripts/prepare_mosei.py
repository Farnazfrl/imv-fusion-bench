"""Turn the CMU-MOSEI pickle into one feature vector per modality and clip.

Input : mosei_senti_data.pkl (word-aligned CMU-MOSEI as distributed with the Multimodal
        Transformer, MulT; Tsai et al., ACL 2019). Keys: train / valid / test, each with
        text (N, T, 300 GloVe), audio (N, T, 74 COVAREP), vision (N, T, 35 Facet),
        labels (N, 1, 1) = sentiment in [-3, 3].
Output: <out_dir>/mosei_pooled.npz with, per modality, the mean and the standard deviation over
        the clip's real (non-padding) time steps: text 600, audio 148, vision 70 features.
        The official train and valid splits are merged into "train" (the benchmark takes its
        own validation split from it); "test" is the official test split.

    python scripts/prepare_mosei.py /path/to/mosei_senti_data.pkl data/mosei
"""

import os
import pickle
import sys

import numpy as np

MODALITIES = ["text", "audio", "vision"]


def load(path):
    with open(path, "rb") as fh:
        try:
            return pickle.load(fh)
        except UnicodeDecodeError:
            fh.seek(0)
            return pickle.load(fh, encoding="latin1")


def pool(split):
    seqs = {}
    for m in MODALITIES:
        x = np.asarray(split[m], dtype=np.float32)
        x[~np.isfinite(x)] = 0.0                    # COVAREP contains -inf / nan
        seqs[m] = x
    # a time step is real if any modality is non-zero there (padding is all zeros)
    real = np.zeros(seqs["text"].shape[:2], dtype=bool)
    for x in seqs.values():
        real |= np.abs(x).sum(axis=2) > 0
    real[real.sum(axis=1) == 0] = True              # safety: empty clip -> use every step
    w = real[:, :, None].astype(np.float32)
    n = w.sum(axis=1)
    out = {}
    for m, x in seqs.items():
        mean = (x * w).sum(axis=1) / n
        var = (((x - mean[:, None, :]) ** 2) * w).sum(axis=1) / n
        out[m] = np.concatenate([mean, np.sqrt(var)], axis=1).astype(np.float32)
    out["y"] = np.asarray(split["labels"], dtype=np.float32).reshape(len(real), -1)[:, 0]
    out["steps"] = real.sum(axis=1)
    return out


def main(src, dst):
    data = load(src)
    print("splits:", list(data.keys()))
    valid_key = "valid" if "valid" in data else "dev"
    for k in ("train", valid_key, "test"):
        print(f"  {k}: " + ", ".join(f"{m} {np.shape(data[k][m])}" for m in MODALITIES + ["labels"]))
    parts = {k: pool(data[k]) for k in ("train", valid_key, "test")}
    out = {}
    for m in MODALITIES + ["y", "steps"]:
        out[f"{m}_train"] = np.concatenate([parts["train"][m], parts[valid_key][m]])
        out[f"{m}_test"] = parts["test"][m]
    os.makedirs(dst, exist_ok=True)
    path = os.path.join(dst, "mosei_pooled.npz")
    np.savez_compressed(path, **out)
    for s in ("train", "test"):
        y = out[f"y_{s}"]
        print(f"{s}: {len(y)} clips, text {out['text_' + s].shape[1]}, audio {out['audio_' + s].shape[1]}, "
              f"vision {out['vision_' + s].shape[1]} features; non-negative sentiment {100 * (y >= 0).mean():.1f}%; "
              f"median {int(np.median(out['steps_' + s]))} time steps")
    print("written", path)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
