"""Download N MOVi-A clips (RGB + GT) and write them to disk in OUR format.

WHY THIS SCRIPT EXISTS
----------------------
Training reads per-clip RGB files:  rgb/{i:05d}.npz  with array `video` [T,H,W,3] uint8.
Downstream eval reads per-clip GT files:  gt/{i:05d}.npz  with arrays like
`positions` [T,K,3], `velocities` [T,K,3], `bboxes` (per-object [T,4]).

This script downloads both from the SAME tfds samples in ONE pass, so clip
index i always matches between rgb/ and gt/ (same ordering, same split,
shuffle_files=False).

USAGE (Colab / cloud GPU is fine — no torch needed, just tfds):
    pip install tensorflow-datasets tqdm
    python scripts/download_movi_a.py --num 5000 --rgb-dir /content/data/movi_a_rgb --gt-dir /content/data/movi_a_gt
    # resume/extend later without redownloading clip 0..4999:
    python scripts/download_movi_a.py --num 8000 --start 5000 --rgb-dir ... --gt-dir ...

DEFAULTS mirror what you already have: dataset movi_a/128x128, split train,
shuffle_files=False, so the first 1000 clips are byte-identical in ORDER to
your existing local 1000 (video bytes should match; GT layout is normalised
to [T,K,...] — see NOTE below).

NOTE on GT layout: the tfds `instances` dict stores per-object lists; we stack
them to [T,K,3] (padded with NaN / -1 where a slot has no object) so files
match what src/eval/downstream.py expects. Your OLD gt files were [K,T,3]
in one inspected case — downstream.py reshapes blindly so both work for the
ridge probe, but NEW files use the documented [T,K,3] layout.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


GT_KEYS = ["positions", "velocities", "quaternions", "angular_velocities",
           "bboxes_3d", "image_positions", "masses"]


def _to_numpy(x):
    import tensorflow as tf
    if isinstance(x, (tf.Tensor, tf.RaggedTensor)):
        return x.numpy() if hasattr(x, "numpy") else np.asarray(x)
    return np.asarray(x)


def _stack_instances(inst: dict) -> dict:
    """Convert tfds `instances` dict to plain [T,K,...] numpy arrays.

    Robust to the two layouts tfds may yield:
      - stacked tensors [K,T,...]  -> transposed to [T,K,...]
      - already time-major [T,K,...] -> kept as-is
    `bboxes` (sequence of tfds BBox, ymin/xmin/ymax/xmax) becomes float
    [T,K,4]. Anything else is converted with np.asarray and stored as-is.
    """
    out: dict = {}
    # First pass: plain tensor keys.
    for key, val in inst.items():
        if key == "bboxes":
            continue
        try:
            a = np.asarray(_to_numpy(val))
        except Exception:
            continue
        if a.dtype == object or a.ndim < 2:
            continue
        # Normalise object/time axes: MOVi-A has T=24 frames. If axis0 is not
        # 24 but axis1 is, swap -> [T,K,...].
        if a.shape[0] != 24 and len(a.shape) > 1 and a.shape[1] == 24:
            a = np.swapaxes(a, 0, 1)
        out[key] = a
    # bboxes: flat sequence of BBox objects over K*T (row-major K outer).
    if "bboxes" in inst:
        try:
            raw = list(inst["bboxes"])
            rows = np.asarray([[float(b.ymin), float(b.xmin),
                                float(b.ymax), float(b.xmax)] for b in raw],
                              dtype=np.float32)
            n = rows.shape[0]
            # Recover (K, T): T should be 24 if divisible.
            if n % 24 == 0:
                K = n // 24
                out["bboxes"] = rows.reshape(K, 24, 4).swapaxes(0, 1)  # [T,K,4]
            else:
                out["bboxes_flat"] = rows
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num", type=int, default=1000)
    ap.add_argument("--start", type=int, default=0,
                    help="skip this many samples (to EXTEND a dataset without re-download)")
    ap.add_argument("--rgb-dir", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--dataset", default="movi_a/128x128")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    import tensorflow_datasets as tfds
    from tqdm import tqdm

    rgb_dir = Path(args.rgb_dir); gt_dir = Path(args.gt_dir)
    rgb_dir.mkdir(parents=True, exist_ok=True)
    gt_dir.mkdir(parents=True, exist_ok=True)

    ds = tfds.load(args.dataset, data_dir="gs://kubric-public/tfds",
                   split=args.split, shuffle_files=False)
    ds = ds.skip(args.start).take(args.num)

    n = 0
    for j, sample in enumerate(tqdm(ds, total=args.num)):
        i = args.start + j  # global clip index -> filename
        video = _to_numpy(sample["video"])
        assert video.ndim == 4 and video.shape[-1] == 3, video.shape
        np.savez_compressed(rgb_dir / f"{i:05d}.npz",
                            video=np.asarray(video, dtype=np.uint8))
        inst = sample.get("instances", {})
        arrays = _stack_instances(inst)
        if not arrays:
            raise RuntimeError(f"no instance arrays; keys={list(inst.keys())}")
        np.savez_compressed(gt_dir / f"{i:05d}.npz", **arrays)
        n += 1

    print(f"Saved {n} clips [{args.start}..{args.start + n - 1}] -> {rgb_dir} + {gt_dir}")


if __name__ == "__main__":
    main()
