"""Export MOVi-A ground-truth trajectories to per-clip .npz (for DOWNSTREAM eval only).

Run once, on a machine/Colab cell with tensorflow_datasets + gs access:
    python scripts/export_gt_trajectories.py --out C:/.../datasets/movi_a_gt --num 1000

Each output gt/{i:05d}.npz holds whatever of these arrays the tfds build exposes:
    positions    : [T, K, 3] float32
    velocities   : [T, K, 3] float32
    object_types : [T, K]    int32   (index/type per slot; -1 padded)
Clip order MUST match the RGB export (shuffle_files=False, same split/num).
These arrays are never touched during self-supervised training.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds
from tqdm import tqdm


CANDIDATE_KEYS = ["positions", "velocities", "object_types", "bboxes"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--num", type=int, default=1000)
    ap.add_argument("--dataset", default="movi_a/128x128")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = tfds.load(args.dataset, data_dir="gs://kubric-public/tfds",
                   split=args.split, shuffle_files=False)

    for i, sample in enumerate(tqdm(ds.take(args.num), total=args.num)):
        inst = sample.get("instances", {})
        arrays = {}
        for key in CANDIDATE_KEYS:
            if key in inst:
                arrays[key] = np.asarray(inst[key].numpy())
        if not arrays:
            raise RuntimeError(f"no instance arrays found; keys={list(inst.keys())}")
        np.savez_compressed(out_dir / f"{i:05d}.npz", **arrays)

    print(f"Saved GT trajectories for {args.num} clips -> {out_dir}")


if __name__ == "__main__":
    main()
