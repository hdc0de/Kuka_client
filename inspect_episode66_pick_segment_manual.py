#!/usr/bin/env python3
"""Inspect episode_66 and print the manually selected pick segment.

This helper does not modify the bridge. It is for scheme 1: inspect one
expert HDF5 file and copy the printed indices into the adaptation config.
"""

from __future__ import annotations

import argparse

import h5py
import numpy as np


DEFAULT_HDF5 = "/media/user/PS2000/ACTData/daizi/hdf5_qpos_10hz_binary/episode_66.hdf5"

# Manual selection for episode_66:
# qpos z begins the main descent around 64-66, gripper closes at 90,
# and the first lift after grasp starts around 96.
MANUAL_START_IDX = 64
MANUAL_END_IDX = 100


def print_window(name: str, arr: np.ndarray, start: int, end: int) -> None:
    z = arr[:, 2]
    g = arr[:, 7]
    print(f"\n{name} frames [{start}, {end})")
    for i in range(max(0, start - 5), min(len(arr), end + 5)):
        dz = z[i] - z[i - 1] if i > 0 else 0.0
        marker = ""
        if i == start:
            marker = "  <-- start_idx"
        elif i == end:
            marker = "  <-- end_idx"
        print(f"{i:03d}  z={z[i]:.4f}  dz={dz:+.4f}  gripper={g[i]:.0f}{marker}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", default=DEFAULT_HDF5)
    parser.add_argument("--start-idx", type=int, default=MANUAL_START_IDX)
    parser.add_argument("--end-idx", type=int, default=MANUAL_END_IDX)
    args = parser.parse_args()

    with h5py.File(args.hdf5, "r") as f:
        qpos = np.asarray(f["observations/qpos"], dtype=np.float32)
        action = np.asarray(f["action"], dtype=np.float32)

    print(f"HDF5: {args.hdf5}")
    print(f"qpos shape={qpos.shape}, action shape={action.shape}")

    for name, arr in (("qpos", qpos), ("action", action)):
        changes = np.where(np.diff(arr[:, 7]) != 0)[0] + 1
        print(f"{name} gripper changes: {changes.tolist()}")

    print_window("qpos", qpos, args.start_idx, args.end_idx)
    print_window("action", action, args.start_idx, args.end_idx)

    print("\nCopy these values into Kuka_client/config_original_act_qpos_10hz_binary_adaptation.yaml:")
    print("adaptation:")
    print(f'  expert_hdf5_path: "{args.hdf5}"')
    print(f"  expert_start_idx: {args.start_idx}")
    print(f"  expert_end_idx: {args.end_idx}")


if __name__ == "__main__":
    main()
