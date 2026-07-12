#!/usr/bin/env python3
"""Find a pick descend/grasp segment from ACT HDF5 data.

Scheme 2:
  1. Find the first gripper close frame, 0 -> 1.
  2. Before that close frame, find the start of the main continuous z descent.
  3. If --current-z is provided, search backward from the close frame to the
     descent start and choose the frame whose z is closest to the live robot z,
     similar to adaption_full.py's z-alignment step.
  4. Set end_idx to a few frames after the close frame.

The printed [start_idx, end_idx) can be copied into the adaptation config.
"""

from __future__ import annotations

import argparse

import h5py
import numpy as np


DEFAULT_HDF5 = "/media/user/PS2000/ACTData/daizi/hdf5_qpos_10hz_binary/episode_66.hdf5"


def first_close_idx(gripper: np.ndarray) -> int:
    candidates = np.where((gripper[:-1] < 0.5) & (gripper[1:] >= 0.5))[0] + 1
    if len(candidates) == 0:
        raise ValueError("No gripper close transition 0 -> 1 was found")
    return int(candidates[0])


def find_main_descent_start(
    z: np.ndarray,
    close_idx: int,
    window: int,
    min_drop: float,
    min_negative_steps: int,
) -> int:
    """Return the high point immediately before the main z descent."""
    best_score = -np.inf
    best_start = 0

    for i in range(0, max(1, close_idx - window)):
        j = min(close_idx, i + window)
        dz = np.diff(z[i : j + 1])
        drop = float(z[i] - z[j])
        negative_steps = int(np.sum(dz < -1e-3))
        upward_noise = float(np.sum(np.maximum(dz, 0.0)))

        if drop < min_drop or negative_steps < min_negative_steps:
            continue

        score = drop - 2.0 * upward_noise
        if score > best_score:
            best_score = score
            best_start = i

    if not np.isfinite(best_score):
        raise ValueError(
            "Could not find a continuous z descent before close_idx. "
            "Try lowering --min-drop or --min-negative-steps."
        )

    # Move to the local high point just before the descent run.
    search_hi = min(close_idx, best_start + window + 1)
    return int(best_start + np.argmax(z[best_start:search_hi]))


def choose_start_by_current_z(
    z: np.ndarray,
    descent_start: int,
    close_idx: int,
    current_z: float | None,
) -> int:
    if current_z is None:
        return descent_start

    lo = max(0, descent_start)
    hi = min(len(z), close_idx + 1)
    local = np.abs(z[lo:hi] - float(current_z))
    return int(lo + np.argmin(local))


def print_window(arr: np.ndarray, start: int, close_idx: int, end: int) -> None:
    z = arr[:, 2]
    g = arr[:, 7]
    lo = max(0, start - 6)
    hi = min(len(arr), end + 6)
    for i in range(lo, hi):
        dz = z[i] - z[i - 1] if i > 0 else 0.0
        marker = ""
        if i == start:
            marker = "  <-- start_idx"
        elif i == close_idx:
            marker = "  <-- close_idx"
        elif i == end:
            marker = "  <-- end_idx"
        print(f"{i:03d}  z={z[i]:.4f}  dz={dz:+.4f}  gripper={g[i]:.0f}{marker}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5", default=DEFAULT_HDF5)
    parser.add_argument(
        "--source",
        choices=["qpos", "action"],
        default="qpos",
        help="Which trajectory to analyze. Use qpos for expert observations.",
    )
    parser.add_argument("--current-z", type=float, default=None)
    parser.add_argument("--window", type=int, default=15)
    parser.add_argument("--min-drop", type=float, default=0.04)
    parser.add_argument("--min-negative-steps", type=int, default=5)
    parser.add_argument("--post-close-frames", type=int, default=10)
    args = parser.parse_args()

    with h5py.File(args.hdf5, "r") as f:
        arr = np.asarray(
            f["observations/qpos"] if args.source == "qpos" else f["action"],
            dtype=np.float32,
        )

    z = arr[:, 2]
    gripper = arr[:, 7]
    close_idx = first_close_idx(gripper)
    descent_start = find_main_descent_start(
        z,
        close_idx=close_idx,
        window=args.window,
        min_drop=args.min_drop,
        min_negative_steps=args.min_negative_steps,
    )
    start_idx = choose_start_by_current_z(
        z,
        descent_start=descent_start,
        close_idx=close_idx,
        current_z=args.current_z,
    )
    end_idx = min(len(arr), close_idx + args.post_close_frames)

    print(f"HDF5: {args.hdf5}")
    print(f"source: {args.source}")
    print(f"descent_start: {descent_start}")
    print(f"close_idx: {close_idx}")
    if args.current_z is not None:
        print(f"current_z: {args.current_z:.4f}")
        print(f"z-matched start_idx: {start_idx}  z={z[start_idx]:.4f}")
    print(f"end_idx: {end_idx}")
    print_window(arr, start_idx, close_idx, end_idx)

    print("\nCopy these values into Kuka_client/config_original_act_qpos_10hz_binary_adaptation.yaml:")
    print("adaptation:")
    print(f'  expert_hdf5_path: "{args.hdf5}"')
    print(f"  expert_start_idx: {start_idx}")
    print(f"  expert_end_idx: {end_idx}")


if __name__ == "__main__":
    main()
