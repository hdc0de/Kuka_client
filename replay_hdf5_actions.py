#!/usr/bin/env python3
"""Replay Cartesian actions from an ACT HDF5 episode on the real robot.

Run this on the ROS machine so it can publish directly to the iiwa Cartesian
command topic and the Robotiq gripper topic.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency on some machines
    yaml = None

from ros_bridge_original_act import (
    DEFAULT_WORKSPACE,
    GRIPPER_WIDTH_MAX,
    GRIPPER_WIDTH_MIN,
    HOME_POSE,
    DummyCartesianRobot,
    ROSCartesianRobot,
    normalized_to_width,
    width_to_normalized,
)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Replay /action from an HDF5 episode directly on the robot."
    )
    parser.add_argument(
        "--episode-path",
        type=Path,
        required=True,
        help="Path to episode_*.hdf5",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=script_dir / "config_original_act.yaml",
        help="Optional YAML config for home pose / workspace / gripper modes",
    )
    parser.add_argument(
        "--timing-mode",
        choices=("auto", "fixed", "timestamps"),
        default="auto",
        help="Replay at fixed --hz, or use timestamps stored in the HDF5 file",
    )
    parser.add_argument(
        "--timestamp-dataset",
        type=str,
        default=None,
        help="Optional HDF5 dataset path for action timestamps",
    )
    parser.add_argument("--hz", type=float, default=15.0, help="Replay rate")
    parser.add_argument("--start-step", type=int, default=0, help="First action index")
    parser.add_argument(
        "--end-step",
        type=int,
        default=None,
        help="Stop before this action index (default: end of episode)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Replay every Nth action",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional hard limit after slicing",
    )
    parser.add_argument(
        "--input-gripper-mode",
        choices=("binary", "width", "normalized"),
        default="binary",
        help="How action[7] is stored in the HDF5 file",
    )
    parser.add_argument(
        "--binary-threshold",
        type=float,
        default=0.5,
        help="Threshold for deciding whether binary gripper is closed",
    )
    parser.add_argument(
        "--wait-enter",
        action="store_true",
        help="Wait for Enter before moving",
    )
    parser.add_argument(
        "--skip-home",
        action="store_true",
        help="Do not move to home pose before replay",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=1.0,
        help="Pause after go_home before replay",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Use DummyCartesianRobot instead of ROS topics",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Print one progress line every N replayed actions",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    if not config_path or not config_path.exists():
        return {}
    if yaml is None:
        raise RuntimeError(
            "PyYAML is not installed, so --config cannot be used on this machine."
        )
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_replay_settings(cfg: dict) -> tuple[np.ndarray, np.ndarray, str, str]:
    robot_cfg = cfg.get("robot", {})
    bridge_cfg = cfg.get("ros_bridge_original_act", cfg.get("ros_bridge", {}))

    home_pose = np.asarray(
        robot_cfg.get("home_cartesian_pose", HOME_POSE),
        dtype=np.float32,
    )
    workspace = np.asarray(
        robot_cfg.get("workspace_xyz", DEFAULT_WORKSPACE),
        dtype=np.float32,
    )
    pose_frame_id = bridge_cfg.get("pose_frame_id", "iiwa_link_0")
    gripper_command_mode = bridge_cfg.get("gripper_command_mode", "width")
    return home_pose, workspace, pose_frame_id, gripper_command_mode


def _find_timestamp_dataset(root: h5py.File, explicit_path: str | None) -> str | None:
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    candidates.extend(
        [
            "/action_timestamps",
            "/timestamps/action",
            "/timestamps",
            "/observations/timestamps",
        ]
    )
    for path in candidates:
        if path in root:
            return path
    return None


def load_actions(
    episode_path: Path,
    start_step: int,
    end_step: int | None,
    stride: int,
    max_steps: int | None,
    timing_mode: str,
    timestamp_dataset: str | None,
) -> tuple[np.ndarray, np.ndarray | None, str]:
    if stride <= 0:
        raise ValueError("--stride must be >= 1")

    with h5py.File(episode_path, "r") as root:
        if "/action" not in root:
            raise KeyError(f"{episode_path} does not contain /action")
        action_ds = root["/action"]
        start = max(0, int(start_step))
        stop = action_ds.shape[0] if end_step is None else min(int(end_step), action_ds.shape[0])
        actions = action_ds[start:stop:stride]
        ts_path = _find_timestamp_dataset(root, timestamp_dataset)
        timestamps = None
        timing_source = "fixed_hz"
        if ts_path is not None:
            timestamps = np.asarray(root[ts_path][start:stop:stride], dtype=np.float64).reshape(-1)
            timing_source = ts_path

    if max_steps is not None:
        actions = actions[: max(0, int(max_steps))]
        if timestamps is not None:
            timestamps = timestamps[: max(0, int(max_steps))]
    if len(actions) == 0:
        raise ValueError("No actions selected. Check --start-step/--end-step/--stride.")
    if timestamps is not None and len(timestamps) != len(actions):
        raise ValueError(
            f"Timestamp length mismatch: {len(timestamps)} timestamps for {len(actions)} actions"
        )
    if timing_mode == "timestamps" and timestamps is None:
        raise ValueError(
            "Requested --timing-mode timestamps, but no timestamp dataset was found in the HDF5 file."
        )
    if timestamps is None:
        timing_source = "fixed_hz"
    return np.asarray(actions, dtype=np.float32), timestamps, timing_source


def compute_sleep_time(
    idx: int,
    timestamps: np.ndarray | None,
    fixed_dt: float,
) -> float:
    if timestamps is None:
        return fixed_dt
    if idx >= len(timestamps) - 1:
        return 0.0
    delta = float(timestamps[idx + 1] - timestamps[idx])
    if delta < 0:
        raise ValueError("Timestamps are not monotonic increasing.")
    return delta


def binary_closed_to_width(value: float, threshold: float) -> float:
    closed = float(value) >= threshold
    return GRIPPER_WIDTH_MIN if closed else GRIPPER_WIDTH_MAX


def convert_gripper(
    value: float,
    input_mode: str,
    output_mode: str,
    threshold: float,
) -> float:
    if input_mode == "binary":
        width = binary_closed_to_width(value, threshold)
    elif input_mode == "width":
        width = float(np.clip(value, GRIPPER_WIDTH_MIN, GRIPPER_WIDTH_MAX))
    elif input_mode == "normalized":
        width = normalized_to_width(float(value))
    else:  # pragma: no cover - argparse already guards this
        raise ValueError(f"Unsupported input_gripper_mode: {input_mode}")

    if output_mode == "width":
        return width
    if output_mode == "normalized":
        return width_to_normalized(width)
    raise ValueError(f"Unsupported gripper_command_mode: {output_mode}")


def prepare_action(
    raw_action: np.ndarray,
    input_gripper_mode: str,
    gripper_command_mode: str,
    binary_threshold: float,
) -> np.ndarray:
    action = np.asarray(raw_action, dtype=np.float32).reshape(-1)
    if action.shape[0] < 7:
        raise ValueError(f"Expected at least 7 pose values, got shape {action.shape}")

    out = np.zeros(8, dtype=np.float32)
    out[:7] = action[:7]
    if action.shape[0] > 7:
        out[7] = convert_gripper(
            action[7],
            input_mode=input_gripper_mode,
            output_mode=gripper_command_mode,
            threshold=binary_threshold,
        )
    else:
        if gripper_command_mode == "width":
            out[7] = GRIPPER_WIDTH_MAX
        else:
            out[7] = width_to_normalized(GRIPPER_WIDTH_MAX)
    return out


def build_robot(
    dummy: bool,
    home_pose: np.ndarray,
    workspace: np.ndarray,
    pose_frame_id: str,
    gripper_command_mode: str,
):
    if dummy:
        robot = DummyCartesianRobot(
            home_pose=home_pose,
            workspace=workspace,
            gripper_feedback_mode="width",
            gripper_command_mode=gripper_command_mode,
        )
        return robot, None

    import rospy

    rospy.init_node("replay_hdf5_actions", anonymous=True)
    robot = ROSCartesianRobot(
        home_pose=home_pose,
        workspace=workspace,
        frame_id=pose_frame_id,
        gripper_feedback_mode="width",
        gripper_command_mode=gripper_command_mode,
    )
    robot.wait_for_state()
    robot.activate_gripper()
    return robot, rospy


def main():
    args = parse_args()
    cfg = load_config(args.config)
    home_pose, workspace, pose_frame_id, gripper_command_mode = get_replay_settings(cfg)

    actions = load_actions(
        args.episode_path,
        start_step=args.start_step,
        end_step=args.end_step,
        stride=args.stride,
        max_steps=args.max_steps,
        timing_mode=args.timing_mode,
        timestamp_dataset=args.timestamp_dataset,
    )
    actions, timestamps, timing_source = actions

    robot, rospy = build_robot(
        dummy=args.dummy,
        home_pose=home_pose,
        workspace=workspace,
        pose_frame_id=pose_frame_id,
        gripper_command_mode=gripper_command_mode,
    )

    dt = 1.0 / args.hz
    print(f"[Replay] episode={args.episode_path}")
    if timestamps is None:
        timing_desc = f"fixed {args.hz:.2f} Hz"
    else:
        avg_dt = float(np.mean(np.diff(timestamps))) if len(timestamps) > 1 else 0.0
        timing_desc = f"timestamps from {timing_source} (avg_dt={avg_dt:.4f}s)"
    print(f"[Replay] selected_actions={actions.shape} timing={timing_desc}")
    print(
        f"[Replay] input_gripper_mode={args.input_gripper_mode} "
        f"robot_gripper_mode={gripper_command_mode}"
    )
    print(
        "[Replay] first action xyz="
        f"{np.array2string(actions[0, :3], precision=4, suppress_small=True)} "
        f"gripper={float(actions[0, 7]) if actions.shape[1] > 7 else float('nan'):.4f}"
    )

    if args.wait_enter:
        input("[Replay] Press Enter to start replay...")

    try:
        if not args.skip_home:
            robot.go_home()
            if args.settle_seconds > 0:
                time.sleep(args.settle_seconds)

        for idx, raw_action in enumerate(actions):
            if rospy is not None and rospy.is_shutdown():
                raise KeyboardInterrupt("ROS shutdown requested")

            t0 = time.time()
            cmd = prepare_action(
                raw_action,
                input_gripper_mode=args.input_gripper_mode,
                gripper_command_mode=gripper_command_mode,
                binary_threshold=args.binary_threshold,
            )
            robot.execute_action(cmd)

            if idx % max(args.log_every, 1) == 0 or idx == len(actions) - 1:
                print(
                    f"[Replay] step {idx:04d}/{len(actions) - 1:04d} "
                    f"xyz={np.array2string(cmd[:3], precision=4, suppress_small=True)} "
                    f"quat={np.array2string(cmd[3:7], precision=4, suppress_small=True)} "
                    f"gripper={float(cmd[7]):.4f}"
                )

            elapsed = time.time() - t0
            target_dt = compute_sleep_time(idx, timestamps, dt)
            sleep_time = target_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        print("[Replay] Replay complete.")
    except KeyboardInterrupt:
        print("\n[Replay] Interrupted.")
    finally:
        robot.stop()
        print("[Replay] Robot stop command sent.")


if __name__ == "__main__":
    main()
