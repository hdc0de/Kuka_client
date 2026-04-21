#!/usr/bin/env python3
"""Feed HDF5 observations to the policy server and execute returned actions.

This script is the HDF5-backed counterpart of ros_bridge_original_act.py:

    HDF5 qpos/images -> ZMQ policy server -> predicted action -> robot.execute_action()

Run it on the ROS machine. The policy server still runs on the laptop/GPU side.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import yaml
import zmq

from msg_utils import pack_observation, unpack_chunk
from ros_bridge_original_act import (
    DEFAULT_WORKSPACE,
    HOME_POSE,
    POSE_FRAME_ID,
    DummyCartesianRobot,
    ROSCartesianRobot,
)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Use HDF5 qpos/images as observations for the policy server."
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
        help="YAML config used by server_original_act.py / ros_bridge_original_act.py",
    )
    parser.add_argument(
        "--server",
        type=str,
        default=None,
        help="Override policy server address, e.g. tcp://192.168.1.103:5555",
    )
    parser.add_argument("--hz", type=float, default=None, help="Execution rate")
    parser.add_argument("--start-step", type=int, default=0, help="First HDF5 timestep")
    parser.add_argument(
        "--end-step",
        type=int,
        default=None,
        help="Stop before this HDF5 timestep",
    )
    parser.add_argument("--stride", type=int, default=1, help="Use every Nth HDF5 step")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Optional hard limit after slicing",
    )
    parser.add_argument(
        "--wait-enter",
        action="store_true",
        help="Wait for Enter before moving to home / starting the episode",
    )
    parser.add_argument(
        "--skip-home",
        action="store_true",
        help="Do not move the robot to home before execution",
    )
    parser.add_argument(
        "--settle-seconds",
        type=float,
        default=1.0,
        help="Pause after go_home before starting",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Use DummyCartesianRobot instead of ROS execution",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=25,
        help="Print one progress line every N executed actions",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=None,
        help="Optional directory to save predicted vs ground-truth actions as .npy",
    )
    return parser.parse_args()


def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def select_steps(
    total_steps: int,
    start_step: int,
    end_step: int | None,
    stride: int,
    max_steps: int | None,
) -> np.ndarray:
    if stride <= 0:
        raise ValueError("--stride must be >= 1")
    start = max(0, int(start_step))
    stop = total_steps if end_step is None else min(int(end_step), total_steps)
    steps = np.arange(start, stop, stride, dtype=np.int64)
    if max_steps is not None:
        steps = steps[: max(0, int(max_steps))]
    if len(steps) == 0:
        raise ValueError("No steps selected. Check --start-step/--end-step/--stride.")
    return steps


def get_runtime_settings(cfg: dict, args: argparse.Namespace):
    server_cfg = cfg.get("server", {})
    bridge_cfg = cfg.get("ros_bridge_original_act", cfg.get("ros_bridge", {}))
    robot_cfg = cfg.get("robot", {})

    server = args.server or bridge_cfg.get("server_address") or server_cfg.get(
        "connect_address", server_cfg.get("address", "tcp://localhost:5555")
    )
    hz = float(args.hz if args.hz is not None else bridge_cfg.get("hz", 30))
    home_pose = np.asarray(
        robot_cfg.get("home_cartesian_pose", HOME_POSE),
        dtype=np.float32,
    )
    workspace = np.asarray(
        robot_cfg.get("workspace_xyz", DEFAULT_WORKSPACE),
        dtype=np.float32,
    )
    pose_frame_id = bridge_cfg.get("pose_frame_id", POSE_FRAME_ID)

    hdf5_camera_keys = list(server_cfg.get("act_camera_names", ["cam_front", "cam_wrist"]))
    obs_camera_keys = list(server_cfg.get("act_camera_sources", hdf5_camera_keys))
    if len(hdf5_camera_keys) != len(obs_camera_keys):
        raise ValueError(
            "Config mismatch: act_camera_names and act_camera_sources must have the same length."
        )
    camera_map = list(zip(hdf5_camera_keys, obs_camera_keys))
    return server, hz, home_pose, workspace, pose_frame_id, camera_map


def build_robot(
    dummy: bool,
    home_pose: np.ndarray,
    workspace: np.ndarray,
    pose_frame_id: str,
):
    if dummy:
        return DummyCartesianRobot(
            home_pose=home_pose,
            workspace=workspace,
            gripper_feedback_mode="width",
            gripper_command_mode="width",
        ), None

    import rospy

    rospy.init_node("hdf5_client_bridge", anonymous=True)
    robot = ROSCartesianRobot(
        home_pose=home_pose,
        workspace=workspace,
        frame_id=pose_frame_id,
        gripper_feedback_mode="width",
        gripper_command_mode="width",
    )
    robot.wait_for_state()
    robot.activate_gripper()
    return robot, rospy


def make_observation(root: h5py.File, step: int, camera_map: list[tuple[str, str]]) -> dict:
    qpos = np.asarray(root["/observations/qpos"][step], dtype=np.float32)
    if qpos.shape[0] < 8:
        raise ValueError(f"Expected qpos dim >= 8, got {qpos.shape}")

    images = {}
    for hdf5_key, obs_key in camera_map:
        ds_path = f"/observations/images/{hdf5_key}"
        if ds_path not in root:
            raise KeyError(f"Missing image dataset in HDF5: {ds_path}")
        images[obs_key] = root[ds_path][step]

    return {
        "joint_positions": qpos[:7].copy(),
        "gripper_state": float(qpos[7]),
        "images": images,
    }


def save_outputs(
    save_dir: Path,
    executed_steps: np.ndarray,
    predicted_actions: np.ndarray,
    gt_actions: np.ndarray,
    query_steps: np.ndarray,
):
    save_dir.mkdir(parents=True, exist_ok=True)
    np.save(save_dir / "executed_steps.npy", executed_steps.astype(np.int64))
    np.save(save_dir / "predicted_actions.npy", predicted_actions.astype(np.float32))
    np.save(save_dir / "ground_truth_actions.npy", gt_actions.astype(np.float32))
    np.save(save_dir / "query_steps.npy", query_steps.astype(np.int64))
    print(f"[HDF5Bridge] Saved outputs to {save_dir}")


def recv_chunk(sock, timeout_ms: int = 1000) -> tuple[np.ndarray, bool]:
    while True:
        try:
            data = sock.recv()
            return unpack_chunk(data)
        except zmq.Again:
            print("[HDF5Bridge] Waiting for policy server reply...")
            continue


def main():
    args = parse_args()
    cfg = load_config(args.config)
    server, hz, home_pose, workspace, pose_frame_id, camera_map = get_runtime_settings(
        cfg, args
    )
    dt = 1.0 / hz

    with h5py.File(args.episode_path, "r") as root:
        if "/action" not in root or "/observations/qpos" not in root:
            raise KeyError(
                f"{args.episode_path} must contain /action and /observations/qpos"
            )
        total_steps = int(root["/action"].shape[0])
        steps = select_steps(
            total_steps=total_steps,
            start_step=args.start_step,
            end_step=args.end_step,
            stride=args.stride,
            max_steps=args.max_steps,
        )

        gt_action_ds = root["/action"]
        robot, rospy = build_robot(
            dummy=args.dummy,
            home_pose=home_pose,
            workspace=workspace,
            pose_frame_id=pose_frame_id,
        )

        ctx = zmq.Context()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVTIMEO, 1000)
        sock.connect(server)

        print(f"[HDF5Bridge] episode={args.episode_path}")
        print(f"[HDF5Bridge] server={server}")
        print(f"[HDF5Bridge] selected_steps={len(steps)} hz={hz:.2f}")
        print(f"[HDF5Bridge] camera_map={camera_map}")

        if args.wait_enter:
            input("[HDF5Bridge] Press Enter to start...")

        predicted_actions = []
        gt_actions = []
        executed_steps = []
        query_steps = []

        action_buf = None
        buf_idx = 0

        try:
            if not args.skip_home:
                robot.go_home()
                if args.settle_seconds > 0:
                    time.sleep(args.settle_seconds)

            first_obs = make_observation(root, int(steps[0]), camera_map)
            sock.send(
                pack_observation(
                    first_obs["images"],
                    first_obs["joint_positions"],
                    first_obs["gripper_state"],
                    reset=True,
                )
            )
            recv_chunk(sock)
            print("[HDF5Bridge] Episode reset acknowledged by policy server.")

            for local_idx, step in enumerate(steps):
                if rospy is not None and rospy.is_shutdown():
                    raise KeyboardInterrupt("ROS shutdown requested")

                t0 = time.time()
                obs = make_observation(root, int(step), camera_map)

                if action_buf is None or buf_idx >= len(action_buf):
                    sock.send(
                        pack_observation(
                            obs["images"],
                            obs["joint_positions"],
                            obs["gripper_state"],
                            reset=False,
                        )
                    )
                    action_buf, done = recv_chunk(sock)
                    action_buf = np.asarray(action_buf, dtype=np.float32)
                    if action_buf.ndim == 1:
                        action_buf = action_buf.reshape(1, -1)
                    buf_idx = 0
                    query_steps.append(int(step))
                    if done:
                        print(f"[HDF5Bridge] Server signaled done at HDF5 step {step}")

                action = np.asarray(action_buf[buf_idx], dtype=np.float32)
                buf_idx += 1

                robot.execute_action(action)

                predicted_actions.append(action.copy())
                gt_actions.append(np.asarray(gt_action_ds[step], dtype=np.float32))
                executed_steps.append(int(step))

                if local_idx % max(args.log_every, 1) == 0 or local_idx == len(steps) - 1:
                    gt = gt_actions[-1]
                    print(
                        f"[HDF5Bridge] step {local_idx:04d}/{len(steps) - 1:04d} "
                        f"hdf5_t={step:04d} "
                        f"pred_xyz={np.array2string(action[:3], precision=4, suppress_small=True)} "
                        f"gt_xyz={np.array2string(gt[:3], precision=4, suppress_small=True)} "
                        f"pred_gripper={float(action[7]):.4f} "
                        f"gt_gripper={float(gt[7]):.4f}"
                    )

                elapsed = time.time() - t0
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            print("[HDF5Bridge] Finished selected HDF5 steps.")

        except KeyboardInterrupt:
            print("\n[HDF5Bridge] Interrupted.")
        finally:
            if args.save_dir is not None and predicted_actions:
                save_outputs(
                    save_dir=args.save_dir,
                    executed_steps=np.asarray(executed_steps),
                    predicted_actions=np.asarray(predicted_actions),
                    gt_actions=np.asarray(gt_actions),
                    query_steps=np.asarray(query_steps),
                )
            robot.stop()
            sock.close()
            ctx.term()
            print("[HDF5Bridge] Shutdown complete.")


if __name__ == "__main__":
    main()
