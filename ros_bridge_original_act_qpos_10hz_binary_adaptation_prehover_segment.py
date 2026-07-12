#!/usr/bin/env python3
"""ROS bridge for the original-ACT 10 Hz qpos binary policy.

This is a self-contained bridge that runs on the ROS machine. It keeps the
same ZMQ protocol as the existing bridge, but packs Cartesian pose into the
`joint_positions` field for compatibility with the current server/msg format.

The bridge sends physical gripper width to the server. The matching policy
adapter converts it with the same threshold used by
bag2hdf5_qpos_10hz_binary_strict.py.

Observation sent to server:
  - images: camera frames
  - joint_positions: actually [x, y, z, qx, qy, qz, qw]
  - gripper_state: gripper width (default) or normalized scalar

Action received from server:
  - action[:7]: target Cartesian pose
  - action[7]: gripper width (default) or normalized scalar
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
import time
from dataclasses import dataclass

import cv2
import msgpack
import numpy as np
import zmq


# ============================================================================
# Embedded msg_utils
# ============================================================================

def encode_image(image: np.ndarray, quality: int = 90) -> bytes:
    _, buf = cv2.imencode(
        ".jpg",
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    return buf.tobytes()


def decode_image(data: bytes) -> np.ndarray:
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def encode_depth(depth: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".png", depth)
    return buf.tobytes()


def _np_to_bytes(arr: np.ndarray) -> bytes:
    dtype_str = arr.dtype.str.encode()
    header = struct.pack("B", len(dtype_str)) + dtype_str
    header += struct.pack("<I", arr.ndim)
    for size in arr.shape:
        header += struct.pack("<I", size)
    return header + arr.tobytes()


def _bytes_to_np(data: bytes) -> np.ndarray:
    offset = 0
    dtype_len = struct.unpack("B", data[offset:offset + 1])[0]
    offset += 1
    dtype_str = data[offset:offset + dtype_len].decode()
    offset += dtype_len
    ndim = struct.unpack("<I", data[offset:offset + 4])[0]
    offset += 4
    shape = []
    for _ in range(ndim):
        shape.append(struct.unpack("<I", data[offset:offset + 4])[0])
        offset += 4
    return np.frombuffer(data[offset:], dtype=np.dtype(dtype_str)).reshape(shape).copy()


def pack_observation(images: dict, joint_positions: np.ndarray,
                     gripper_state: float, reset: bool = False,
                     jpeg_quality: int = 70) -> bytes:
    encoded_images = {}
    for name, img in images.items():
        if name.endswith("_depth"):
            encoded_images[name] = encode_depth(img)
        else:
            encoded_images[name] = encode_image(img, quality=jpeg_quality)

    msg = {
        "images": encoded_images,
        "joint_positions": _np_to_bytes(joint_positions.astype(np.float32)),
        "gripper_state": float(gripper_state),
        "timestamp": time.time(),
        "reset": reset,
    }
    return msgpack.packb(msg, use_bin_type=True)


def unpack_chunk(data: bytes) -> tuple[np.ndarray, bool]:
    msg = msgpack.unpackb(data, raw=False)
    return _bytes_to_np(msg["actions"]), msg["done"]


# ============================================================================
# Embedded camera capture
# ============================================================================

class CameraManager:
    def __init__(self, configs: list[dict]):
        import pyrealsense2 as rs

        self.rs = rs
        self.configs = configs
        self.pipelines = []
        self._start_cameras()

    def _start_cameras(self):
        rs = self.rs
        for cfg in self.configs:
            pipeline = rs.pipeline()
            rs_config = rs.config()

            serial = cfg.get("serial", "")
            if serial:
                rs_config.enable_device(serial)

            width = cfg.get("width", 640)
            height = cfg.get("height", 480)
            fps = cfg.get("fps", 30)
            mode = cfg.get("mode", "rgb")

            rs_config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
            if mode == "rgbd":
                rs_config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

            pipeline.start(rs_config)
            self.pipelines.append(pipeline)
            print(
                f"[Camera] Started '{cfg['name']}' "
                f"(serial={serial or 'auto'}, {width}x{height}, {mode})"
            )

    def capture(self) -> dict:
        result = {}
        for cfg, pipeline in zip(self.configs, self.pipelines):
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            result[cfg["name"]] = np.asarray(color_frame.get_data())
            if cfg.get("mode", "rgb") == "rgbd":
                depth_frame = frames.get_depth_frame()
                result[f"{cfg['name']}_depth"] = np.asarray(depth_frame.get_data())
        return result

    def stop(self):
        for pipeline in self.pipelines:
            pipeline.stop()
        self.pipelines.clear()
        print("[Camera] All cameras stopped.")


class DummyCamera:
    def __init__(self, configs: list[dict]):
        self.configs = configs
        print(f"[DummyCamera] Initialized {len(configs)} virtual cameras")

    def capture(self) -> dict:
        result = {}
        for cfg in self.configs:
            width = cfg.get("width", 640)
            height = cfg.get("height", 480)
            result[cfg["name"]] = np.random.randint(
                0, 255, (height, width, 3), dtype=np.uint8
            )
            if cfg.get("mode", "rgb") == "rgbd":
                result[f"{cfg['name']}_depth"] = np.random.randint(
                    0, 5000, (height, width), dtype=np.uint16
                )
        return result

    def stop(self):
        print("[DummyCamera] Stopped.")


class ROSTopicCameraManager:
    def __init__(self, configs: list[dict]):
        import rospy
        from cv_bridge import CvBridge
        from sensor_msgs.msg import Image

        self._rospy = rospy
        self._bridge = CvBridge()
        self._Image = Image
        self.configs = configs
        self._latest_images = {cfg["name"]: None for cfg in configs}

        for cfg in configs:
            topic = cfg.get("topic")
            if not topic:
                raise ValueError(
                    f"Camera '{cfg['name']}' is missing 'topic' for ros_topics mode"
                )
            rospy.Subscriber(
                topic,
                Image,
                lambda msg, c=cfg: self._image_cb(msg, c),
                queue_size=1,
                buff_size=2**24,
            )
            rospy.loginfo(
                f"[ROSCamera] Subscribed '{cfg['name']}' <- {topic} "
                f"encoding={cfg.get('encoding', 'rgb8')}"
            )

    def _image_cb(self, msg, cfg: dict):
        try:
            encoding = cfg.get("encoding", "rgb8")
            img = self._bridge.imgmsg_to_cv2(msg, desired_encoding=encoding)
            width = cfg.get("width", 640)
            height = cfg.get("height", 480)
            if img.shape[:2] != (height, width):
                img = cv2.resize(img, (width, height))
            self._latest_images[cfg["name"]] = img
        except Exception as exc:
            self._rospy.logwarn_throttle(
                5, f"[ROSCamera] Image decode error ({cfg['name']}): {exc}"
            )

    def wait_until_ready(self):
        rate = self._rospy.Rate(10)
        while not self._rospy.is_shutdown():
            ready = all(img is not None for img in self._latest_images.values())
            if ready:
                return
            self._rospy.logwarn_throttle(2, "[ROSCamera] Waiting for camera images...")
            rate.sleep()

    def capture(self) -> dict:
        result = {}
        for cfg in self.configs:
            img = self._latest_images[cfg["name"]]
            if img is None:
                raise RuntimeError(f"Camera '{cfg['name']}' image not ready yet")
            result[cfg["name"]] = img.copy()
        return result

    def stop(self):
        print("[ROSCamera] Stopped")


# ============================================================================
# Cartesian ACT robot interface
# ============================================================================

CARTESIAN_STATE_TOPIC = "/iiwa/state/CartesianPose"
CMD_POSE_TOPIC = "/iiwa/command/CartesianPose"
GRIPPER_STATE_TOPIC = "/Robotiq2FGripperRobotInput"
GRIPPER_CMD_TOPIC = "/Robotiq2FGripperRobotOutput"

HOME_POSE = np.array(
    [0.61, 0.0, 0.37, 0.0, 0.9998, 0.0, -0.0175],
    dtype=np.float32,
)
HOME_POS_TOL = 0.01
HOME_ANG_TOL_DEG = 1.0
HOME_TIMEOUT = 15.0
POSE_FRAME_ID = "iiwa_link_0"

DEFAULT_WORKSPACE = np.array(
    [
        [0.45, 0.75],
        [-0.22, 0.42],
        [0.197, 1.00],
    ],
    dtype=np.float32,
)

GRIPPER_WIDTH_MIN = 0.0
GRIPPER_WIDTH_MAX = 0.085


def pose_to_array(pose) -> np.ndarray:
    pos = pose.position
    ori = pose.orientation
    return np.array(
        [pos.x, pos.y, pos.z, ori.x, ori.y, ori.z, ori.w],
        dtype=np.float32,
    )


def array_to_posestamped(arr: np.ndarray, frame_id: str):
    from geometry_msgs.msg import PoseStamped

    msg = PoseStamped()
    msg.header.stamp = __import__("rospy").Time.now()
    msg.header.frame_id = frame_id
    msg.pose.position.x = float(arr[0])
    msg.pose.position.y = float(arr[1])
    msg.pose.position.z = float(arr[2])
    msg.pose.orientation.x = float(arr[3])
    msg.pose.orientation.y = float(arr[4])
    msg.pose.orientation.z = float(arr[5])
    msg.pose.orientation.w = float(arr[6])
    return msg


def width_to_normalized(width: float) -> float:
    span = GRIPPER_WIDTH_MAX - GRIPPER_WIDTH_MIN
    if span <= 1e-6:
        return -1.0
    alpha = (float(width) - GRIPPER_WIDTH_MIN) / span
    return float(np.clip(alpha * 2.0 - 1.0, -1.0, 1.0))


def normalized_to_width(value: float) -> float:
    alpha = (float(value) + 1.0) * 0.5
    return float(
        np.clip(
            GRIPPER_WIDTH_MIN + alpha * (GRIPPER_WIDTH_MAX - GRIPPER_WIDTH_MIN),
            GRIPPER_WIDTH_MIN,
            GRIPPER_WIDTH_MAX,
        )
    )


def gripper_width_from_input(msg) -> float:
    return float(np.clip((-msg.gPO + 229) / 226.0 * 0.085, 0.0, 0.085))


def build_gripper_cmd(gripper_width: float):
    from robotiq_2f_gripper_control.msg import Robotiq2FGripper_robot_output

    width = float(np.clip(gripper_width, GRIPPER_WIDTH_MIN, GRIPPER_WIDTH_MAX))
    msg = Robotiq2FGripper_robot_output()
    msg.rACT = 1
    msg.rGTO = 1
    msg.rSP = 255
    msg.rFR = 50
    msg.rPR = 255 if width < 0.065 else 0
    return msg


def normalize_quaternion(quat: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat)
    if norm > 1e-6:
        return quat / norm
    return fallback.copy()


def safety_clip(action: np.ndarray, workspace: np.ndarray, home_pose: np.ndarray) -> np.ndarray:
    out = np.asarray(action, dtype=np.float32).copy()
    for axis in range(3):
        lo, hi = workspace[axis]
        out[axis] = float(np.clip(out[axis], lo, hi))
    out[3:7] = normalize_quaternion(out[3:7], home_pose[3:7])
    if out.shape[0] > 7:
        out[7] = float(np.clip(out[7], GRIPPER_WIDTH_MIN, GRIPPER_WIDTH_MAX))
    return out


def is_home_reached(current_pose, home_pose: np.ndarray) -> bool:
    import tf

    current = pose_to_array(current_pose)
    pos_err = np.linalg.norm(current[:3] - home_pose[:3])

    q_curr = current[3:7].tolist()
    q_home = home_pose[3:7].tolist()
    q_home_inv = tf.transformations.quaternion_inverse(q_home)
    q_rel = tf.transformations.quaternion_multiply(q_home_inv, q_curr)
    ang = 2.0 * math.acos(max(min(abs(q_rel[3]), 1.0), 0.0))
    ang_deg = math.degrees(ang)

    return pos_err < HOME_POS_TOL and ang_deg < HOME_ANG_TOL_DEG


class ROSCartesianRobot:
    def __init__(
        self,
        home_pose: np.ndarray,
        workspace: np.ndarray,
        frame_id: str = POSE_FRAME_ID,
        gripper_feedback_mode: str = "width",
        gripper_command_mode: str = "width",
    ):
        import threading
        import rospy
        from iiwa_msgs.msg import CartesianPose
        from robotiq_2f_gripper_control.msg import (
            Robotiq2FGripper_robot_input,
            Robotiq2FGripper_robot_output,
        )
        from geometry_msgs.msg import PoseStamped

        self._rospy = rospy
        self._CartesianPose = CartesianPose
        self._PoseStamped = PoseStamped
        self._GripperInput = Robotiq2FGripper_robot_input
        self._GripperOutput = Robotiq2FGripper_robot_output

        self._lock = threading.Lock()
        self._current_pose = None
        self._gripper_width = GRIPPER_WIDTH_MAX
        self._home_pose = np.asarray(home_pose, dtype=np.float32)
        self._workspace = np.asarray(workspace, dtype=np.float32)
        self._frame_id = frame_id
        self._gripper_feedback_mode = gripper_feedback_mode
        self._gripper_command_mode = gripper_command_mode

        rospy.Subscriber(CARTESIAN_STATE_TOPIC, CartesianPose, self._pose_cb, queue_size=1)
        self._pose_pub = rospy.Publisher(CMD_POSE_TOPIC, PoseStamped, queue_size=1)

        if GRIPPER_STATE_TOPIC:
            rospy.Subscriber(
                GRIPPER_STATE_TOPIC,
                Robotiq2FGripper_robot_input,
                self._gripper_cb,
                queue_size=1,
            )
        if GRIPPER_CMD_TOPIC:
            self._gripper_pub = rospy.Publisher(
                GRIPPER_CMD_TOPIC,
                Robotiq2FGripper_robot_output,
                queue_size=1,
            )

        rospy.loginfo(f"[ROSCartesianRobot] Subscribed to {CARTESIAN_STATE_TOPIC}")
        rospy.loginfo(f"[ROSCartesianRobot] Publishing pose to {CMD_POSE_TOPIC}")

    def _pose_cb(self, msg):
        with self._lock:
            if hasattr(msg, "poseStamped"):
                self._current_pose = msg.poseStamped.pose
            else:
                self._current_pose = msg.pose

    def _gripper_cb(self, msg):
        with self._lock:
            self._gripper_width = gripper_width_from_input(msg)

    def wait_for_state(self):
        while not self._rospy.is_shutdown():
            with self._lock:
                ready = self._current_pose is not None
            if ready:
                return
            self._rospy.logwarn_throttle(2, "[ROSCartesianRobot] Waiting for robot pose...")
            time.sleep(0.1)

    def get_cartesian_pose(self) -> np.ndarray:
        with self._lock:
            if self._current_pose is None:
                return self._home_pose.copy()
            return pose_to_array(self._current_pose)

    def get_gripper_state(self) -> float:
        with self._lock:
            width = float(self._gripper_width)
        if self._gripper_feedback_mode == "width":
            return width
        if self._gripper_feedback_mode == "normalized":
            return width_to_normalized(width)
        raise ValueError(
            f"Unsupported gripper_feedback_mode: {self._gripper_feedback_mode}"
        )

    def activate_gripper(self):
        if not GRIPPER_CMD_TOPIC:
            return
        rospy = self._rospy

        rospy.loginfo("[ROSCartesianRobot] Gripper activation: reset...")
        msg = self._GripperOutput()
        msg.rACT = 0
        for _ in range(10):
            self._gripper_pub.publish(msg)
            time.sleep(0.1)

        rospy.sleep(3.0)
        rospy.loginfo("[ROSCartesianRobot] Gripper activation: enable...")
        msg = self._GripperOutput()
        msg.rACT = 1
        msg.rGTO = 1
        msg.rSP = 255
        msg.rFR = 50
        msg.rPR = 0
        self._gripper_pub.publish(msg)
        rospy.sleep(5.0)
        rospy.loginfo("[ROSCartesianRobot] Gripper ready.")

    def execute_action(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] < 7:
            raise ValueError(f"Cartesian action must have at least 7 values, got {action.shape[0]}")

        action_cmd = action.copy()
        if action_cmd.shape[0] >= 8 and self._gripper_command_mode == "normalized":
            action_cmd[7] = normalized_to_width(action_cmd[7])
        elif self._gripper_command_mode not in ("width", "normalized"):
            raise ValueError(
                f"Unsupported gripper_command_mode: {self._gripper_command_mode}"
            )

        action_cmd = safety_clip(action_cmd, self._workspace, self._home_pose)
        pose_msg = array_to_posestamped(action_cmd[:7], self._frame_id)
        self._pose_pub.publish(pose_msg)

        if action_cmd.shape[0] > 7 and GRIPPER_CMD_TOPIC:
            self._gripper_pub.publish(build_gripper_cmd(action_cmd[7]))

    def go_home(self):
        rospy = self._rospy
        rospy.loginfo("[ROSCartesianRobot] Moving to home pose...")
        deadline = time.time() + HOME_TIMEOUT
        while not rospy.is_shutdown():
            with self._lock:
                pose = self._current_pose
            if pose is not None and is_home_reached(pose, self._home_pose):
                rospy.loginfo("[ROSCartesianRobot] Home pose reached.")
                break
            self.execute_action(np.append(self._home_pose, GRIPPER_WIDTH_MAX))
            if time.time() > deadline:
                rospy.logwarn("[ROSCartesianRobot] Home timeout. Proceeding anyway.")
                break
            time.sleep(1.0 / 30.0)

    def stop(self):
        pose = self.get_cartesian_pose()
        self.execute_action(np.append(pose, GRIPPER_WIDTH_MAX))
        self._rospy.loginfo("[ROSCartesianRobot] Stop: holding current pose")

    def open_gripper(self):
        pose = self.get_cartesian_pose()
        open_action = np.append(pose, GRIPPER_WIDTH_MAX)
        for _ in range(5):
            self.execute_action(open_action)
            time.sleep(0.05)
        self._rospy.loginfo("[ROSCartesianRobot] Open gripper at current pose x5")


class DummyCartesianRobot:
    def __init__(
        self,
        home_pose: np.ndarray,
        workspace: np.ndarray,
        gripper_feedback_mode: str = "width",
        gripper_command_mode: str = "width",
    ):
        self._pose = np.asarray(home_pose, dtype=np.float32).copy()
        self._home_pose = np.asarray(home_pose, dtype=np.float32).copy()
        self._workspace = np.asarray(workspace, dtype=np.float32).copy()
        self._gripper_width = GRIPPER_WIDTH_MAX
        self._gripper_feedback_mode = gripper_feedback_mode
        self._gripper_command_mode = gripper_command_mode
        print("[DummyCartesianRobot] Initialized")

    def wait_for_state(self):
        return

    def get_cartesian_pose(self) -> np.ndarray:
        return self._pose.copy()

    def get_gripper_state(self) -> float:
        if self._gripper_feedback_mode == "width":
            return float(self._gripper_width)
        if self._gripper_feedback_mode == "normalized":
            return width_to_normalized(self._gripper_width)
        raise ValueError(
            f"Unsupported gripper_feedback_mode: {self._gripper_feedback_mode}"
        )

    def activate_gripper(self):
        print("[DummyCartesianRobot] Gripper ready")

    def execute_action(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] >= 8 and self._gripper_command_mode == "normalized":
            action = action.copy()
            action[7] = normalized_to_width(action[7])
        action = safety_clip(action, self._workspace, self._home_pose)
        self._pose = action[:7].copy()
        if action.shape[0] > 7:
            self._gripper_width = float(action[7])

    def go_home(self):
        self._pose = self._home_pose.copy()
        self._gripper_width = GRIPPER_WIDTH_MAX
        print("[DummyCartesianRobot] Moved to home pose")

    def open_gripper(self):
        self.execute_action(np.append(self._pose, GRIPPER_WIDTH_MAX))
        print("[DummyCartesianRobot] Opened gripper")

    def stop(self):
        print("[DummyCartesianRobot] Stopped")


# ============================================================================
# Main loop
# ============================================================================

DEFAULT_CAMERAS = [
    {"name": "front", "serial": "308222301757", "width": 640, "height": 480, "fps": 30, "mode": "rgb"},
    {"name": "wrist", "serial": "306322300452", "width": 640, "height": 480, "fps": 30, "mode": "rgb"},
]


def _get_bridge_config(cfg: dict) -> dict:
    return cfg.get(
        "ros_bridge_original_act_qpos_10hz_binary",
        cfg.get("ros_bridge_original_act", cfg.get("ros_bridge", {})),
    )


def _ros_is_shutdown() -> bool:
    rospy = sys.modules.get("rospy")
    return bool(rospy is not None and rospy.is_shutdown())


def _recv_reply(sock, label: str):
    while True:
        try:
            return sock.recv()
        except zmq.Again:
            if _ros_is_shutdown():
                raise KeyboardInterrupt(f"ROS shutdown while waiting for {label}")
            print(f"[Bridge] Waiting for server reply: {label} ...")


@dataclass
class AdaptationConfig:
    enabled: bool = False
    object_pose_fixed: np.ndarray | None = None
    object_pose_topic: str | None = None
    use_pose_topic: bool = False
    hover_z_offset: float = 0.10
    approach_xy_threshold: float = 0.12
    hover_xy_tolerance: float = 0.015
    hover_z_tolerance: float = 0.015
    hover_kp: float = 0.7
    hover_max_step: float = 0.025
    fixed_place_entry_pose: np.ndarray | None = None
    place_entry_tolerance: float = 0.015
    carry_kp: float = 0.5
    carry_max_step: float = 0.025
    expert_hdf5_path: str | None = None
    expert_start_idx: int = 0
    expert_end_idx: int | None = None
    expert_auto_segment: bool = True
    expert_auto_z_source: str = "qpos"
    expert_auto_close_source: str = "action"
    expert_auto_window: int = 15
    expert_auto_min_drop: float = 0.04
    expert_auto_min_negative_steps: int = 5
    expert_auto_post_close_frames: int = 10
    expert_use_images: bool = True
    expert_use_qpos: bool = True
    expert_follow_xy_object: bool = True
    expert_z_relative: bool = True
    expert_close_width_threshold: float = 0.065
    policy_reset_after_manual: bool = True
    open_width: float = GRIPPER_WIDTH_MAX
    closed_width: float = GRIPPER_WIDTH_MIN


def _array_or_none(value, dtype=np.float32):
    if value is None:
        return None
    return np.asarray(value, dtype=dtype)


def load_adaptation_config(cfg: dict | None) -> AdaptationConfig:
    raw = (cfg or {}).get("adaptation", {})
    expert_end_idx = raw.get("expert_end_idx")
    if expert_end_idx is not None:
        expert_end_idx = int(expert_end_idx)
    return AdaptationConfig(
        enabled=bool(raw.get("enabled", False)),
        object_pose_fixed=_array_or_none(raw.get("object_pose_fixed")),
        object_pose_topic=raw.get("object_pose_topic", "/pose_cube"),
        use_pose_topic=bool(raw.get("use_pose_topic", False)),
        hover_z_offset=float(raw.get("hover_z_offset", 0.10)),
        approach_xy_threshold=float(raw.get("approach_xy_threshold", 0.12)),
        hover_xy_tolerance=float(raw.get("hover_xy_tolerance", 0.015)),
        hover_z_tolerance=float(raw.get("hover_z_tolerance", 0.015)),
        hover_kp=float(raw.get("hover_kp", 0.7)),
        hover_max_step=float(raw.get("hover_max_step", 0.025)),
        fixed_place_entry_pose=_array_or_none(raw.get("fixed_place_entry_pose")),
        place_entry_tolerance=float(raw.get("place_entry_tolerance", 0.015)),
        carry_kp=float(raw.get("carry_kp", 0.5)),
        carry_max_step=float(raw.get("carry_max_step", 0.025)),
        expert_hdf5_path=raw.get("expert_hdf5_path"),
        expert_start_idx=int(raw.get("expert_start_idx", 0)),
        expert_end_idx=expert_end_idx,
        expert_auto_segment=bool(raw.get("expert_auto_segment", True)),
        expert_auto_z_source=raw.get("expert_auto_z_source", "qpos"),
        expert_auto_close_source=raw.get("expert_auto_close_source", "action"),
        expert_auto_window=int(raw.get("expert_auto_window", 15)),
        expert_auto_min_drop=float(raw.get("expert_auto_min_drop", 0.04)),
        expert_auto_min_negative_steps=int(raw.get("expert_auto_min_negative_steps", 5)),
        expert_auto_post_close_frames=int(raw.get("expert_auto_post_close_frames", 10)),
        expert_use_images=bool(raw.get("expert_use_images", True)),
        expert_use_qpos=bool(raw.get("expert_use_qpos", True)),
        expert_follow_xy_object=bool(raw.get("expert_follow_xy_object", True)),
        expert_z_relative=bool(raw.get("expert_z_relative", True)),
        expert_close_width_threshold=float(raw.get("expert_close_width_threshold", 0.065)),
        policy_reset_after_manual=bool(raw.get("policy_reset_after_manual", True)),
        open_width=float(raw.get("open_width", GRIPPER_WIDTH_MAX)),
        closed_width=float(raw.get("closed_width", GRIPPER_WIDTH_MIN)),
    )


class ObjectPoseProvider:
    def __init__(self, cfg: AdaptationConfig, dummy: bool = False):
        import threading

        self._lock = threading.Lock()
        self._pose = None
        if cfg.object_pose_fixed is not None:
            fixed = np.asarray(cfg.object_pose_fixed, dtype=np.float32).reshape(-1)
            if fixed.shape[0] < 3:
                raise ValueError("adaptation.object_pose_fixed must contain at least [x, y, z]")
            self._pose = fixed[:3].copy()

        if cfg.use_pose_topic and not dummy:
            import rospy
            from std_msgs.msg import Float32MultiArray

            rospy.Subscriber(
                cfg.object_pose_topic,
                Float32MultiArray,
                self._pose_cb,
                queue_size=1,
            )
            rospy.loginfo(f"[Adaptation] Subscribed object pose <- {cfg.object_pose_topic}")

    def _pose_cb(self, msg):
        data = np.asarray(msg.data, dtype=np.float32).reshape(-1)
        if data.shape[0] >= 3:
            with self._lock:
                self._pose = data[:3].copy()

    def get(self) -> np.ndarray | None:
        with self._lock:
            if self._pose is None:
                return None
            return self._pose.copy()


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
            "Try lowering adaptation.expert_auto_min_drop or "
            "adaptation.expert_auto_min_negative_steps."
        )

    search_hi = min(close_idx, best_start + window + 1)
    return int(best_start + np.argmax(z[best_start:search_hi]))


def choose_start_by_close_hover_offset(
    z: np.ndarray,
    descent_start: int,
    close_idx: int,
    hover_z_offset: float,
) -> tuple[int, float, float]:
    lo = max(0, int(descent_start))
    hi = min(len(z), int(close_idx) + 1)
    close_z = float(z[int(close_idx)])
    target_start_z = close_z + float(hover_z_offset)
    local = np.abs(z[lo:hi] - target_start_z)
    return int(lo + np.argmin(local)), close_z, target_start_z


def get_expert_array_by_source(qpos, actions, source: str) -> np.ndarray:
    if source == "qpos":
        return np.asarray(qpos, dtype=np.float32)
    if source == "action":
        return np.asarray(actions, dtype=np.float32)
    raise ValueError(
        f"Unsupported expert auto source '{source}'. Use 'qpos' or 'action'."
    )


class ExpertReplay:
    def __init__(
        self,
        path: str,
        start_idx: int = 0,
        end_idx: int | None = None,
        image_key_map: dict[str, str] | None = None,
    ):
        import h5py

        self.path = path
        self._h5 = h5py.File(path, "r")
        self.qpos = self._h5["observations/qpos"]
        self.actions = self._h5["action"]
        self.images_group = self._h5["observations/images"]
        self.image_key_map = image_key_map or {"front": "cam_front", "wrist": "cam_wrist"}

        self.n_frames = min(len(self.qpos), len(self.actions))
        self.set_range(start_idx, end_idx)
        print(
            f"[ExpertReplay] Loaded {path}, range=[{self.start_idx}, {self.end_idx}), "
            f"image_map={self.image_key_map}"
        )

    def set_range(self, start_idx: int, end_idx: int | None):
        self.start_idx = max(0, int(start_idx))
        self.end_idx = self.n_frames if end_idx is None else min(self.n_frames, int(end_idx))
        if self.start_idx >= self.end_idx:
            raise ValueError(
                f"Invalid expert replay range [{self.start_idx}, {self.end_idx}) for {self.path}"
            )
        self.index = self.start_idx
        self.start_qpos = np.asarray(self.qpos[self.start_idx], dtype=np.float32)

    def configure_prehover_segment(self, cfg: AdaptationConfig):
        z_arr = get_expert_array_by_source(self.qpos, self.actions, cfg.expert_auto_z_source)
        close_arr = get_expert_array_by_source(
            self.qpos,
            self.actions,
            cfg.expert_auto_close_source,
        )

        close_idx = first_close_idx(close_arr[:, 7])
        descent_start = find_main_descent_start(
            z_arr[:, 2],
            close_idx=close_idx,
            window=cfg.expert_auto_window,
            min_drop=cfg.expert_auto_min_drop,
            min_negative_steps=cfg.expert_auto_min_negative_steps,
        )
        start_idx, close_z, target_start_z = choose_start_by_close_hover_offset(
            z_arr[:, 2],
            descent_start=descent_start,
            close_idx=close_idx,
            hover_z_offset=cfg.hover_z_offset,
        )
        end_idx = min(self.n_frames, close_idx + cfg.expert_auto_post_close_frames)
        self.set_range(start_idx, end_idx)

        print(
            "[ExpertReplay] Pre-hover auto segment "
            f"z_source={cfg.expert_auto_z_source}, "
            f"close_source={cfg.expert_auto_close_source}, "
            f"descent_start={descent_start}, close_idx={close_idx}, "
            f"close_z={close_z:.4f}, hover_z_offset={cfg.hover_z_offset:.4f}, "
            f"target_start_z={target_start_z:.4f}, range=[{self.start_idx}, {self.end_idx}), "
            f"start_z={float(z_arr[self.start_idx, 2]):.4f}"
        )

    def reset(self):
        self.index = self.start_idx

    def done(self) -> bool:
        return self.index >= self.end_idx

    def next_observation(self) -> tuple[dict, np.ndarray, float]:
        if self.done():
            raise StopIteration

        images = {}
        for out_key, h5_key in self.image_key_map.items():
            images[out_key] = np.asarray(self.images_group[h5_key][self.index])

        qpos = np.asarray(self.qpos[self.index], dtype=np.float32)
        gripper_binary = float(qpos[7])
        gripper_width = GRIPPER_WIDTH_MIN if gripper_binary >= 0.5 else GRIPPER_WIDTH_MAX
        self.index += 1
        return images, qpos[:7].copy(), gripper_width

    def close(self):
        self._h5.close()


def position_error(current_pose: np.ndarray, target_pose: np.ndarray) -> tuple[float, float]:
    err = np.asarray(target_pose[:3], dtype=np.float32) - np.asarray(current_pose[:3], dtype=np.float32)
    return float(np.linalg.norm(err[:2])), float(abs(err[2]))


def step_pose_action(
    current_pose: np.ndarray,
    target_pose: np.ndarray,
    gripper_width: float,
    kp: float,
    max_step: float,
) -> np.ndarray:
    current_pose = np.asarray(current_pose, dtype=np.float32).reshape(-1)
    target_pose = np.asarray(target_pose, dtype=np.float32).reshape(-1)
    if target_pose.shape[0] < 3:
        raise ValueError("target_pose must contain at least xyz")

    action_pose = current_pose[:7].copy()
    err = target_pose[:3] - current_pose[:3]
    delta = np.clip(kp * err, -max_step, max_step)
    action_pose[:3] = current_pose[:3] + delta
    if target_pose.shape[0] >= 7:
        action_pose[3:7] = target_pose[3:7]
    return np.append(action_pose, float(gripper_width)).astype(np.float32)


def query_policy(sock, images, pose, gripper, jpeg_quality, label: str):
    t_net0 = time.time()
    sock.send(
        pack_observation(
            images,
            pose,
            gripper,
            jpeg_quality=jpeg_quality,
        )
    )
    chunk_data = _recv_reply(sock, label)
    t_net1 = time.time()
    action_buf, done = unpack_chunk(chunk_data)
    print(f"  [{label}] new chunk, net={1000 * (t_net1 - t_net0):.1f}ms")
    return action_buf, done


def reset_policy(sock, cameras, robot, jpeg_quality, label: str):
    images = cameras.capture()
    pose = robot.get_cartesian_pose()
    gripper = robot.get_gripper_state()
    sock.send(pack_observation(images, pose, gripper, reset=True, jpeg_quality=jpeg_quality))
    _recv_reply(sock, label)


def main():
    parser = argparse.ArgumentParser(
        description="ROS bridge for original ACT 10 Hz qpos binary inference with pre-hover expert segment"
    )
    parser.add_argument("--server", type=str, default="tcp://localhost:5555")
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--hz", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    camera_configs = DEFAULT_CAMERAS
    home_pose = HOME_POSE.copy()
    workspace = DEFAULT_WORKSPACE.copy()
    jpeg_quality = 70
    frame_id = POSE_FRAME_ID
    gripper_feedback_mode = "width"
    gripper_command_mode = "width"
    camera_source = "ros_topics"
    adapt_cfg = AdaptationConfig()

    if args.config:
        import yaml

        with open(args.config) as f:
            cfg = yaml.safe_load(f)

        bridge_cfg = _get_bridge_config(cfg)
        robot_cfg = cfg.get("robot", {})
        adapt_cfg = load_adaptation_config(cfg)
        args.server = bridge_cfg.get("server_address", args.server)
        args.hz = bridge_cfg.get("hz", args.hz)
        args.max_steps = bridge_cfg.get("max_steps", args.max_steps)
        args.num_episodes = bridge_cfg.get("num_episodes", args.num_episodes)
        jpeg_quality = bridge_cfg.get("jpeg_quality", jpeg_quality)
        frame_id = bridge_cfg.get("pose_frame_id", frame_id)
        gripper_feedback_mode = bridge_cfg.get(
            "gripper_feedback_mode", gripper_feedback_mode
        )
        gripper_command_mode = bridge_cfg.get(
            "gripper_command_mode", gripper_command_mode
        )
        camera_source = bridge_cfg.get("camera_source", camera_source)

        if "cameras" in cfg:
            camera_configs = cfg["cameras"]
        if "home_cartesian_pose" in robot_cfg:
            home_pose = np.asarray(robot_cfg["home_cartesian_pose"], dtype=np.float32)
        if "workspace_xyz" in robot_cfg:
            workspace = np.asarray(robot_cfg["workspace_xyz"], dtype=np.float32)

    dt = 1.0 / args.hz
    object_pose_provider = None

    if args.dummy:
        cameras = DummyCamera(camera_configs)
        robot = DummyCartesianRobot(
            home_pose=home_pose,
            workspace=workspace,
            gripper_feedback_mode=gripper_feedback_mode,
            gripper_command_mode=gripper_command_mode,
        )
        if adapt_cfg.enabled:
            object_pose_provider = ObjectPoseProvider(adapt_cfg, dummy=True)
    else:
        import rospy

        rospy.init_node(
            "ros_bridge_original_act_qpos_10hz_binary_adaptation_prehover_segment", anonymous=True
        )
        if adapt_cfg.enabled:
            object_pose_provider = ObjectPoseProvider(adapt_cfg, dummy=False)
        if camera_source == "ros_topics":
            cameras = ROSTopicCameraManager(camera_configs)
        elif camera_source == "realsense":
            cameras = CameraManager(camera_configs)
        else:
            raise ValueError(f"Unsupported camera_source: {camera_source}")
        robot = ROSCartesianRobot(
            home_pose=home_pose,
            workspace=workspace,
            frame_id=frame_id,
            gripper_feedback_mode=gripper_feedback_mode,
            gripper_command_mode=gripper_command_mode,
        )
        robot.wait_for_state()
        robot.activate_gripper()
        if hasattr(cameras, "wait_until_ready"):
            cameras.wait_until_ready()

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, 1000)
    sock.connect(args.server)
    print(f"[Bridge] Connected to policy server: {args.server}")
    print(
        "[Bridge] Data contract: 10 Hz observation, absolute Cartesian target, "
        "gripper feedback/command transported as width"
    )
    if adapt_cfg.enabled:
        print("[Adaptation] Enabled")
        print(
            "[Adaptation] states: POLICY_APPROACH -> HOVER_ALIGN -> "
            "EXPERT_DESCEND_GRASP -> MOVE_TO_IN_DIST_POSE -> POLICY_PLACE"
        )
        print(
            "[Adaptation] expert_auto_segment="
            f"{adapt_cfg.expert_auto_segment}, "
            f"z_source={adapt_cfg.expert_auto_z_source}, "
            f"close_source={adapt_cfg.expert_auto_close_source}, "
            f"start_by=close_z+hover_z_offset"
        )

    try:
        for ep in range(args.num_episodes):
            input(f"\n[Bridge] Episode {ep + 1}/{args.num_episodes} - Press Enter to start...")
            robot.go_home()

            images = cameras.capture()
            pose = robot.get_cartesian_pose()
            gripper = robot.get_gripper_state()
            sock.send(pack_observation(images, pose, gripper, reset=True, jpeg_quality=jpeg_quality))
            _recv_reply(sock, "episode reset")

            print(
                f"[Bridge] Running episode {ep + 1} "
                f"(max {args.max_steps} steps, {args.hz} Hz)"
            )

            action_buf = None
            buf_idx = 0
            done = False
            adapt_state = "POLICY_APPROACH" if adapt_cfg.enabled else "POLICY"
            expert_replay = None
            expert_live_start_pose = None
            expert_close_count = 0
            expert_segment_ready = False

            if adapt_cfg.enabled:
                if not adapt_cfg.expert_hdf5_path:
                    raise ValueError("adaptation.expert_hdf5_path is required")
                if adapt_cfg.fixed_place_entry_pose is None:
                    raise ValueError("adaptation.fixed_place_entry_pose is required")
                expert_replay = ExpertReplay(
                    adapt_cfg.expert_hdf5_path,
                    start_idx=adapt_cfg.expert_start_idx,
                    end_idx=adapt_cfg.expert_end_idx,
                )

            for step in range(args.max_steps):
                t0 = time.time()

                pose = robot.get_cartesian_pose()
                gripper = robot.get_gripper_state()

                if not adapt_cfg.enabled:
                    if action_buf is None or buf_idx >= len(action_buf):
                        images = cameras.capture()
                        action_buf, done = query_policy(
                            sock,
                            images,
                            pose,
                            gripper,
                            jpeg_quality,
                            f"action chunk at step {step}",
                        )
                        buf_idx = 0

                    action = action_buf[buf_idx]
                    buf_idx += 1
                    robot.execute_action(action)

                else:
                    object_pose = object_pose_provider.get() if object_pose_provider else None

                    if adapt_state == "POLICY_APPROACH":
                        if object_pose is not None:
                            dist_xy = float(np.linalg.norm(pose[:2] - object_pose[:2]))
                            if dist_xy < adapt_cfg.approach_xy_threshold:
                                if adapt_cfg.expert_auto_segment and not expert_segment_ready:
                                    expert_replay.configure_prehover_segment(adapt_cfg)
                                    expert_segment_ready = True
                                adapt_state = "HOVER_ALIGN"
                                action_buf = None
                                buf_idx = 0
                                print(
                                    f"  [step {step:4d}] -> HOVER_ALIGN "
                                    f"(dist_xy={dist_xy:.4f})"
                                )

                        if adapt_state == "POLICY_APPROACH":
                            if action_buf is None or buf_idx >= len(action_buf):
                                images = cameras.capture()
                                action_buf, done = query_policy(
                                    sock,
                                    images,
                                    pose,
                                    gripper,
                                    jpeg_quality,
                                    f"policy approach step {step}",
                                )
                                buf_idx = 0
                            action = action_buf[buf_idx]
                            buf_idx += 1
                            robot.execute_action(action)

                    if adapt_state == "HOVER_ALIGN":
                        if object_pose is None:
                            print("  [Adaptation] Waiting for object pose before hover alignment...")
                            action = np.append(pose, adapt_cfg.open_width)
                            robot.execute_action(action)
                        else:
                            hover_target = pose.copy()
                            hover_target[:3] = object_pose[:3]
                            hover_target[2] += adapt_cfg.hover_z_offset
                            action = step_pose_action(
                                pose,
                                hover_target,
                                adapt_cfg.open_width,
                                adapt_cfg.hover_kp,
                                adapt_cfg.hover_max_step,
                            )
                            robot.execute_action(action)

                            err_xy, err_z = position_error(pose, hover_target)
                            if err_xy < adapt_cfg.hover_xy_tolerance and err_z < adapt_cfg.hover_z_tolerance:
                                adapt_state = "EXPERT_DESCEND_GRASP"
                                expert_live_start_pose = robot.get_cartesian_pose()
                                expert_replay.reset()
                                expert_close_count = 0
                                action_buf = None
                                buf_idx = 0
                                print(
                                    f"  [step {step:4d}] -> EXPERT_DESCEND_GRASP "
                                    f"(err_xy={err_xy:.4f}, err_z={err_z:.4f})"
                                )

                    elif adapt_state == "EXPERT_DESCEND_GRASP":
                        if expert_replay.done():
                            adapt_state = "MOVE_TO_IN_DIST_POSE"
                            action_buf = None
                            buf_idx = 0
                            print(f"  [step {step:4d}] -> MOVE_TO_IN_DIST_POSE (expert replay ended)")
                            action = np.append(pose, adapt_cfg.closed_width)
                            robot.execute_action(action)
                        else:
                            expert_images, expert_pose, expert_gripper = expert_replay.next_observation()
                            images = expert_images if adapt_cfg.expert_use_images else cameras.capture()
                            policy_pose = expert_pose if adapt_cfg.expert_use_qpos else pose
                            policy_gripper = expert_gripper if adapt_cfg.expert_use_qpos else gripper
                            expert_chunk, _ = query_policy(
                                sock,
                                images,
                                policy_pose,
                                policy_gripper,
                                jpeg_quality,
                                f"expert descend step {step}",
                            )
                            expert_action = expert_chunk[0].copy()

                            action = expert_action.copy()
                            live_pose = robot.get_cartesian_pose()
                            object_pose = object_pose_provider.get() if object_pose_provider else object_pose

                            if adapt_cfg.expert_follow_xy_object and object_pose is not None:
                                xy_err = object_pose[:2] - live_pose[:2]
                                xy_delta = np.clip(
                                    adapt_cfg.hover_kp * xy_err,
                                    -adapt_cfg.hover_max_step,
                                    adapt_cfg.hover_max_step,
                                )
                                action[:2] = live_pose[:2] + xy_delta

                            if adapt_cfg.expert_z_relative:
                                expert_delta_z = expert_action[2] - expert_replay.start_qpos[2]
                                action[2] = expert_live_start_pose[2] + expert_delta_z

                            # Keep live orientation during adapted descent to avoid
                            # large absolute-pose jumps when the object is out of range.
                            action[3:7] = live_pose[3:7]
                            robot.execute_action(action)

                            if len(action) > 7 and action[7] < adapt_cfg.expert_close_width_threshold:
                                expert_close_count += 1
                            else:
                                expert_close_count = 0
                            if expert_close_count >= 3:
                                adapt_state = "MOVE_TO_IN_DIST_POSE"
                                action_buf = None
                                buf_idx = 0
                                print(f"  [step {step:4d}] -> MOVE_TO_IN_DIST_POSE (gripper closed)")

                    elif adapt_state == "MOVE_TO_IN_DIST_POSE":
                        target_pose = adapt_cfg.fixed_place_entry_pose
                        action = step_pose_action(
                            pose,
                            target_pose,
                            adapt_cfg.closed_width,
                            adapt_cfg.carry_kp,
                            adapt_cfg.carry_max_step,
                        )
                        robot.execute_action(action)
                        err_xy, err_z = position_error(pose, target_pose)
                        if err_xy < adapt_cfg.place_entry_tolerance and err_z < adapt_cfg.place_entry_tolerance:
                            if adapt_cfg.policy_reset_after_manual:
                                reset_policy(
                                    sock,
                                    cameras,
                                    robot,
                                    jpeg_quality,
                                    "adapt reset before policy place",
                                )
                            adapt_state = "POLICY_PLACE"
                            action_buf = None
                            buf_idx = 0
                            print(
                                f"  [step {step:4d}] -> POLICY_PLACE "
                                f"(err_xy={err_xy:.4f}, err_z={err_z:.4f})"
                            )

                    elif adapt_state == "POLICY_PLACE":
                        if action_buf is None or buf_idx >= len(action_buf):
                            images = cameras.capture()
                            action_buf, done = query_policy(
                                sock,
                                images,
                                pose,
                                gripper,
                                jpeg_quality,
                                f"policy place step {step}",
                            )
                            buf_idx = 0
                        action = action_buf[buf_idx]
                        buf_idx += 1
                        robot.execute_action(action)

                    else:
                        raise RuntimeError(f"Unknown adaptation state: {adapt_state}")

                elapsed = time.time() - t0
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                actual_hz = 1.0 / max(time.time() - t0, 1e-6)
                if step % 50 == 0:
                    print(
                        f"  [step {step:4d}] pose={np.array2string(pose[:3], precision=3, suppress_small=True)} "
                        f"state={adapt_state} "
                        f"gripper={gripper:.4f} "
                        f"action_xyz={np.array2string(action[:3], precision=4, suppress_small=True)} "
                        f"action_gripper={float(action[7]) if len(action) > 7 else float('nan'):.4f} "
                        f"hz={actual_hz:.1f}"
                    )

                if done:
                    print(f"  [step {step}] Policy signaled done")
                    break

            nsteps = step + 1 if args.max_steps > 0 else 0
            if expert_replay is not None:
                expert_replay.close()
            print(f"[Bridge] Episode {ep + 1} finished ({nsteps} steps)")
            robot.open_gripper()

            images = cameras.capture()
            pose = robot.get_cartesian_pose()
            gripper = robot.get_gripper_state()
            sock.send(pack_observation(images, pose, gripper, reset=True, jpeg_quality=jpeg_quality))
            _recv_reply(sock, "episode done")
            print(f"[Bridge] Episode {ep + 1} done signal sent -> server saving data")

    except KeyboardInterrupt:
        print("\n[Bridge] Interrupted")
    finally:
        robot.stop()
        cameras.stop()
        sock.close()
        ctx.term()
        print("[Bridge] Shutdown complete")


if __name__ == "__main__":
    main()
