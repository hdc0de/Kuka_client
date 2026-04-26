#!/usr/bin/env python3
"""ROS bridge for the original ACT Cartesian-pose policy.

This is a self-contained bridge that runs on the ROS machine. It keeps the
same ZMQ protocol as the existing bridge, but packs Cartesian pose into the
`joint_positions` field for compatibility with the current server/msg format.

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
    msg.rPR = 255 if width < 0.045 else 0
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
        self.execute_action(np.append(pose, GRIPPER_WIDTH_MAX))
        self._rospy.loginfo("[ROSCartesianRobot] Open gripper at current pose")


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
    return cfg.get("ros_bridge_original_act", cfg.get("ros_bridge", {}))


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


def main():
    parser = argparse.ArgumentParser(
        description="ROS bridge for original ACT Cartesian-pose inference"
    )
    parser.add_argument("--server", type=str, default="tcp://localhost:5555")
    parser.add_argument("--dummy", action="store_true")
    parser.add_argument("--hz", type=int, default=15)
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

    if args.config:
        import yaml

        with open(args.config) as f:
            cfg = yaml.safe_load(f)

        bridge_cfg = _get_bridge_config(cfg)
        robot_cfg = cfg.get("robot", {})
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

    if args.dummy:
        cameras = DummyCamera(camera_configs)
        robot = DummyCartesianRobot(
            home_pose=home_pose,
            workspace=workspace,
            gripper_feedback_mode=gripper_feedback_mode,
            gripper_command_mode=gripper_command_mode,
        )
    else:
        import rospy

        rospy.init_node("ros_bridge_original_act", anonymous=True)
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

            for step in range(args.max_steps):
                t0 = time.time()

                pose = robot.get_cartesian_pose()
                gripper = robot.get_gripper_state()

                if action_buf is None or buf_idx >= len(action_buf):
                    images = cameras.capture()
                    t_net0 = time.time()
                    sock.send(
                        pack_observation(
                            images,
                            pose,
                            gripper,
                            jpeg_quality=jpeg_quality,
                        )
                    )
                    chunk_data = _recv_reply(sock, f"action chunk at step {step}")
                    t_net1 = time.time()
                    action_buf, done = unpack_chunk(chunk_data)
                    buf_idx = 0
                    print(f"  [step {step:4d}] new chunk, net={1000 * (t_net1 - t_net0):.1f}ms")

                action = action_buf[buf_idx]
                buf_idx += 1
                robot.execute_action(action)

                elapsed = time.time() - t0
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                actual_hz = 1.0 / max(time.time() - t0, 1e-6)
                if step % 50 == 0:
                    print(
                        f"  [step {step:4d}] pose={np.array2string(pose[:3], precision=3, suppress_small=True)} "
                        f"gripper={gripper:.4f} "
                        f"action_xyz={np.array2string(action[:3], precision=4, suppress_small=True)} "
                        f"action_gripper={float(action[7]) if len(action) > 7 else float('nan'):.4f} "
                        f"hz={actual_hz:.1f}"
                    )

                if done:
                    print(f"  [step {step}] Policy signaled done")
                    break

            nsteps = step + 1 if args.max_steps > 0 else 0
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
