#!/usr/bin/env python
"""ROS Bridge: ZMQ client that runs on the ROS machine (KUKA + RealSense cameras).

Captures images, reads robot state, sends observations to the policy server,
receives actions, and executes them on the robot.

This file is **self-contained** — no imports from the rest of the project.
Copy it (+ a small config YAML) to the ROS machine via scp.

Dependencies on ROS machine:
    pip install zmq msgpack opencv-python numpy pyrealsense2 pyyaml

Usage (on ROS machine):
    source /opt/ros/noetic/setup.bash
    source ~/catkin_ws/devel/setup.bash
    python ros_bridge.py --server tcp://192.168.x.x:5555
    python ros_bridge.py --server tcp://192.168.x.x:5555 --dummy   # no hardware

Usage (dummy test on any machine, no ROS needed):
    python ros_bridge.py --dummy --server tcp://localhost:5555
"""
from __future__ import annotations

import argparse
import struct
import time

import cv2
import numpy as np
import msgpack
import zmq

# ============================================================================
# Embedded msg_utils  (from msg_utils.py — kept in sync)
# ============================================================================

def encode_image(image: np.ndarray, quality: int = 90) -> bytes:
    """Encode RGB image to JPEG bytes."""
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                          [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def decode_image(data: bytes) -> np.ndarray:
    """Decode JPEG bytes to RGB numpy array."""
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def encode_depth(depth: np.ndarray) -> bytes:
    """Encode uint16 depth image to PNG bytes."""
    _, buf = cv2.imencode(".png", depth)
    return buf.tobytes()


def _np_to_bytes(arr: np.ndarray) -> bytes:
    """Serialize numpy array: 1-byte dtype_len + dtype + 4-byte ndim + shape + data."""
    dtype_str = arr.dtype.str.encode()
    header = struct.pack("B", len(dtype_str)) + dtype_str
    header += struct.pack("<I", arr.ndim)
    for s in arr.shape:
        header += struct.pack("<I", s)
    return header + arr.tobytes()


def _bytes_to_np(data: bytes) -> np.ndarray:
    """Deserialize numpy array from bytes."""
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
    """Pack observation into msgpack bytes."""
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


def unpack_action(data: bytes) -> tuple:
    """Unpack msgpack bytes into (action_array, done_bool)."""
    msg = msgpack.unpackb(data, raw=False)
    return _bytes_to_np(msg["action"]), msg["done"]


def unpack_chunk(data: bytes) -> tuple:
    """Unpack msgpack bytes into (actions_array (N, 8), done_bool)."""
    msg = msgpack.unpackb(data, raw=False)
    return _bytes_to_np(msg["actions"]), msg["done"]


# ============================================================================
# Embedded camera capture  (from camera.py — kept in sync)
# ============================================================================

class CameraManager:
    """Manages multiple RealSense cameras."""

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

            w = cfg.get("width", 640)
            h = cfg.get("height", 480)
            fps = cfg.get("fps", 30)
            mode = cfg.get("mode", "rgb")

            rs_config.enable_stream(rs.stream.color, w, h, rs.format.rgb8, fps)
            if mode == "rgbd":
                rs_config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)

            pipeline.start(rs_config)
            self.pipelines.append(pipeline)
            print(f"[Camera] Started '{cfg['name']}' "
                  f"(serial={serial or 'auto'}, {w}x{h}, {mode})")

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
    """Fake camera for testing without hardware."""

    def __init__(self, configs: list[dict]):
        self.configs = configs
        print(f"[DummyCamera] Initialized {len(configs)} virtual cameras")

    def capture(self) -> dict:
        result = {}
        for cfg in self.configs:
            w = cfg.get("width", 640)
            h = cfg.get("height", 480)
            result[cfg["name"]] = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
            if cfg.get("mode", "rgb") == "rgbd":
                result[f"{cfg['name']}_depth"] = np.random.randint(
                    0, 5000, (h, w), dtype=np.uint16)
        return result

    def stop(self):
        print("[DummyCamera] Stopped.")


# ============================================================================
# ROS robot interface
# ============================================================================

# iiwa_stack topic / config — MODIFY THESE IF YOUR SETUP IS DIFFERENT
JOINT_STATE_TOPIC = "/iiwa/state/JointPosition"       # iiwa_msgs/JointPosition
JOINT_CMD_TOPIC = "/iiwa/command/JointPosition"      # iiwa_msgs/JointPosition
HOME_POSITION = [0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, 0.0]  # 7 joints (rad)

# Gripper — adapt to your gripper driver
GRIPPER_STATE_TOPIC = "/Robotiq2FGripperRobotInput"   # Robotiq2FGripper_robot_input
GRIPPER_CMD_TOPIC = "/Robotiq2FGripperRobotOutput"    # Robotiq2FGripper_robot_output


class ROSRobot:
    """Read joint states / send commands via ROS topics (requires rospy)."""

    def __init__(self, home_position=None):
        import threading
        import rospy
        from iiwa_msgs.msg import JointPosition
        from robotiq_2f_gripper_control.msg import (
            Robotiq2FGripper_robot_input,
            Robotiq2FGripper_robot_output,
        )

        self._rospy = rospy
        self._JointPosition = JointPosition
        self._GripperOutput = Robotiq2FGripper_robot_output

        self._joint_positions = np.zeros(7, dtype=np.float64)
        self._gripper_state = -1.0  # default open
        self._lock = threading.Lock()
        self._home = np.array(home_position or HOME_POSITION, dtype=np.float64)

        rospy.Subscriber(JOINT_STATE_TOPIC, JointPosition, self._joint_cb)
        self._cmd_pub = rospy.Publisher(JOINT_CMD_TOPIC, JointPosition, queue_size=1)

        if GRIPPER_STATE_TOPIC:
            rospy.Subscriber(GRIPPER_STATE_TOPIC, Robotiq2FGripper_robot_input,
                             self._gripper_cb)
        if GRIPPER_CMD_TOPIC:
            self._gripper_pub = rospy.Publisher(
                GRIPPER_CMD_TOPIC, Robotiq2FGripper_robot_output, queue_size=1)

        rospy.loginfo(f"[ROSRobot] Subscribed to {JOINT_STATE_TOPIC}")
        rospy.loginfo(f"[ROSRobot] Publishing to {JOINT_CMD_TOPIC}")

        self._gripper_sta = 0   # track gSTA from feedback
        self._gripper_flt = 0   # track gFLT from feedback

    def _joint_cb(self, msg):
        with self._lock:
            pos = msg.position
            self._joint_positions = np.array(
                [pos.a1, pos.a2, pos.a3, pos.a4, pos.a5, pos.a6, pos.a7],
                dtype=np.float64,
            )

    def _gripper_cb(self, msg):
        # gPO: 0 (open) → -1.0, 255 (closed) → 1.0
        with self._lock:
            self._gripper_state = (msg.gPO / 255.0) * 2.0 - 1.0
            self._gripper_sta = msg.gSTA
            self._gripper_flt = msg.gFLT

    def get_joint_positions(self) -> np.ndarray:
        with self._lock:
            return self._joint_positions.astype(np.float32).copy()

    def get_gripper_state(self) -> float:
        with self._lock:
            return self._gripper_state

    def execute_action(self, action: np.ndarray):
        """Send joint command. action = [7 joints (+ optional gripper)]."""
        msg = self._JointPosition()
        msg.header.stamp = self._rospy.Time.now()
        msg.position.a1 = float(action[0])
        msg.position.a2 = float(action[1])
        msg.position.a3 = float(action[2])
        msg.position.a4 = float(action[3])
        msg.position.a5 = float(action[4])
        msg.position.a6 = float(action[5])
        msg.position.a7 = float(action[6])
        self._cmd_pub.publish(msg)
        # Gripper command: action[7] in [-1, 1] → rPR in [0, 255]
        if len(action) > 7 and GRIPPER_CMD_TOPIC:
            grip_val = float(action[7])  # -1 (open) to 1 (closed)
            rPR = int(np.clip((grip_val + 1.0) / 2.0 * 255.0, 0, 255))
            msg_g = self._GripperOutput()
            msg_g.rACT = 1
            msg_g.rGTO = 1
            msg_g.rSP = 255   # max speed
            msg_g.rFR = 150   # moderate force
            msg_g.rPR = rPR
            self._gripper_pub.publish(msg_g)

    def activate_gripper(self):
        """Run Robotiq 2F activation sequence. Must be called before position commands."""
        if not GRIPPER_CMD_TOPIC:
            return
        rospy = self._rospy
        rospy.loginfo("[ROSRobot] Gripper activation: reset...")
        # Step 1: reset (rACT=0)
        msg = self._GripperOutput()
        msg.rACT = 0
        for _ in range(5):
            self._gripper_pub.publish(msg)
            time.sleep(0.1)

        # Step 2: activate (rACT=1, rGTO=0)
        rospy.loginfo("[ROSRobot] Gripper activation: activating...")
        msg = self._GripperOutput()
        msg.rACT = 1
        msg.rGTO = 0
        msg.rSP = 255
        msg.rFR = 150
        for _ in range(5):
            self._gripper_pub.publish(msg)
            time.sleep(0.1)

        # Step 3: wait until gSTA == 3 (activation complete)
        deadline = time.time() + 10.0
        while time.time() < deadline:
            with self._lock:
                sta = self._gripper_sta
                flt = self._gripper_flt
            rospy.loginfo(f"[ROSRobot] Gripper gSTA={sta} gFLT={flt}")
            if sta == 3:
                rospy.loginfo("[ROSRobot] Gripper activated (gSTA=3)")
                break
            time.sleep(0.2)
        else:
            rospy.logwarn(f"[ROSRobot] Gripper activation timeout! gSTA={sta} gFLT={flt}")

        # Step 4: open gripper
        rospy.loginfo("[ROSRobot] Gripper activation: opening...")
        msg = self._GripperOutput()
        msg.rACT = 1
        msg.rGTO = 1
        msg.rSP = 255
        msg.rFR = 150
        msg.rPR = 0  # fully open
        for _ in range(5):
            self._gripper_pub.publish(msg)
            time.sleep(0.1)
        rospy.loginfo("[ROSRobot] Gripper activation complete")

    def go_home(self):
        action = np.append(self._home, -1.0)  # -1.0 = open gripper
        self.execute_action(action)
        self._rospy.loginfo("[ROSRobot] Going home + opening gripper")

    def stop(self):
        state = self.get_joint_positions()
        self.execute_action(state)
        self._rospy.loginfo("[ROSRobot] Stop — holding current position")


class DummyRobot:
    """Fake robot for testing without ROS or hardware."""

    def __init__(self, home_position=None):
        self._joints = np.zeros(7, dtype=np.float32)
        self._gripper = 0.0
        self._home = np.array(home_position or HOME_POSITION, dtype=np.float32)
        print("[DummyRobot] Initialized")

    def get_joint_positions(self) -> np.ndarray:
        return self._joints.copy()

    def get_gripper_state(self) -> float:
        return self._gripper

    def execute_action(self, action: np.ndarray):
        self._joints = action[:7].astype(np.float32)
        if len(action) > 7:
            self._gripper = float(action[7])

    def go_home(self):
        self._joints = self._home.copy()
        self._gripper = -1.0  # open
        print("[DummyRobot] Moved to home position, gripper open")

    def stop(self):
        print("[DummyRobot] Stopped")


# ============================================================================
# Main loop
# ============================================================================

DEFAULT_CAMERAS = [
    {"name": "base_camera",  "serial": "308222301757", "width": 640, "height": 480, "fps": 30, "mode": "rgb"},  # D455
    {"name": "wrist_camera", "serial": "306322300452", "width": 640, "height": 480, "fps": 30, "mode": "rgb"},  # D456
]


def main():
    parser = argparse.ArgumentParser(
        description="ROS Bridge: capture cameras + robot state, send to policy server")
    parser.add_argument("--server", type=str, default="tcp://localhost:5555",
                        help="Policy server ZMQ address (e.g. tcp://192.168.1.50:5555)")
    parser.add_argument("--dummy", action="store_true",
                        help="Use DummyRobot + DummyCamera (no hardware / no ROS)")
    parser.add_argument("--hz", type=int, default=15,
                        help="Control loop frequency")
    parser.add_argument("--max-steps", type=int, default=300,
                        help="Max steps per episode")
    parser.add_argument("--num-episodes", type=int, default=10,
                        help="Number of episodes to run")
    parser.add_argument("--config", type=str, default=None,
                        help="Optional YAML config (overrides CLI args for cameras etc.)")
    args = parser.parse_args()

    # ---- Load optional config ----
    camera_configs = DEFAULT_CAMERAS
    home_position = HOME_POSITION
    if args.config:
        import yaml
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        ros_cfg = cfg.get("ros_bridge", {})
        args.server = ros_cfg.get("server_address", args.server)
        args.hz = ros_cfg.get("hz", args.hz)
        args.max_steps = ros_cfg.get("max_steps", args.max_steps)
        args.num_episodes = ros_cfg.get("num_episodes", args.num_episodes)
        args.jpeg_quality = ros_cfg.get("jpeg_quality", 70)
        if "cameras" in cfg:
            camera_configs = cfg["cameras"]
        if "robot" in cfg:
            home_position = cfg["robot"].get("home_position", HOME_POSITION)

    dt = 1.0 / args.hz
    jpeg_quality = getattr(args, "jpeg_quality", 70)

    # ---- Init hardware ----
    if args.dummy:
        cameras = DummyCamera(camera_configs)
        robot = DummyRobot(home_position)
    else:
        import rospy
        rospy.init_node("ros_bridge", anonymous=True)
        cameras = CameraManager(camera_configs)
        robot = ROSRobot(home_position)
        # Wait for first joint state + gripper feedback
        print("[Bridge] Waiting for first joint state from ROS...")
        time.sleep(1.0)
        robot.activate_gripper()

    # ---- ZMQ setup (REQ client → server REP) ----
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(args.server)
    print(f"[Bridge] Connected to policy server: {args.server}")

    # ---- Episode loop ----
    try:
        for ep in range(args.num_episodes):
            input(f"\n[Bridge] Episode {ep+1}/{args.num_episodes} "
                  f"— Press Enter to start...")
            robot.go_home()

            # Wait until robot reaches home position
            home = np.array(HOME_POSITION, dtype=np.float32)
            print(f"[Bridge] Moving to home {home.tolist()} ...")
            while True:
                joints = robot.get_joint_positions()
                err = np.max(np.abs(joints - home))
                if err < 0.05:
                    print(f"[Bridge] Home reached (max_err={err:.4f} rad)")
                    break
                time.sleep(0.1)

            # Explicitly open gripper after reaching home (send multiple times to avoid drops)
            open_action = np.append(home, -1.0)
            for i in range(5):
                robot.execute_action(open_action)
                time.sleep(0.05)
            gripper_after = robot.get_gripper_state()
            print(f"[Bridge] Gripper open cmd sent x5, gripper_state={gripper_after:.3f} (expect ~-1.0)")

            # Send reset signal
            images = cameras.capture()
            joints = robot.get_joint_positions()
            gripper = robot.get_gripper_state()
            sock.send(pack_observation(images, joints, gripper, reset=True))
            sock.recv()  # consume reset ack

            print(f"[Bridge] Running episode {ep+1} "
                  f"(max {args.max_steps} steps, {args.hz} Hz)")

            # Local action buffer — only query server when empty
            action_buf = None
            buf_idx = 0
            done = False

            for step in range(args.max_steps):
                t0 = time.time()

                # 1. Always read robot state
                joints = robot.get_joint_positions()
                gripper = robot.get_gripper_state()

                # 2. Query server only when buffer is empty
                if action_buf is None or buf_idx >= len(action_buf):
                    images = cameras.capture()
                    t_net0 = time.time()
                    sock.send(pack_observation(images, joints, gripper,
                                              jpeg_quality=jpeg_quality))
                    chunk_data = sock.recv()
                    t_net1 = time.time()
                    action_buf, done = unpack_chunk(chunk_data)
                    buf_idx = 0
                    print(f"  [step {step:4d}] new chunk, net={1000*(t_net1-t_net0):.1f}ms")

                # 3. Take action from local buffer
                action = action_buf[buf_idx]
                buf_idx += 1

                # 4. Execute
                robot.execute_action(action)

                # 5. Rate limiting
                elapsed = time.time() - t0
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                actual_hz = 1.0 / max(time.time() - t0, 1e-6)
                if step % 50 == 0:
                    joints_str = np.array2string(joints, precision=3, suppress_small=True)
                    print(f"  [step {step:4d}] joints={joints_str} gripper={gripper:.3f} "
                          f"hz={actual_hz:.1f}")

                if done:
                    print(f"  [step {step}] Policy signaled done")
                    break

            nsteps = step + 1 if args.max_steps > 0 else 0
            print(f"[Bridge] Episode {ep+1} finished ({nsteps} steps)")

            # Notify server that episode is done → triggers immediate save
            images = cameras.capture()
            joints = robot.get_joint_positions()
            gripper = robot.get_gripper_state()
            sock.send(pack_observation(images, joints, gripper, reset=True))
            sock.recv()  # consume ack
            print(f"[Bridge] Episode {ep+1} done signal sent → server saving data")

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
