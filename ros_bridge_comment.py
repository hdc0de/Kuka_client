#!/usr/bin/env python
"""ROS Bridge: 运行在 ROS 机器上的 ZMQ 客户端（KUKA + RealSense）。

核心职责：
1) 采集多路相机图像
2) 读取机器人关节和夹爪状态
3) 将观测打包发送给策略服务器
4) 接收服务器返回动作并执行到机器人

设计说明：
- 本文件是单文件可迁移脚本，不依赖项目内其它模块
- 便于通过 scp 拷贝到机器人控制机后直接运行
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
# Embedded msg_utils（消息编解码工具）
# ============================================================================
# 这一段负责“观测/动作”的序列化与反序列化，约束了和服务器通信的数据格式。
# 协议要点：
# - 图像：RGB 走 JPEG，深度图走 PNG
# - numpy 数组：自定义头部 + 原始 bytes，避免 msgpack 对 ndarray 的不透明处理
# - 最外层：msgpack 字典结构


def encode_image(image: np.ndarray, quality: int = 90) -> bytes:
    """将 RGB 图编码为 JPEG bytes。

    参数：
    - image: HxWx3，RGB 顺序
    - quality: JPEG 压缩质量（0~100，越高质量越好、体积越大）

    返回：
    - JPEG 字节流
    """
    # OpenCV 的 imencode 默认按 BGR 解释，因此先把 RGB 转 BGR
    _, buf = cv2.imencode(
        ".jpg",
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    return buf.tobytes()


def decode_image(data: bytes) -> np.ndarray:
    """将 JPEG bytes 解码回 RGB numpy 数组。"""
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    # OpenCV 输出 BGR，这里转回 RGB 以统一上层语义
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def encode_depth(depth: np.ndarray) -> bytes:
    """将 uint16 深度图编码为 PNG bytes（无损，适合深度信息）。"""
    _, buf = cv2.imencode(".png", depth)
    return buf.tobytes()


def _np_to_bytes(arr: np.ndarray) -> bytes:
    """把 numpy 数组序列化为 bytes。

    自定义布局：
    - 1 byte: dtype 字符串长度
    - N byte: dtype 字符串（如 <f4）
    - 4 byte: ndim（小端 uint32）
    - ndim * 4 byte: 每个维度大小（小端 uint32）
    - 剩余: 原始数据区（arr.tobytes()）

    这样可以在另一端无损恢复 dtype 和 shape。
    """
    dtype_str = arr.dtype.str.encode()
    header = struct.pack("B", len(dtype_str)) + dtype_str
    header += struct.pack("<I", arr.ndim)
    for s in arr.shape:
        header += struct.pack("<I", s)
    return header + arr.tobytes()


def _bytes_to_np(data: bytes) -> np.ndarray:
    """从 _np_to_bytes 产生的 bytes 中恢复 numpy 数组。"""
    offset = 0

    # 读取 dtype 字符串长度
    dtype_len = struct.unpack("B", data[offset:offset + 1])[0]
    offset += 1

    # 读取 dtype 字符串
    dtype_str = data[offset:offset + dtype_len].decode()
    offset += dtype_len

    # 读取维度数量
    ndim = struct.unpack("<I", data[offset:offset + 4])[0]
    offset += 4

    # 读取 shape
    shape = []
    for _ in range(ndim):
        shape.append(struct.unpack("<I", data[offset:offset + 4])[0])
        offset += 4

    # 用 frombuffer 零拷贝创建，再 copy 一份，避免依赖原 bytes 生命周期
    return np.frombuffer(data[offset:], dtype=np.dtype(dtype_str)).reshape(shape).copy()


def pack_observation(
    images: dict,
    joint_positions: np.ndarray,
    gripper_state: float,
    reset: bool = False,
    jpeg_quality: int = 70,
) -> bytes:
    """把一帧观测打包成 msgpack bytes。

    字段定义：
    - images: dict[name -> encoded_bytes]
    - joint_positions: float32 ndarray bytes
    - gripper_state: float
    - timestamp: 发送端时间戳（秒）
    - reset: 是否是“重置信号帧”
    """
    encoded_images = {}
    for name, img in images.items():
        # 约定：键名以 _depth 结尾即深度图
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
    """解析单步动作包 -> (action_array, done_bool)。"""
    msg = msgpack.unpackb(data, raw=False)
    return _bytes_to_np(msg["action"]), msg["done"]


def unpack_chunk(data: bytes) -> tuple:
    """解析动作块包 -> (actions_array[N,8], done_bool)。

    使用 chunk 的目的是降低网络往返开销：
    客户端只在本地缓存耗尽时才请求服务器。
    """
    msg = msgpack.unpackb(data, raw=False)
    return _bytes_to_np(msg["actions"]), msg["done"]


# ============================================================================
# Embedded camera capture（相机采集）
# ============================================================================
# 包含两个实现：
# - CameraManager：真实 RealSense 多相机
# - DummyCamera：无硬件时随机图像模拟


class CameraManager:
    """管理多台 RealSense 相机。"""

    def __init__(self, configs: list[dict]):
        # 延迟导入：只有真实模式才需要 pyrealsense2
        import pyrealsense2 as rs

        self.rs = rs
        self.configs = configs
        self.pipelines = []
        self._start_cameras()

    def _start_cameras(self):
        """按配置逐个启动相机 pipeline。"""
        rs = self.rs
        for cfg in self.configs:
            pipeline = rs.pipeline()
            rs_config = rs.config()

            # 指定序列号可绑定固定物理相机；为空则自动选择
            serial = cfg.get("serial", "")
            if serial:
                rs_config.enable_device(serial)

            w = cfg.get("width", 640)
            h = cfg.get("height", 480)
            fps = cfg.get("fps", 30)
            mode = cfg.get("mode", "rgb")

            # 彩色流始终启用
            rs_config.enable_stream(rs.stream.color, w, h, rs.format.rgb8, fps)
            # rgbd 模式额外启用深度流
            if mode == "rgbd":
                rs_config.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)

            pipeline.start(rs_config)
            self.pipelines.append(pipeline)

            print(
                f"[Camera] Started '{cfg['name']}' "
                f"(serial={serial or 'auto'}, {w}x{h}, {mode})"
            )

    def capture(self) -> dict:
        """采集所有相机一帧，返回 dict[name -> np.ndarray]。"""
        result = {}
        for cfg, pipeline in zip(self.configs, self.pipelines):
            frames = pipeline.wait_for_frames()

            # 彩色帧
            color_frame = frames.get_color_frame()
            result[cfg["name"]] = np.asarray(color_frame.get_data())

            # 深度帧（可选）
            if cfg.get("mode", "rgb") == "rgbd":
                depth_frame = frames.get_depth_frame()
                result[f"{cfg['name']}_depth"] = np.asarray(depth_frame.get_data())

        return result

    def stop(self):
        """停止所有 pipeline，释放设备资源。"""
        for pipeline in self.pipelines:
            pipeline.stop()
        self.pipelines.clear()
        print("[Camera] All cameras stopped.")


class DummyCamera:
    """无硬件测试用相机：生成随机 RGB/Depth。"""

    def __init__(self, configs: list[dict]):
        self.configs = configs
        print(f"[DummyCamera] Initialized {len(configs)} virtual cameras")

    def capture(self) -> dict:
        result = {}
        for cfg in self.configs:
            w = cfg.get("width", 640)
            h = cfg.get("height", 480)

            # 随机 RGB 图（0~254）
            result[cfg["name"]] = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)

            # 若配置 rgbd，再生成随机深度（单位一般是毫米，示例 0~4999）
            if cfg.get("mode", "rgb") == "rgbd":
                result[f"{cfg['name']}_depth"] = np.random.randint(
                    0, 5000, (h, w), dtype=np.uint16
                )
        return result

    def stop(self):
        print("[DummyCamera] Stopped.")


# ============================================================================
# ROS robot interface（机器人状态与控制）
# ============================================================================
# 下面主题和消息类型默认对应 iiwa_stack + Robotiq 2F。
# 如果你的系统不同，需要改 topic 名和消息类型映射。


# iiwa topic / config —— 按你的系统改
JOINT_STATE_TOPIC = "/iiwa/state/JointPosition"      # iiwa_msgs/JointPosition
JOINT_CMD_TOPIC = "/iiwa/command/JointPosition"      # iiwa_msgs/JointPosition
HOME_POSITION = [0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, 0.0]  # 7 轴 home 位姿（弧度）

# Robotiq 2F 夹爪 topic
GRIPPER_STATE_TOPIC = "/Robotiq2FGripperRobotInput"    # Robotiq2FGripper_robot_input
GRIPPER_CMD_TOPIC = "/Robotiq2FGripperRobotOutput"     # Robotiq2FGripper_robot_output


class ROSRobot:
    """基于 ROS topic 的机器人接口：读状态 + 发命令。"""

    def __init__(self, home_position=None):
        import threading
        import rospy
        from iiwa_msgs.msg import JointPosition
        from robotiq_2f_gripper_control.msg import (
            Robotiq2FGripper_robot_input,
            Robotiq2FGripper_robot_output,
        )

        # 保存类引用，避免在其他方法里重复导入
        self._rospy = rospy
        self._JointPosition = JointPosition
        self._GripperOutput = Robotiq2FGripper_robot_output

        # 共享状态：由回调线程更新，主控制循环读取
        self._joint_positions = np.zeros(7, dtype=np.float64)
        self._gripper_state = -1.0  # 约定 -1=open, 1=closed
        self._lock = threading.Lock()

        self._home = np.array(home_position or HOME_POSITION, dtype=np.float64)

        # 订阅关节状态
        rospy.Subscriber(JOINT_STATE_TOPIC, JointPosition, self._joint_cb)
        # 发布关节命令
        self._cmd_pub = rospy.Publisher(JOINT_CMD_TOPIC, JointPosition, queue_size=1)

        # 夹爪反馈订阅 / 夹爪命令发布（可选）
        if GRIPPER_STATE_TOPIC:
            rospy.Subscriber(
                GRIPPER_STATE_TOPIC,
                Robotiq2FGripper_robot_input,
                self._gripper_cb,
            )
        if GRIPPER_CMD_TOPIC:
            self._gripper_pub = rospy.Publisher(
                GRIPPER_CMD_TOPIC,
                Robotiq2FGripper_robot_output,
                queue_size=1,
            )

        rospy.loginfo(f"[ROSRobot] Subscribed to {JOINT_STATE_TOPIC}")
        rospy.loginfo(f"[ROSRobot] Publishing to {JOINT_CMD_TOPIC}")

        # 记录 Robotiq 反馈状态字，便于激活流程诊断
        self._gripper_sta = 0  # gSTA：状态
        self._gripper_flt = 0  # gFLT：故障码

    def _joint_cb(self, msg):
        """关节状态回调：把消息映射为长度 7 的 numpy。"""
        with self._lock:
            pos = msg.position
            self._joint_positions = np.array(
                [pos.a1, pos.a2, pos.a3, pos.a4, pos.a5, pos.a6, pos.a7],
                dtype=np.float64,
            )

    def _gripper_cb(self, msg):
        """夹爪反馈回调。

        gPO 范围 0~255，将其线性映射到 -1~1：
        - 0   -> -1.0（开）
        - 255 ->  1.0（闭）
        """
        with self._lock:
            self._gripper_state = (msg.gPO / 255.0) * 2.0 - 1.0
            self._gripper_sta = msg.gSTA
            self._gripper_flt = msg.gFLT

    def get_joint_positions(self) -> np.ndarray:
        """线程安全地读取当前关节位置（float32 拷贝）。"""
        with self._lock:
            return self._joint_positions.astype(np.float32).copy()

    def get_gripper_state(self) -> float:
        """线程安全地读取当前夹爪状态（-1~1）。"""
        with self._lock:
            return self._gripper_state

    def execute_action(self, action: np.ndarray):
        """执行动作。

        约定：
        - action[:7] 是七轴关节目标
        - action[7]（可选）是夹爪目标，范围 -1~1
        """
        # 发布七轴关节命令
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

        # 若动作中含夹爪维度，则同步发布夹爪位置命令
        if len(action) > 7 and GRIPPER_CMD_TOPIC:
            grip_val = float(action[7])  # -1=open, 1=closed
            # 线性映射到 Robotiq rPR（0=open, 255=closed）
            rPR = int(np.clip((grip_val + 1.0) / 2.0 * 255.0, 0, 255))

            msg_g = self._GripperOutput()
            msg_g.rACT = 1
            msg_g.rGTO = 1
            msg_g.rSP = 255   # 速度
            msg_g.rFR = 150   # 力
            msg_g.rPR = rPR   # 目标位置
            self._gripper_pub.publish(msg_g)

    def activate_gripper(self):
        """执行 Robotiq 2F 激活序列。

        激活通常需要：
        1) reset
        2) activate
        3) 等待 gSTA == 3
        4) 发送 open
        """
        if not GRIPPER_CMD_TOPIC:
            return

        rospy = self._rospy

        rospy.loginfo("[ROSRobot] Gripper activation: reset...")
        # Step 1: reset（rACT=0）
        msg = self._GripperOutput()
        msg.rACT = 0
        for _ in range(5):
            self._gripper_pub.publish(msg)
            time.sleep(0.1)

        rospy.loginfo("[ROSRobot] Gripper activation: activating...")
        # Step 2: activate（rACT=1, rGTO=0）
        msg = self._GripperOutput()
        msg.rACT = 1
        msg.rGTO = 0
        msg.rSP = 255
        msg.rFR = 150
        for _ in range(5):
            self._gripper_pub.publish(msg)
            time.sleep(0.1)

        # Step 3: 等待激活完成，超时 10 秒
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
            rospy.logwarn(
                f"[ROSRobot] Gripper activation timeout! gSTA={sta} gFLT={flt}"
            )

        # Step 4: 激活后先打开夹爪
        rospy.loginfo("[ROSRobot] Gripper activation: opening...")
        msg = self._GripperOutput()
        msg.rACT = 1
        msg.rGTO = 1
        msg.rSP = 255
        msg.rFR = 150
        msg.rPR = 0
        for _ in range(5):
            self._gripper_pub.publish(msg)
            time.sleep(0.1)

        rospy.loginfo("[ROSRobot] Gripper activation complete")

    def go_home(self):
        """回 home，并打开夹爪。"""
        action = np.append(self._home, -1.0)
        self.execute_action(action)
        self._rospy.loginfo("[ROSRobot] Going home + opening gripper")

    def stop(self):
        """停止策略输出：通过重复当前关节保持姿态。"""
        state = self.get_joint_positions()
        self.execute_action(state)
        self._rospy.loginfo("[ROSRobot] Stop - holding current position")


class DummyRobot:
    """无 ROS / 无硬件时的机器人模拟器。"""

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
        # 直接“瞬时赋值”模拟控制结果，不做动力学/轨迹约束
        self._joints = action[:7].astype(np.float32)
        if len(action) > 7:
            self._gripper = float(action[7])

    def go_home(self):
        self._joints = self._home.copy()
        self._gripper = -1.0
        print("[DummyRobot] Moved to home position, gripper open")

    def stop(self):
        print("[DummyRobot] Stopped")


# ============================================================================
# 主循环（初始化 -> 多 episode 控制 -> 收尾）
# ============================================================================

DEFAULT_CAMERAS = [
    {"name": "base_camera",  "serial": "308222301757", "width": 640, "height": 480, "fps": 30, "mode": "rgb"},
    {"name": "wrist_camera", "serial": "306322300452", "width": 640, "height": 480, "fps": 30, "mode": "rgb"},
]


def main():
    # -------------------------------
    # 1) 解析命令行参数
    # -------------------------------
    parser = argparse.ArgumentParser(
        description="ROS Bridge: capture cameras + robot state, send to policy server"
    )
    parser.add_argument(
        "--server",
        type=str,
        default="tcp://localhost:5555",
        help="Policy server ZMQ address (e.g. tcp://192.168.1.50:5555)",
    )
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Use DummyRobot + DummyCamera (no hardware / no ROS)",
    )
    parser.add_argument(
        "--hz",
        type=int,
        default=15,
        help="Control loop frequency",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=300,
        help="Max steps per episode",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of episodes to run",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Optional YAML config (overrides CLI args for cameras etc.)",
    )
    args = parser.parse_args()

    # -------------------------------
    # 2) 读取可选 YAML 配置，覆盖 CLI 参数
    # -------------------------------
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

    # -------------------------------
    # 3) 初始化硬件或仿真对象
    # -------------------------------
    if args.dummy:
        cameras = DummyCamera(camera_configs)
        robot = DummyRobot(home_position)
    else:
        import rospy

        rospy.init_node("ros_bridge", anonymous=True)
        cameras = CameraManager(camera_configs)
        robot = ROSRobot(home_position)

        # 等待 ROS 回调填充初始状态，避免起步时读到全零
        print("[Bridge] Waiting for first joint state from ROS...")
        time.sleep(1.0)

        # 真实模式下先激活夹爪
        robot.activate_gripper()

    # -------------------------------
    # 4) ZMQ REQ 客户端连接策略服务器（服务端应为 REP）
    # -------------------------------
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(args.server)
    print(f"[Bridge] Connected to policy server: {args.server}")

    # -------------------------------
    # 5) Episode 主循环
    # -------------------------------
    try:
        for ep in range(args.num_episodes):
            input(
                f"\n[Bridge] Episode {ep + 1}/{args.num_episodes} "
                f"- Press Enter to start..."
            )

            # 每个 episode 开始先回 home
            robot.go_home()

            # 等待达到 home，误差阈值 0.05 rad
            home = np.array(HOME_POSITION, dtype=np.float32)
            print(f"[Bridge] Moving to home {home.tolist()} ...")
            while True:
                joints = robot.get_joint_positions()
                err = np.max(np.abs(joints - home))
                if err < 0.05:
                    print(f"[Bridge] Home reached (max_err={err:.4f} rad)")
                    break
                time.sleep(0.1)

            # 到位后重复发送开夹命令，提升链路不稳定时的可靠性
            open_action = np.append(home, -1.0)
            for _ in range(5):
                robot.execute_action(open_action)
                time.sleep(0.05)

            gripper_after = robot.get_gripper_state()
            print(
                "[Bridge] Gripper open cmd sent x5, "
                f"gripper_state={gripper_after:.3f} (expect ~-1.0)"
            )

            # reset 帧：通知服务器“新 episode 开始”
            images = cameras.capture()
            joints = robot.get_joint_positions()
            gripper = robot.get_gripper_state()
            sock.send(pack_observation(images, joints, gripper, reset=True))
            sock.recv()  # REQ/REP 必须一发一收，消费掉 reset ack

            print(
                f"[Bridge] Running episode {ep + 1} "
                f"(max {args.max_steps} steps, {args.hz} Hz)"
            )

            # 本地动作缓存：
            # - 缓存不空时，只在本地逐步执行
            # - 缓存空时，再向服务器请求下一 chunk
            action_buf = None
            buf_idx = 0
            done = False

            for step in range(args.max_steps):
                t0 = time.time()

                # 1) 每步都读取实时机器人状态
                joints = robot.get_joint_positions()
                gripper = robot.get_gripper_state()

                # 2) 仅在缓存耗尽时采图并请求服务器
                if action_buf is None or buf_idx >= len(action_buf):
                    images = cameras.capture()

                    t_net0 = time.time()
                    sock.send(
                        pack_observation(
                            images,
                            joints,
                            gripper,
                            jpeg_quality=jpeg_quality,
                        )
                    )
                    chunk_data = sock.recv()
                    t_net1 = time.time()

                    action_buf, done = unpack_chunk(chunk_data)
                    buf_idx = 0

                    print(
                        f"  [step {step:4d}] new chunk, "
                        f"net={1000 * (t_net1 - t_net0):.1f}ms"
                    )

                # 3) 从本地缓存取一条动作
                action = action_buf[buf_idx]
                buf_idx += 1

                # 4) 下发到机器人
                robot.execute_action(action)

                # 5) 频率控制：补足到目标 dt
                elapsed = time.time() - t0
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                # 实测本步频率
                actual_hz = 1.0 / max(time.time() - t0, 1e-6)

                # 降采样打印，避免日志过多影响实时性
                if step % 50 == 0:
                    joints_str = np.array2string(
                        joints,
                        precision=3,
                        suppress_small=True,
                    )
                    print(
                        f"  [step {step:4d}] "
                        f"joints={joints_str} gripper={gripper:.3f} "
                        f"hz={actual_hz:.1f}"
                    )

                # 服务器可通过 done 提前终止当前 episode
                if done:
                    print(f"  [step {step}] Policy signaled done")
                    break

            nsteps = step + 1 if args.max_steps > 0 else 0
            print(f"[Bridge] Episode {ep + 1} finished ({nsteps} steps)")

            # episode 结束信号：再次发 reset=True，触发服务器立即落盘
            images = cameras.capture()
            joints = robot.get_joint_positions()
            gripper = robot.get_gripper_state()
            sock.send(pack_observation(images, joints, gripper, reset=True))
            sock.recv()
            print(f"[Bridge] Episode {ep + 1} done signal sent - server saving data")

    except KeyboardInterrupt:
        print("\n[Bridge] Interrupted")

    finally:
        # 统一收尾，确保通信和硬件资源被释放
        robot.stop()
        cameras.stop()
        sock.close()
        ctx.term()
        print("[Bridge] Shutdown complete")


if __name__ == "__main__":
    main()