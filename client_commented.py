"""
机器人控制客户端（带详细注释版）。

这个文件的核心作用是：
1. 从配置文件中读取相机、机器人、控制频率、服务端地址等参数。
2. 初始化相机和机器人接口。
3. 在每个控制周期里采集观测：
   - 相机图像
   - 机器人关节角
   - 夹爪状态
4. 把观测数据通过 ZMQ 发给策略服务端（server.py）。
5. 接收服务端返回的动作，并立刻在机器人上执行。
6. 按设定频率循环执行，直到 episode 结束。

它适合两种场景：
- 真机/真实相机联调：用于主机本地直接采集并控制。
- dummy 测试：不连真实硬件，用 DummyRobot + DummyCamera 验证通信和流程。

使用方式：
    python client.py --config config.yaml
    python client.py --config config.yaml --dummy
"""

import argparse
import time

import numpy as np
import yaml
import zmq

# 相机相关抽象：
# - CameraConfig: 单个相机的配置结构
# - CameraManager: 管理真实相机
# - DummyCamera: 用于无硬件测试时的假相机
from camera import CameraConfig, CameraManager, DummyCamera

# 消息编解码工具：
# - pack_observation: 把图像、关节、夹爪等观测打包成可通过网络发送的字节流
# - unpack_action: 把服务端返回的动作字节流解码成 action 和 done 标志
from msg_utils import pack_observation, unpack_action

# 机器人接口抽象：
# - DummyRobot: 假机器人，用于本地流程测试
# - KukaIIWAInterface: 真实 KUKA iiwa 机械臂接口
from robot_interface import DummyRobot, KukaIIWAInterface


def build_cameras(config: dict, dummy: bool = False):
    """
    根据配置文件创建相机对象。

    参数：
        config:
            从 yaml 读出来的完整配置字典。
        dummy:
            为 True 时，不连接真实相机，而是返回 DummyCamera。

    返回：
        CameraManager 或 DummyCamera 实例。

    说明：
    配置文件中的 cameras 段通常长这样：

        cameras:
          - name: base_camera
            serial: "123456"
            width: 640
            height: 480
            fps: 30
            mode: rgb

    这里会把每个相机配置项转成 CameraConfig，再交给相机管理器统一初始化。
    """
    cam_configs = [
        CameraConfig(
            # 相机逻辑名称，例如 base_camera / wrist_camera
            name=c["name"],
            # 设备序列号，真实相机时通常要靠它绑定具体设备
            serial=c.get("serial", ""),
            # 分辨率和帧率采用配置中的值；若缺失则给默认值
            width=c.get("width", 640),
            height=c.get("height", 480),
            fps=c.get("fps", 30),
            # mode 可能是 rgb / depth 等，默认 rgb
            mode=c.get("mode", "rgb"),
        )
        for c in config["cameras"]
    ]

    # dummy 模式下返回假的相机对象，便于在没有硬件时测试整条链路
    if dummy:
        return DummyCamera(cam_configs)

    # 非 dummy 模式则连接真实相机
    return CameraManager(cam_configs)


def build_robot(config: dict, dummy: bool = False):
    """
    根据配置文件创建机器人接口对象。

    参数：
        config:
            完整配置字典。
        dummy:
            为 True 时返回 DummyRobot；否则返回真实 KUKA 接口。

    返回：
        DummyRobot 或 KukaIIWAInterface 实例。

    说明：
    robot 配置通常包含 home_position、网络地址等信息。
    """
    robot_cfg = config["robot"]

    # dummy 模式下只需要 home_position 即可模拟基本行为
    if dummy:
        return DummyRobot(home_position=robot_cfg.get("home_position"))

    # 真实模式下把整段 robot 配置传给底层接口类
    return KukaIIWAInterface(robot_cfg)


def main():
    """
    主控制流程。

    整体执行顺序如下：
    1. 解析命令行参数
    2. 读取配置文件
    3. 初始化相机和机器人
    4. 连接策略服务端（ZMQ REQ）
    5. 按 episode 循环执行：
       - 回 home
       - 发送 reset 观测
       - 进入 step 循环：采集观测 -> 发给服务端 -> 收动作 -> 执行动作
    6. 程序退出时释放资源
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="Robot control client")

    # 指定配置文件路径，默认使用当前目录下的 config.yaml
    parser.add_argument("--config", type=str, default="config.yaml")

    # 开启后不连接真实机器人和真实相机，而是使用假设备
    parser.add_argument(
        "--dummy",
        action="store_true",
        help="Use DummyRobot + DummyCamera for testing",
    )

    args = parser.parse_args()

    # 读取 YAML 配置
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # 提取控制相关配置；如果 control 段不存在，就退回到空字典
    control_cfg = config.get("control", {})

    # 控制频率（Hz）：例如 10 表示每秒执行 10 次控制循环
    hz = control_cfg.get("hz", 10)

    # 每个 episode 最多执行多少步
    max_steps = control_cfg.get("max_steps", 300)

    # 总共执行多少个 episode
    num_episodes = control_cfg.get("num_episodes", 10)

    # 每一步理论上应该持续的时间长度（秒）
    dt = 1.0 / hz

    # 初始化硬件接口：
    # cameras 负责采图，robot 负责读状态和执行动作
    cameras = build_cameras(config, dummy=args.dummy)
    robot = build_robot(config, dummy=args.dummy)

    # 初始化 ZMQ 通信
    #
    # 这里客户端使用 REQ（request）模式，服务端通常对应 REP（reply）模式。
    # 通信节奏必须严格遵守“一发一收”的顺序：
    #   send -> recv -> send -> recv ...
    #
    # 如果配置里有 connect_address，就优先用它；
    # 否则退回到 server.address。
    address = config["server"].get("connect_address", config["server"]["address"])
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REQ)
    socket.connect(address)
    print(f"[Client] Connected to {address}")

    try:
        # 外层循环：逐个执行多个 episode
        for ep in range(num_episodes):
            # 手动等待用户确认后再开始，避免机器人立刻动作
            input(f"\n[Client] Episode {ep+1}/{num_episodes} — Press Enter to start...")

            # 每个 episode 开始前先让机器人回到 home 位
            robot.go_home()

            # 发送 reset 信号，通知服务端：
            # “一个新 episode 要开始了，请清空内部状态/动作缓存/时序上下文”
            #
            # 即使是 reset 包，仍然会附带一帧当前观测，方便服务端初始化。
            images = cameras.capture()
            joints = robot.get_joint_positions()
            gripper = robot.get_gripper_state()
            socket.send(pack_observation(images, joints, gripper, reset=True))

            # REQ/REP 模式下，发送 reset 后也必须接收一次回复。
            # 这里不关心回复内容，只是把 ack 收掉，确保通信状态机正确推进。
            socket.recv()

            print(f"[Client] Running episode {ep+1} (max {max_steps} steps, {hz} Hz)")

            # 这个变量在当前版本里没有继续使用，
            # 可能是为后续统计成功步数或有效步数预留的。
            success_steps = 0

            # 内层循环：episode 中的每一个控制 step
            for step in range(max_steps):
                # 记录本 step 的起始时间，用于控制频率
                t0 = time.time()

                # 1. 采集观测
                #
                # images: 多路相机图像，通常是一个 dict
                # joints: 机器人当前关节角
                # gripper: 当前夹爪开合状态
                images = cameras.capture()
                joints = robot.get_joint_positions()
                gripper = robot.get_gripper_state()

                # 2. 把观测发给策略服务端
                #
                # 服务端会根据这些观测执行一次 policy inference，
                # 然后返回下一步要执行的动作。
                socket.send(pack_observation(images, joints, gripper))

                # 3. 接收服务端返回的动作
                action_data = socket.recv()

                # unpack_action 会把网络字节流解析成：
                # - action: 要发给机器人的动作向量
                # - done: 当前 episode 是否应当结束
                action, done = unpack_action(action_data)

                # 4. 在机器人上执行动作
                #
                # 对真实机器人来说，这通常意味着发送关节目标或末端执行器命令；
                # 对 DummyRobot 来说，则是内部状态更新。
                robot.execute_action(action)

                # 5. 控频
                #
                # 目标是让整个循环尽量接近 hz 指定的频率。
                # 如果采集、通信、推理、执行太快，就 sleep 一小段时间补齐周期。
                elapsed = time.time() - t0
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                # 用“这一整步实际耗时”反推出真实运行频率
                actual_hz = 1.0 / (time.time() - t0)

                # 每 50 步打印一次日志，便于观察运行状态
                if step % 50 == 0:
                    print(
                        f"  [step {step:4d}] joints[:3]={joints[:3]}, "
                        f"hz={actual_hz:.1f}"
                    )

                # 如果策略返回 done=True，提前结束当前 episode
                if done:
                    print(f"  [step {step}] Policy signaled done")
                    break

            # 当前 episode 结束后打印统计信息
            print(f"[Client] Episode {ep+1} finished ({step+1} steps)")

    except KeyboardInterrupt:
        # 用户按 Ctrl+C 时会进入这里，避免直接抛异常退出
        print("\n[Client] Interrupted")
    finally:
        # 无论是正常结束还是异常中断，都要做资源清理
        #
        # 1. 停止机器人接口
        # 2. 停止相机采集
        # 3. 关闭 ZMQ socket
        # 4. 终止 ZMQ context
        robot.stop()
        cameras.stop()
        socket.close()
        ctx.term()
        print("[Client] Shutdown complete")


# 只有当这个文件被“直接运行”时才进入 main()。
# 如果它被别的文件 import，则不会自动启动控制循环。
if __name__ == "__main__":
    main()
