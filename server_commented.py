"""策略推理服务端。

这个脚本的职责是：
1. 通过 ZMQ 接收客户端/桥接层发送来的观测数据。
2. 根据配置加载对应的策略模型。
3. 调用策略做推理，得到动作序列（chunk）。
4. 把动作结果再通过 ZMQ 发回客户端。
5. 在开启 ``--save`` 时，把每个 episode 的图像、状态、动作保存到磁盘，便于调试。

常见启动方式：
    python server_commented.py --config config.yaml
    python server_commented.py --config config.yaml --policy random
"""

import argparse
import os
import time

import numpy as np
import yaml
import zmq

from msg_utils import pack_chunk, unpack_observation


def save_episode(
    save_root: str,
    episode_count: int,
    images_buf: dict,
    states_buf: list,
    actions_buf: list,
    fps: int = 10,
    policy=None,
):
    """在一个 episode 结束时，把缓存的数据落盘。

    参数说明：
    - save_root: 所有调试输出的根目录。
    - episode_count: 当前 episode 编号，会用于生成类似 ``ep001`` 的目录名。
    - images_buf: 每个相机对应的图像帧缓存，结构通常是 ``{camera_name: [frame1, frame2, ...]}``。
    - states_buf: 每一步缓存的机器人状态，通常是长度为 8 的向量。
    - actions_buf: 每一步对应保存的动作，当前实现保存的是每次 chunk 的第一个动作。
    - fps: 导出视频时使用的帧率。
    - policy: 可选。某些策略（例如 Pi0.5）会额外缓存查询帧，这里一并保存。

    注意：
    - 如果 ``states_buf`` 为空，说明这一局没有有效数据，直接返回。
    - 该函数不会清理缓存，只负责保存。
    """
    if not states_buf:
        return

    # 为当前 episode 创建单独目录，便于后续按回合检查数据。
    ep_dir = os.path.join(save_root, f"ep{episode_count:03d}")
    os.makedirs(ep_dir, exist_ok=True)

    # 延迟导入 imageio，避免在不需要保存视频时增加启动成本。
    import imageio

    # 把每个相机的帧序列保存成 mp4 视频。
    # 这样比单独存很多张图片更容易回放和检查。
    for cam_name, frames in images_buf.items():
        video_path = os.path.join(ep_dir, f"{cam_name}.mp4")
        imageio.mimsave(video_path, frames, fps=fps)

    # 状态和动作保存为 numpy 数组。
    # 这里约定状态和动作都是 float32，便于后续训练/分析时直接加载。
    np.save(
        os.path.join(ep_dir, "states.npy"),
        np.array(states_buf, dtype=np.float32),
    )
    np.save(
        os.path.join(ep_dir, "actions.npy"),
        np.array(actions_buf, dtype=np.float32),
    )

    # 某些策略会记录“真正送入模型推理的图像帧”。
    # 如果存在这些 query_frames，就把 base 和 wrist 视角横向拼接成一个视频保存下来，
    # 这样排查模型输入是否正确会更直观。
    if policy is not None and hasattr(policy, "query_frames") and policy.query_frames:
        frames_side_by_side = [
            np.concatenate([base, wrist], axis=1)
            for base, wrist in policy.query_frames
        ]
        imageio.mimsave(
            os.path.join(ep_dir, "query_frames.mp4"),
            frames_side_by_side,
            fps=1,
        )

    # T 表示本回合缓存了多少条状态记录。
    # Q 表示额外保存了多少条策略查询帧。
    T = len(states_buf)
    Q = len(policy.query_frames) if (policy and hasattr(policy, "query_frames")) else 0
    print(
        f"  [Save] ep{episode_count:03d} -> {ep_dir}  "
        f"({T} steps, queries={Q}, cams={list(images_buf.keys())})"
    )


def load_policy_from_config(config: dict):
    """根据配置文件加载策略对象。

    这里把不同策略的初始化逻辑集中在一个地方，方便统一维护。
    配置主要读取 ``config['server']`` 下的字段，例如：
    - policy: 策略名称
    - checkpoint: 模型权重路径
    - device: 推理设备，例如 cuda / cpu

    返回值：
    - 一个已经完成 ``load(...)`` 的策略实例，可直接用于后续推理。
    """
    policy_name = config["server"]["policy"]
    checkpoint = config["server"].get("checkpoint", "")
    device = config["server"].get("device", "cuda")

    if policy_name == "random":
        # 随机策略，通常用于联调消息链路是否正常。
        from policies.base_policy import RandomPolicy

        policy = RandomPolicy()
        policy.load(checkpoint, device)
        print("[Server] Loaded RandomPolicy")
        return policy

    elif policy_name == "go_home":
        # 回零/回家策略，用于让机械臂回到预设位置。
        from policies.base_policy import GoHomePolicy

        home = config.get("robot", {}).get("home_position", None)
        policy = GoHomePolicy(home_position=home)
        policy.load(checkpoint, device)
        print(f"[Server] Loaded GoHomePolicy -> {policy._home}")
        return policy

    elif policy_name == "act":
        # ACT 策略，会读取相机到特征的映射关系。
        from policies.act_policy import ACTPolicy

        camera_map = config["server"].get("camera_to_feature_map", None)
        policy = ACTPolicy(camera_to_feature_map=camera_map)
        policy.load(checkpoint, device)
        return policy

    elif policy_name == "pi05":
        # Pi0.5 策略，可通过配置指定模型名、动作步数、任务描述。
        from policies.pi05_policy import Pi05Policy

        config_name = config["server"].get("config_name", "pi05_greycup_lite")
        n_action_steps = config["server"].get("n_action_steps", 25)
        task_description = config["server"].get(
            "task_description",
            "Pick up grey_cup and place on blue_cup",
        )
        policy = Pi05Policy(
            config_name=config_name,
            n_action_steps=n_action_steps,
            task_description=task_description,
        )
        policy.load(checkpoint, device)
        return policy

    elif policy_name == "pi05_h50":
        # 这是一个写死了特定 LoRA checkpoint 的 Pi0.5 变体。
        # 从实现上看，它不会使用 config 里的 checkpoint，而是固定加载下面这条路径。
        from policies.pi05_policy import Pi05Policy

        n_action_steps = config["server"].get("n_action_steps", 25)
        task_description = config["server"].get(
            "task_description",
            "Pick up grey_cup and place on blue_cup",
        )
        policy = Pi05Policy(
            config_name="pi05_greycup_h50_lora",
            n_action_steps=n_action_steps,
            task_description=task_description,
        )
        policy.load(
            "/home/rl/projects/X-Sim/openpi/checkpoints/pi05_greycup_h50_lora/greycup_lora/20000",
            device,
        )
        return policy

    elif policy_name == "dp":
        # Diffusion Policy，相关参数较多，因此从配置里读取了更多约束项。
        from policies.dp_policy import DPPolicy

        n_action_steps = config["server"].get("n_action_steps", 1)
        camera_map = config["server"].get("camera_to_feature_map", None)
        gripper_binarize_threshold = config["server"].get(
            "gripper_binarize_threshold",
            None,
        )
        max_joint_delta = config["server"].get("max_joint_delta", None)
        joint_limits = config["server"].get("joint_limits", None)
        xsim_root = config["server"].get(
            "xsim_root",
            "/home/rl/projects/X-Sim/diffusion_policy_xsim",
        )

        policy = DPPolicy(
            n_action_steps=n_action_steps,
            camera_to_feature_map=camera_map,
            gripper_binarize_threshold=gripper_binarize_threshold,
            max_joint_delta=max_joint_delta,
            joint_limits=joint_limits,
            xsim_root=xsim_root,
        )
        policy.load(checkpoint, device)
        return policy

    else:
        # 如果配置里的策略名不在支持列表中，直接报错，避免服务启动后行为不明确。
        raise ValueError(
            f"Unknown policy: '{policy_name}'. "
            f"Available: random, go_home, act, pi05, pi05_h50, dp"
        )


def main():
    """程序入口。

    整体流程如下：
    1. 解析命令行参数。
    2. 读取 YAML 配置。
    3. 加载策略对象。
    4. 初始化 ZMQ REP socket，等待客户端请求。
    5. 在循环中接收观测、执行推理、返回动作 chunk。
    6. 在 episode reset 或程序退出时，按需保存调试数据。
    """
    parser = argparse.ArgumentParser(description="Policy inference server")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Override policy name from config",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save per-episode images/states/actions to debug_output/",
    )
    args = parser.parse_args()

    # 读取配置文件。这里默认假设配置文件存在且 YAML 格式合法。
    with open(args.config) as f:
        config = yaml.safe_load(f)

    # 如果命令行显式传了 --policy，则以命令行为准，覆盖配置文件中的设置。
    if args.policy:
        config["server"]["policy"] = args.policy

    # 加载策略模型/对象。
    policy = load_policy_from_config(config)

    # 初始化 ZMQ 的 REP（reply）端。
    # REP 模式要求服务端严格遵循“收一条 -> 回一条”的节奏。
    address = config["server"]["address"]
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REP)
    socket.bind(address)
    print(f"[Server] Listening on {address}")
    print("[Server] Ready. Waiting for client...")

    # 调试输出目录，默认放在当前脚本同级目录下的 debug_output/ 中。
    save_root = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "debug_output",
    )

    # step_count: 当前 episode 内已经处理了多少次查询。
    # episode_count: 当前是第几个 episode。
    step_count = 0
    episode_count = 0

    # 以下三个缓存仅在 --save 开启时真正有意义：
    # - images_buf: 每个相机的图像序列
    # - states_buf: 每一步的状态向量
    # - actions_buf: 每一步保存的动作
    images_buf: dict[str, list] = {}
    states_buf: list = []
    actions_buf: list = []

    try:
        while True:
            # 只有当桥接层那边的 chunk buffer 为空时，才会再次向这里发送新观测。
            # 因此这里一次 recv 通常对应一次“新的推理请求”。
            obs_data = socket.recv()

            # 把二进制消息还原为 Python 字典形式的观测数据。
            obs = unpack_observation(obs_data)

            # 如果收到 reset 标记，说明上一个 episode 已结束，或者客户端要求重新开始。
            if obs.get("reset", False):
                # 如果用户要求保存，而且当前缓存里确实有数据，就先把上一局落盘。
                if args.save and states_buf:
                    save_episode(
                        save_root,
                        episode_count,
                        images_buf,
                        states_buf,
                        actions_buf,
                        policy=policy,
                    )

                # 通知策略清空内部状态，避免跨 episode 污染。
                policy.reset()

                # 开启新 episode，并清空计数器和缓存。
                episode_count += 1
                step_count = 0
                images_buf = {}
                states_buf = []
                actions_buf = []

                print(f"\n[Server] Episode {episode_count} reset")

                # reset 时仍然回复一个合法的空动作 chunk，保持通信协议完整。
                socket.send(
                    pack_chunk(np.zeros((1, 8), dtype=np.float32), done=False)
                )
                continue

            # 正常推理路径。
            # 这里计时主要是为了观察模型推理耗时。
            t0 = time.time()

            if hasattr(policy, "predict_chunk"):
                # 新版策略接口：一次返回多个未来动作，形状通常为 (n_action_steps, 8)。
                chunk = policy.predict_chunk(obs)
            else:
                # 兼容旧版接口：如果策略只实现了 predict()，则把单步动作包装成 (1, action_dim)。
                chunk = np.asarray(policy.predict(obs), dtype=np.float32).reshape(1, -1)

            # 确保最终拿到的是 float32 的 numpy 数组。
            chunk = np.asarray(chunk, dtype=np.float32)

            # 如果策略意外返回了一维向量，这里统一转成二维，保持下游接口稳定。
            if chunk.ndim == 1:
                chunk = chunk.reshape(1, -1)

            dt = time.time() - t0

            step_count += 1
            joints = obs["joint_positions"]
            grip = obs["gripper_state"]
            print(
                f"  [query {step_count:4d}] inference: {dt * 1000:.1f}ms  "
                f"joints[:3]={joints[:3]}  gripper={grip:.3f}"
            )

            # 如果开启了保存功能，就把当前观测和动作缓存起来。
            # 注意这里是一“次查询”存一条，而不是把整个 chunk 的所有动作都展开保存。
            if args.save:
                for cam_name, img in obs["images"].items():
                    images_buf.setdefault(cam_name, []).append(img.copy())

                # 约定状态维度为 8：
                # 前 7 维是关节位置，第 8 维是 gripper 状态。
                state = np.zeros(8, dtype=np.float32)
                state[:7] = obs["joint_positions"][:7]
                state[7] = float(obs["gripper_state"])
                states_buf.append(state)

                # 当前实现只保存 chunk 的第一个动作。
                # 这通常代表“本次查询立刻会执行的动作”。
                actions_buf.append(chunk[0].copy())

            # 把完整 chunk 回传给桥接层，由桥接层负责后续缓冲和逐步执行。
            socket.send(pack_chunk(chunk, done=False))

    except KeyboardInterrupt:
        # 用户按 Ctrl+C 时，优雅退出，并尽量保存未落盘的数据。
        print("\n[Server] Shutting down...")
        if args.save and states_buf:
            save_episode(
                save_root,
                episode_count,
                images_buf,
                states_buf,
                actions_buf,
                policy=policy,
            )
    finally:
        # 无论是正常退出还是异常退出，都把 socket 和 context 关闭干净。
        socket.close()
        ctx.term()


if __name__ == "__main__":
    main()
