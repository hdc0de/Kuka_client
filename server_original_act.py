"""Policy inference server with support for the original ACT repository."""

import argparse
import os
import time

import numpy as np
import yaml
import zmq

from msg_utils import pack_chunk, unpack_observation


def save_episode(save_root: str, episode_count: int,
                 images_buf: dict, states_buf: list, actions_buf: list,
                 fps: int = 10, policy=None):
    """Save buffered episode data to disk at episode end."""
    if not states_buf:
        return
    ep_dir = os.path.join(save_root, f"ep{episode_count:03d}")
    os.makedirs(ep_dir, exist_ok=True)

    import imageio

    for cam_name, frames in images_buf.items():
        video_path = os.path.join(ep_dir, f"{cam_name}.mp4")
        imageio.mimsave(video_path, frames, fps=fps)

    np.save(os.path.join(ep_dir, "states.npy"), np.array(states_buf, dtype=np.float32))
    np.save(os.path.join(ep_dir, "actions.npy"), np.array(actions_buf, dtype=np.float32))

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

    t_steps = len(states_buf)
    queries = len(policy.query_frames) if (policy and hasattr(policy, "query_frames")) else 0
    print(
        f"  [Save] ep{episode_count:03d} -> {ep_dir} "
        f"({t_steps} steps, queries={queries}, cams={list(images_buf.keys())})"
    )


def load_policy_from_config(config: dict):
    """Load policy based on config."""
    policy_name = config["server"]["policy"]
    checkpoint = config["server"].get("checkpoint", "")
    device = config["server"].get("device", "cuda")

    if policy_name == "random":
        from policies.base_policy import RandomPolicy

        policy = RandomPolicy()
        policy.load(checkpoint, device)
        print("[Server] Loaded RandomPolicy")
        return policy

    if policy_name == "go_home":
        from policies.base_policy import GoHomePolicy

        home = config.get("robot", {}).get("home_position", None)
        policy = GoHomePolicy(home_position=home)
        policy.load(checkpoint, device)
        print(f"[Server] Loaded GoHomePolicy -> {policy._home}")
        return policy

    if policy_name == "act":
        from policies.act_policy import ACTPolicy

        camera_map = config["server"].get("camera_to_feature_map", None)
        policy = ACTPolicy(camera_to_feature_map=camera_map)
        policy.load(checkpoint, device)
        return policy

    if policy_name == "act_original":
        from policies.original_act_policy import OriginalACTPolicy

        server_cfg = config["server"]
        policy = OriginalACTPolicy(
            act_root=server_cfg.get("act_root", "/home/user/catkin_ws/src/iiwa_python/src/my_ACT"),
            chunk_size=server_cfg.get("act_chunk_size", 100),
            hidden_dim=server_cfg.get("act_hidden_dim", 512),
            dim_feedforward=server_cfg.get("act_dim_feedforward", 3200),
            kl_weight=server_cfg.get("act_kl_weight", 10),
            n_action_steps=server_cfg.get("n_action_steps", 1),
            temporal_agg=server_cfg.get("act_temporal_agg", False),
            ensemble_decay=server_cfg.get("act_ensemble_decay", 0.01),
            camera_names=server_cfg.get("act_camera_names", ["front", "side"]),
            camera_sources=server_cfg.get("act_camera_sources", ["base_camera", "wrist_camera"]),
            image_height=server_cfg.get("act_image_height", 480),
            image_width=server_cfg.get("act_image_width", 640),
            state_key=server_cfg.get("act_state_key", "joint_positions"),
            action_pose_mode=server_cfg.get("act_action_pose_mode", "absolute"),
            input_gripper_mode=server_cfg.get("act_input_gripper_mode", "normalized"),
            output_gripper_mode=server_cfg.get("act_output_gripper_mode", "normalized"),
            gripper_width_min=server_cfg.get("act_gripper_width_min", 0.0),
            gripper_width_max=server_cfg.get("act_gripper_width_max", 0.085),
            debug_actions=server_cfg.get("act_debug_actions", False),
            debug_interval=server_cfg.get("act_debug_interval", 50),
        )
        policy.load(checkpoint, device)
        return policy

    if policy_name == "pi05":
        from policies.pi05_policy import Pi05Policy

        config_name = config["server"].get("config_name", "pi05_greycup_lite")
        n_action_steps = config["server"].get("n_action_steps", 25)
        task_description = config["server"].get(
            "task_description", "Pick up grey_cup and place on blue_cup"
        )
        policy = Pi05Policy(
            config_name=config_name,
            n_action_steps=n_action_steps,
            task_description=task_description,
        )
        policy.load(checkpoint, device)
        return policy

    if policy_name == "pi05_h50":
        from policies.pi05_policy import Pi05Policy

        n_action_steps = config["server"].get("n_action_steps", 25)
        task_description = config["server"].get(
            "task_description", "Pick up grey_cup and place on blue_cup"
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

    if policy_name == "dp":
        from policies.dp_policy import DPPolicy

        n_action_steps = config["server"].get("n_action_steps", 1)
        camera_map = config["server"].get("camera_to_feature_map", None)
        gripper_binarize_threshold = config["server"].get(
            "gripper_binarize_threshold", None
        )
        max_joint_delta = config["server"].get("max_joint_delta", None)
        joint_limits = config["server"].get("joint_limits", None)
        xsim_root = config["server"].get(
            "xsim_root", "/home/rl/projects/X-Sim/diffusion_policy_xsim"
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

    raise ValueError(
        f"Unknown policy: '{policy_name}'. "
        f"Available: random, go_home, act, act_original, pi05, pi05_h50, dp"
    )


def main():
    parser = argparse.ArgumentParser(description="Policy inference server")
    parser.add_argument("--config", type=str, default="config_original_act.yaml")
    parser.add_argument("--policy", type=str, default=None,
                        help="Override policy name from config")
    parser.add_argument("--save", action="store_true",
                        help="Save per-episode images/states/actions to debug_output/")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.policy:
        config["server"]["policy"] = args.policy

    policy = load_policy_from_config(config)

    address = config["server"]["address"]
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REP)
    socket.bind(address)
    print(f"[Server] Listening on {address}")
    print("[Server] Ready. Waiting for client...")

    save_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_output")
    step_count = 0
    episode_count = 0
    images_buf: dict[str, list] = {}
    states_buf: list = []
    actions_buf: list = []

    try:
        while True:
            obs_data = socket.recv()
            obs = unpack_observation(obs_data)

            if obs.get("reset", False):
                if args.save and states_buf:
                    save_episode(
                        save_root,
                        episode_count,
                        images_buf,
                        states_buf,
                        actions_buf,
                        policy=policy,
                    )
                policy.reset()
                episode_count += 1
                step_count = 0
                images_buf = {}
                states_buf = []
                actions_buf = []
                print(f"\n[Server] Episode {episode_count} reset")
                socket.send(pack_chunk(np.zeros((1, 8), dtype=np.float32), done=False))
                continue

            t0 = time.time()
            if hasattr(policy, "predict_chunk"):
                chunk = policy.predict_chunk(obs)
            else:
                chunk = np.asarray(policy.predict(obs), dtype=np.float32).reshape(1, -1)
            chunk = np.asarray(chunk, dtype=np.float32)
            if chunk.ndim == 1:
                chunk = chunk.reshape(1, -1)
            dt = time.time() - t0

            step_count += 1
            joints = obs["joint_positions"]
            grip = obs["gripper_state"]
            print(
                f"  [query {step_count:4d}] inference: {dt * 1000:.1f}ms  "
                f"state[:3]={joints[:3]}  gripper={grip:.3f}"
            )

            if args.save:
                for cam_name, img in obs["images"].items():
                    images_buf.setdefault(cam_name, []).append(img.copy())
                state = np.zeros(8, dtype=np.float32)
                state[:7] = obs["joint_positions"][:7]
                state[7] = float(obs["gripper_state"])
                states_buf.append(state)
                actions_buf.append(chunk[0].copy())

            socket.send(pack_chunk(chunk, done=False))

    except KeyboardInterrupt:
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
        socket.close()
        ctx.term()


if __name__ == "__main__":
    main()
