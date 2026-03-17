"""Policy inference server.

Receives observations via ZMQ, runs policy inference, returns actions.

Usage:
    python server.py --config config.yaml
    python server.py --config config.yaml --policy random   # test with random policy
"""

import argparse
import os
import time

import numpy as np
import yaml
import zmq

from msg_utils import pack_action, pack_chunk, unpack_observation


def save_episode(save_root: str, episode_count: int,
                 images_buf: dict, states_buf: list, actions_buf: list,
                 fps: int = 10, policy=None):
    """Save buffered episode data to disk at episode end."""
    if not states_buf:
        return
    ep_dir = os.path.join(save_root, f"ep{episode_count:03d}")
    os.makedirs(ep_dir, exist_ok=True)

    # Camera images → mp4 video via imageio (H264, widely compatible)
    import imageio
    for cam_name, frames in images_buf.items():
        video_path = os.path.join(ep_dir, f"{cam_name}.mp4")
        imageio.mimsave(video_path, frames, fps=fps)

    # States and actions → (T, 8) npy
    np.save(os.path.join(ep_dir, "states.npy"),  np.array(states_buf,  dtype=np.float32))
    np.save(os.path.join(ep_dir, "actions.npy"), np.array(actions_buf, dtype=np.float32))

    # Query frames: side-by-side video of frames actually fed into Pi0.5
    if policy is not None and hasattr(policy, "query_frames") and policy.query_frames:
        frames_side_by_side = [
            np.concatenate([base, wrist], axis=1)
            for base, wrist in policy.query_frames
        ]
        imageio.mimsave(os.path.join(ep_dir, "query_frames.mp4"), frames_side_by_side, fps=1)

    T = len(states_buf)
    Q = len(policy.query_frames) if (policy and hasattr(policy, "query_frames")) else 0
    print(f"  [Save] ep{episode_count:03d} → {ep_dir}  ({T} steps, "
          f"queries={Q}, cams={list(images_buf.keys())})")


def load_policy_from_config(config: dict):
    """Load policy based on config."""
    policy_name = config["server"]["policy"]
    checkpoint = config["server"].get("checkpoint", "")
    device = config["server"].get("device", "cuda")

    if policy_name == "random":
        from policies.base_policy import RandomPolicy
        policy = RandomPolicy()
        policy.load(checkpoint, device)
        print(f"[Server] Loaded RandomPolicy")
        return policy

    elif policy_name == "go_home":
        from policies.base_policy import GoHomePolicy
        home = config.get("robot", {}).get("home_position", None)
        policy = GoHomePolicy(home_position=home)
        policy.load(checkpoint, device)
        print(f"[Server] Loaded GoHomePolicy → {policy._home}")
        return policy

    elif policy_name == "act":
        from policies.act_policy import ACTPolicy
        camera_map = config["server"].get("camera_to_feature_map", None)
        policy = ACTPolicy(camera_to_feature_map=camera_map)
        policy.load(checkpoint, device)
        return policy

    elif policy_name == "pi05":
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

    elif policy_name == "pi05_h50":
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

    elif policy_name == "dp":
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

    else:
        raise ValueError(
            f"Unknown policy: '{policy_name}'. "
            f"Available: random, go_home, act, pi05, pi05_h50, dp"
        )


def main():
    parser = argparse.ArgumentParser(description="Policy inference server")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--policy", type=str, default=None,
                        help="Override policy name from config")
    parser.add_argument("--save", action="store_true",
                        help="Save per-episode images/states/actions to debug_output/")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.policy:
        config["server"]["policy"] = args.policy

    # Load policy
    policy = load_policy_from_config(config)

    # Setup ZMQ
    address = config["server"]["address"]
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REP)
    socket.bind(address)
    print(f"[Server] Listening on {address}")
    print(f"[Server] Ready. Waiting for client...")

    save_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug_output")
    step_count = 0
    episode_count = 0
    # Per-episode buffers (only used when --save is set)
    images_buf: dict[str, list] = {}
    states_buf: list = []
    actions_buf: list = []

    try:
        while True:
            # Receive observation (only sent by bridge when chunk buffer is empty)
            obs_data = socket.recv()
            obs = unpack_observation(obs_data)

            # Handle episode reset
            if obs.get("reset", False):
                if args.save and states_buf:
                    save_episode(save_root, episode_count, images_buf, states_buf, actions_buf, policy=policy)
                policy.reset()
                episode_count += 1
                step_count = 0
                images_buf = {}
                states_buf = []
                actions_buf = []
                print(f"\n[Server] Episode {episode_count} reset")
                socket.send(pack_chunk(np.zeros((1, 8), dtype=np.float32), done=False))
                continue

            # Run inference — always a real inference call (buffer managed by bridge)
            t0 = time.time()
            if hasattr(policy, "predict_chunk"):
                chunk = policy.predict_chunk(obs)   # (n_action_steps, 8)
            else:
                # Backward-compat: policies that only implement predict()
                chunk = np.asarray(policy.predict(obs), dtype=np.float32).reshape(1, -1)
            chunk = np.asarray(chunk, dtype=np.float32)
            if chunk.ndim == 1:
                chunk = chunk.reshape(1, -1)
            dt = time.time() - t0

            step_count += 1
            joints = obs["joint_positions"]
            grip = obs["gripper_state"]
            print(f"  [query {step_count:4d}] inference: {dt*1000:.1f}ms  "
                  f"joints[:3]={joints[:3]}  gripper={grip:.3f}")

            # Buffer data for end-of-episode save (one entry per chunk query)
            if args.save:
                for cam_name, img in obs["images"].items():
                    images_buf.setdefault(cam_name, []).append(img.copy())
                state = np.zeros(8, dtype=np.float32)
                state[:7] = obs["joint_positions"][:7]
                state[7]  = float(obs["gripper_state"])
                states_buf.append(state)
                actions_buf.append(chunk[0].copy())

            # Send full chunk back to bridge
            socket.send(pack_chunk(chunk, done=False))

    except KeyboardInterrupt:
        print("\n[Server] Shutting down...")
        if args.save and states_buf:
            save_episode(save_root, episode_count, images_buf, states_buf, actions_buf, policy=policy)
    finally:
        socket.close()
        ctx.term()


if __name__ == "__main__":
    main()
