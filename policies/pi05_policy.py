"""Pi0.5 (openpi/JAX) policy wrapper for real robot deployment.

Loads an openpi checkpoint and provides predict() for the real robot pipeline.
Handles action chunking internally: each inference call returns `action_horizon`
actions; the policy executes `n_action_steps` from the buffer before re-querying.

Camera mapping (config camera name → openpi key):
    base_camera  → observation/image
    wrist_camera → observation/wrist_image
"""

import sys
sys.path.insert(0, "/home/rl/projects/X-Sim/openpi/src")
sys.path.insert(0, "/home/rl/projects/X-Sim/openpi/packages/openpi-client/src")

import numpy as np
from pathlib import Path

from .base_policy import BasePolicy

H, W = 480, 640


class Pi05Policy(BasePolicy):
    """Wraps openpi Pi0.5 policy for real robot deployment."""

    def __init__(
        self,
        config_name: str,
        n_action_steps: int = 50,
        task_description: str = "Pick up grey_cup and place on blue_cup",
    ):
        """
        Args:
            config_name: openpi training config name (e.g. "pi05_greycup_lite").
            n_action_steps: Steps to execute from each predicted chunk before
                re-querying the model.
            task_description: Language prompt sent to the VLM head.
        """
        self.config_name = config_name
        self.n_action_steps = n_action_steps
        self.task_description = task_description

        self._policy = None
        self._action_horizon = None
        self._chunk_buffer = None   # np.ndarray (action_horizon, 8)
        self._chunk_idx = 0
        # Frames actually fed into Pi0.5 at each inference call
        self.query_frames: list[tuple[np.ndarray, np.ndarray]] = []  # [(base, wrist), ...]

    def load(self, checkpoint_path: str, device: str = "cuda"):
        import openpi.training.config as _config
        import openpi.policies.policy_config as _policy_config

        ckpt = checkpoint_path
        if not Path(ckpt).is_absolute():
            ckpt = str(Path("/home/rl/projects/X-Sim/openpi") / ckpt)

        train_config = _config.get_config(self.config_name)
        self._action_horizon = train_config.model.action_horizon

        self._policy = _policy_config.create_trained_policy(
            train_config, ckpt, pytorch_device=device,
        )
        print(f"[Pi05Policy] Loaded from {ckpt}")
        print(f"[Pi05Policy] config={self.config_name}  "
              f"action_horizon={self._action_horizon}  "
              f"n_action_steps={self.n_action_steps}")

    def predict(self, obs: dict) -> np.ndarray:
        """Predict action from observation with action chunking.

        Args:
            obs: {
                "images": {"base_camera": (H,W,3) uint8, "wrist_camera": (H,W,3) uint8},
                "joint_positions": (7,) float32,
                "gripper_state": float,   # -1 (open) ~ 1 (closed)
            }

        Returns:
            action: (8,) float32 — 7 joint targets + 1 gripper
        """
        import cv2

        # Re-query when chunk exhausted
        if self._chunk_buffer is None or self._chunk_idx >= self.n_action_steps:
            images = obs["images"]
            base_img  = images.get("base_camera",  np.zeros((H, W, 3), dtype=np.uint8))
            wrist_img = images.get("wrist_camera", np.zeros((H, W, 3), dtype=np.uint8))

            if base_img.shape[:2] != (H, W):
                base_img = cv2.resize(base_img, (W, H))
            if wrist_img.shape[:2] != (H, W):
                wrist_img = cv2.resize(wrist_img, (W, H))

            state = np.zeros(8, dtype=np.float32)
            state[:7] = obs["joint_positions"][:7]
            state[7]  = float(obs["gripper_state"])

            self.query_frames.append((base_img.copy(), wrist_img.copy()))

            pi_obs = {
                "observation/image":       base_img,
                "observation/wrist_image": wrist_img,
                "observation/state":       state,
                "prompt":                  self.task_description,
            }
            result = self._policy.infer(pi_obs)
            chunk = np.asarray(result["actions"], dtype=np.float32)
            if chunk.ndim == 3:
                chunk = chunk[0]  # remove batch dim → (action_horizon, 8)
            self._chunk_buffer = chunk
            self._chunk_idx = 0

        action = self._chunk_buffer[self._chunk_idx].copy()
        self._chunk_idx += 1
        return action

    def predict_chunk(self, obs: dict) -> np.ndarray:
        """Force inference and return n_action_steps actions as (N, 8) array.

        Used when the action buffer is managed externally (e.g. on the bridge),
        so the server always runs fresh inference and returns the full chunk.
        """
        self._chunk_buffer = None  # force re-inference
        self._chunk_idx = 0
        self.predict(obs)           # fills _chunk_buffer, advances _chunk_idx to 1
        self._chunk_idx = 0        # reset so buffer is clean
        return self._chunk_buffer[:self.n_action_steps].copy()

    def reset(self):
        self._chunk_buffer = None
        self._chunk_idx = 0
        self.query_frames = []
