"""Diffusion Policy wrapper for real robot inference.

Loads a diffusion_policy_xsim checkpoint and exposes predict()/predict_chunk().
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

from .base_policy import BasePolicy


class DPPolicy(BasePolicy):
    """Wraps diffusion_policy_xsim checkpoint for real robot deployment."""

    DEFAULT_JOINT_LOWER = np.array(
        [-2.9671, -2.0944, -2.9671, -2.0944, -2.9671, -2.0944, -3.0543],
        dtype=np.float32,
    )
    DEFAULT_JOINT_UPPER = np.array(
        [2.9671, 2.0944, 2.9671, 2.0944, 2.9671, 2.0944, 3.0543],
        dtype=np.float32,
    )

    def __init__(
        self,
        n_action_steps: int = 1,
        camera_to_feature_map: dict | None = None,
        gripper_binarize_threshold: float | None = None,
        max_joint_delta: float | None = None,
        joint_limits: dict | None = None,
        xsim_root: str = "/home/rl/projects/X-Sim/diffusion_policy_xsim",
    ):
        self.n_action_steps = int(max(1, n_action_steps))
        self.camera_to_feature_map = camera_to_feature_map or {
            "base_camera": "image1",
            "wrist_camera": "image2",
        }
        self.gripper_binarize_threshold = gripper_binarize_threshold
        self.max_joint_delta = max_joint_delta
        self.xsim_root = str(xsim_root)

        lower = self.DEFAULT_JOINT_LOWER.copy()
        upper = self.DEFAULT_JOINT_UPPER.copy()
        if joint_limits:
            if "lower" in joint_limits:
                lower = np.asarray(joint_limits["lower"], dtype=np.float32)
            if "upper" in joint_limits:
                upper = np.asarray(joint_limits["upper"], dtype=np.float32)
        self.joint_lower = lower
        self.joint_upper = upper

        self.policy = None
        self.dataset = None
        self.cfg = None
        self.device = "cuda"
        self.query_frames: list[tuple[np.ndarray, np.ndarray]] = []

    def load(self, checkpoint_path: str, device: str = "cuda"):
        self.device = device
        root = Path(self.xsim_root).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"diffusion_policy_xsim root not found: {root}")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))

        from scripts.dp_training_rgb import load_model

        ckpt = str(Path(checkpoint_path).expanduser().resolve())
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

        self.policy, self.dataset, self.cfg = load_model(ckpt, device, verbose=False)
        self.policy.eval()

        print(f"[DPPolicy] Loaded from {ckpt}")
        print(
            f"[DPPolicy] camera_views={self.dataset.camera_views}, "
            f"prop_dim={self.dataset.prop_dim}, action_dim={self.dataset.action_dim}, "
            f"n_action_steps={self.n_action_steps}"
        )
        print(f"[DPPolicy] Camera mapping: {self.camera_to_feature_map}")

    def _build_policy_obs(self, obs: dict):
        if self.dataset is None:
            raise RuntimeError("Policy not loaded. Call load() first.")

        state = np.zeros(self.dataset.prop_dim, dtype=np.float32)
        joints = np.asarray(obs["joint_positions"][:7], dtype=np.float32)
        state[:7] = joints
        if self.dataset.prop_dim >= 8:
            state[7] = float(obs["gripper_state"])

        obs_dict = {"proprio": state}
        images = obs["images"]
        for cam_name, feature_key in self.camera_to_feature_map.items():
            img = images.get(cam_name)
            if img is None:
                raise ValueError(
                    f"Camera '{cam_name}' not found in obs. "
                    f"Available: {list(images.keys())}"
                )
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            obs_dict[feature_key] = img

        base = images.get("base_camera")
        wrist = images.get("wrist_camera")
        if base is not None and wrist is not None:
            self.query_frames.append((base.copy(), wrist.copy()))

        processed = self.dataset.process_observation(obs_dict)
        for k, v in processed.items():
            processed[k] = v.to(self.device)
        return processed, joints

    def _sanitize_action(self, action: np.ndarray, current_joints: np.ndarray) -> np.ndarray:
        action = np.nan_to_num(action.astype(np.float32), copy=False)
        if action.shape[0] < 8:
            action = np.pad(action, (0, 8 - action.shape[0]), mode="constant")

        arm = action[:7].copy()
        arm = np.clip(arm, self.joint_lower, self.joint_upper)
        if self.max_joint_delta is not None and self.max_joint_delta > 0:
            lo = current_joints - self.max_joint_delta
            hi = current_joints + self.max_joint_delta
            arm = np.clip(arm, lo, hi)
        action[:7] = arm

        grip = float(np.clip(action[7], -1.0, 1.0))
        if self.gripper_binarize_threshold is not None:
            grip = 1.0 if grip >= self.gripper_binarize_threshold else -1.0
        action[7] = grip

        return action

    def predict_chunk(self, obs: dict) -> np.ndarray:
        processed_obs, cur_joints = self._build_policy_obs(obs)

        with torch.no_grad():
            action_seq = self.policy.act(processed_obs, cpu=True, sim=False)

        if torch.is_tensor(action_seq):
            action_seq = action_seq.detach().cpu().numpy()
        action_seq = np.asarray(action_seq, dtype=np.float32)
        if action_seq.ndim == 3:
            action_seq = action_seq[0]
        if action_seq.ndim == 1:
            action_seq = action_seq.reshape(1, -1)

        chunk = action_seq[: self.n_action_steps]
        if len(chunk) == 0:
            chunk = np.zeros((1, 8), dtype=np.float32)
        if len(chunk) < self.n_action_steps:
            pad = np.repeat(chunk[-1:], self.n_action_steps - len(chunk), axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)

        out = np.zeros((self.n_action_steps, 8), dtype=np.float32)
        running_joints = cur_joints.copy()
        for i in range(self.n_action_steps):
            out[i] = self._sanitize_action(chunk[i], running_joints)
            running_joints = out[i, :7]
        return out

    def predict(self, obs: dict) -> np.ndarray:
        return self.predict_chunk(obs)[0]

    def reset(self):
        self.query_frames = []

