"""ACT Policy wrapper for real robot inference.

Loads a LeRobot ACT checkpoint and provides predict() for the real robot pipeline.
The loading logic mirrors eval_act_paper.py to ensure checkpoint compatibility.
"""

import json
import tempfile
from pathlib import Path

import draccus
import numpy as np
import safetensors.torch
import torch

from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy as LeRobotACTPolicy

from .base_policy import BasePolicy


class ACTPolicy(BasePolicy):
    """Wraps LeRobot ACTPolicy for real robot deployment."""

    def __init__(self, camera_to_feature_map: dict | None = None):
        """
        Args:
            camera_to_feature_map: Maps real camera names to ACT input feature keys.
                e.g. {"base_camera": "image1", "side_camera": "image2"}
                If None, uses default mapping.
        """
        self.policy = None
        self.device = "cuda"
        self.camera_to_feature_map = camera_to_feature_map or {
            "base_camera": "image1",
            "side_camera": "image2",
        }

    def load(self, checkpoint_path: str, device: str = "cuda"):
        self.device = device
        ckpt = Path(checkpoint_path)

        # Parse config from checkpoint
        with open(ckpt / "config.json") as f:
            config_dict = json.load(f)
        config_dict.pop("type", None)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            json.dump(config_dict, tmp)
            tmp_path = tmp.name

        config = draccus.parse(ACTConfig, tmp_path, args=[])
        Path(tmp_path).unlink()

        # Build policy and load weights (with shape matching for edge cases)
        self.policy = LeRobotACTPolicy(config)

        model_sd = self.policy.state_dict()
        ckpt_sd = safetensors.torch.load_file(str(ckpt / "model.safetensors"))

        for key in ckpt_sd:
            if key not in model_sd:
                continue
            if ckpt_sd[key].shape == model_sd[key].shape:
                continue
            src = ckpt_sd[key]
            target_shape = model_sd[key].shape
            while src.ndim < len(target_shape):
                src = src.unsqueeze(0)
            ckpt_sd[key] = src.expand(target_shape).clone()

        self.policy.load_state_dict(ckpt_sd, strict=True)
        self.policy.to(device)
        self.policy.eval()

        # Extract image feature keys from config for validation
        self._image_feature_keys = list(config.image_features.keys())
        print(f"[ACTPolicy] Loaded from {ckpt}")
        print(f"[ACTPolicy] Image features: {self._image_feature_keys}")
        print(f"[ACTPolicy] Camera mapping: {self.camera_to_feature_map}")

    def predict(self, obs: dict) -> np.ndarray:
        """Convert real robot observation to ACT input format and predict.

        Args:
            obs: {
                "images": {"base_camera": (H,W,3) uint8, "side_camera": ...},
                "joint_positions": (7,) float32,
                "gripper_state": float,
            }

        Returns:
            action: (8,) float32 — 7 joint targets + 1 gripper
        """
        # Build state vector: 7 arm joints + 1 gripper
        state = np.zeros(8, dtype=np.float32)
        state[:7] = obs["joint_positions"][:7]
        state[7] = float(obs["gripper_state"])

        obs_dict = {
            "observation.state": torch.tensor(state, dtype=torch.float32)
                .unsqueeze(0).to(self.device),
        }

        # Map camera images to ACT feature keys
        for cam_name, feature_key in self.camera_to_feature_map.items():
            img = obs["images"].get(cam_name)
            if img is None:
                raise ValueError(
                    f"Camera '{cam_name}' not found in obs. "
                    f"Available: {list(obs['images'].keys())}"
                )
            # (H,W,3) uint8 -> (1,3,H,W) float32 [0,1]
            tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
            obs_dict[feature_key] = tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.policy.select_action(obs_dict)

        action_np = action.cpu().numpy().flatten()

        # Binarize gripper: > 0 → close (1.0), else → open (-1.0)
        action_np[7] = 1.0 if action_np[7] > 0 else -1.0

        return action_np.astype(np.float32)

    def reset(self):
        if self.policy is not None:
            self.policy.reset()
