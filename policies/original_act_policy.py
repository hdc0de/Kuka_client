"""Original ACT policy wrapper for real robot inference.

This wrapper targets the original ACT repository layout, not LeRobot.
It expects a checkpoint directory containing:
  - policy_best.ckpt
  - dataset_stats.pkl

Important:
  The state/action semantics follow the ACT training setup. If your ACT model
  was trained on Cartesian pose + gripper width, the bridge side must provide
  that same representation to the server. This wrapper only handles model-side
  loading, normalization, image packing, and optional gripper conversion.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

from .base_policy import BasePolicy


class OriginalACTPolicy(BasePolicy):
    """Adapter from this repo's BasePolicy API to the original ACT model.

    The original ACT class from `my_ACT/policy.py` is still the model that runs
    inference. This wrapper only translates between:
      - server observations: dict(images, joint_positions, gripper_state)
      - ACT inputs: qpos tensor + stacked camera tensor
      - server actions: numpy action chunks
    """

    def __init__(
        self,
        act_root: str,
        chunk_size: int,
        hidden_dim: int = 512,
        dim_feedforward: int = 3200,
        kl_weight: int = 10,
        n_action_steps: int | None = None,
        temporal_agg: bool = False,
        ensemble_decay: float = 0.01,
        camera_names: list[str] | None = None,
        camera_sources: list[str] | None = None,
        image_height: int = 480,
        image_width: int = 640,
        state_key: str = "joint_positions",
        action_pose_mode: str = "absolute",
        input_gripper_mode: str = "normalized",
        output_gripper_mode: str = "normalized",
        gripper_width_min: float = 0.0,
        gripper_width_max: float = 0.085,
        debug_actions: bool = False,
        debug_interval: int = 50,
    ):
        self.act_root = str(Path(act_root).expanduser())
        self.chunk_size = int(chunk_size)
        self.hidden_dim = int(hidden_dim)
        self.dim_feedforward = int(dim_feedforward)
        self.kl_weight = int(kl_weight)
        self.n_action_steps = int(max(1, n_action_steps or chunk_size))
        self.temporal_agg = bool(temporal_agg)
        self.ensemble_decay = float(ensemble_decay)

        self.camera_names = list(camera_names or ["front", "side"])
        self.camera_sources = list(camera_sources or self.camera_names)
        if len(self.camera_names) != len(self.camera_sources):
            raise ValueError(
                "camera_names and camera_sources must have the same length"
            )

        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.state_key = state_key
        self.action_pose_mode = action_pose_mode
        self.input_gripper_mode = input_gripper_mode
        self.output_gripper_mode = output_gripper_mode
        self.gripper_width_min = float(gripper_width_min)
        self.gripper_width_max = float(gripper_width_max)
        self.debug_actions = bool(debug_actions)
        self.debug_interval = int(max(1, debug_interval))

        self.policy = None
        self.stats = None
        self.device = "cuda"
        self.query_frames: list[tuple[np.ndarray, np.ndarray]] = []
        self._step = 0
        self._action_history: list[np.ndarray] = []
        self._debug_action_count = 0

    def load(self, checkpoint_path: str, device: str = "cuda"):
        act_root = Path(self.act_root).resolve()
        if not act_root.exists():
            raise FileNotFoundError(f"ACT repo root not found: {act_root}")
        if str(act_root) not in sys.path:
            sys.path.insert(0, str(act_root))

        # Import lazily after sys.path is prepared, so this wrapper can live in
        # Kuka_client while the original ACT repository remains outside it.
        from policy import ACTPolicy as OriginalACTModel

        self.device = str(
            torch.device(device if device == "cpu" or torch.cuda.is_available() else "cpu")
        )

        ckpt_dir = Path(checkpoint_path).expanduser().resolve()
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {ckpt_dir}")

        ckpt_file = ckpt_dir / "policy_best.ckpt"
        stats_file = ckpt_dir / "dataset_stats.pkl"
        if not ckpt_file.exists():
            raise FileNotFoundError(f"ACT checkpoint not found: {ckpt_file}")
        if not stats_file.exists():
            raise FileNotFoundError(f"ACT stats file not found: {stats_file}")

        policy_config = {
            "lr": 1e-5,
            "num_queries": self.chunk_size,
            "kl_weight": self.kl_weight,
            "hidden_dim": self.hidden_dim,
            "dim_feedforward": self.dim_feedforward,
            "lr_backbone": 1e-5,
            "backbone": "resnet18",
            "enc_layers": 4,
            "dec_layers": 7,
            "nheads": 8,
            "state_dim": 8,
            "camera_names": self.camera_names,
        }
        # The original ACT builder calls argparse.parse_args() internally.
        # Hide the Kuka_client server CLI args and provide the required ACT
        # placeholders so model construction stays side-effect free here.
        old_argv = sys.argv[:]
        sys.argv = [
            old_argv[0],
            "--ckpt_dir", str(ckpt_dir),
            "--policy_class", "ACT",
            "--task_name", "real_robot",
            "--seed", "0",
            "--num_epochs", "1",
        ]
        try:
            self.policy = OriginalACTModel(policy_config)
        finally:
            sys.argv = old_argv

        state_dict = torch.load(ckpt_file, map_location=self.device)
        self.policy.load_state_dict(state_dict)
        self.policy.to(self.device)
        self.policy.eval()

        with open(stats_file, "rb") as f:
            self.stats = pickle.load(f)

        print(f"[OriginalACTPolicy] Loaded checkpoint from {ckpt_dir}")
        print(
            f"[OriginalACTPolicy] cameras={self.camera_names} "
            f"sources={self.camera_sources} chunk_size={self.chunk_size} "
            f"n_action_steps={self.n_action_steps} temporal_agg={self.temporal_agg} "
            f"action_pose_mode={self.action_pose_mode}"
        )

    def _normalize_qpos(self, qpos: np.ndarray) -> np.ndarray:
        qpos_mean = np.asarray(self.stats["qpos_mean"], dtype=np.float32)
        qpos_std = np.asarray(self.stats["qpos_std"], dtype=np.float32)
        qpos_std = np.where(np.abs(qpos_std) < 1e-6, 1.0, qpos_std)
        return (qpos.astype(np.float32) - qpos_mean) / qpos_std

    def _denormalize_action(self, action: np.ndarray) -> np.ndarray:
        action_mean = np.asarray(self.stats["action_mean"], dtype=np.float32)
        action_std = np.asarray(self.stats["action_std"], dtype=np.float32)
        return action.astype(np.float32) * action_std + action_mean

    def _normalized_to_width(self, value: float) -> float:
        alpha = (float(value) + 1.0) * 0.5
        return self.gripper_width_min + alpha * (
            self.gripper_width_max - self.gripper_width_min
        )

    def _width_to_normalized(self, value: float) -> float:
        span = self.gripper_width_max - self.gripper_width_min
        if span <= 1e-6:
            return -1.0
        alpha = (float(value) - self.gripper_width_min) / span
        return float(np.clip(alpha * 2.0 - 1.0, -1.0, 1.0))

    def _width_to_binary_closed(self, value: float) -> float:
        span = self.gripper_width_max - self.gripper_width_min
        if span <= 1e-6:
            return 1.0
        alpha_open = (float(value) - self.gripper_width_min) / span
        return float(np.clip(1.0 - alpha_open, 0.0, 1.0))

    def _binary_closed_to_width(self, value: float) -> float:
        closed = float(np.clip(value, 0.0, 1.0))
        return self.gripper_width_max - closed * (
            self.gripper_width_max - self.gripper_width_min
        )

    @staticmethod
    def _normalize_quaternion(quat: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(quat)
        if norm > 1e-6:
            return quat / norm
        return fallback.copy()

    @classmethod
    def _quat_multiply(cls, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        x1, y1, z1, w1 = cls._normalize_quaternion(
            np.asarray(q1, dtype=np.float32), np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        )
        x2, y2, z2, w2 = cls._normalize_quaternion(
            np.asarray(q2, dtype=np.float32), np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        )
        out = np.array(
            [
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ],
            dtype=np.float32,
        )
        return cls._normalize_quaternion(out, np.asarray(q1, dtype=np.float32))

    def _build_qpos(self, obs: dict) -> np.ndarray:
        state = obs.get(self.state_key)
        if state is None:
            available = sorted(obs.keys())
            raise ValueError(
                f"State key '{self.state_key}' not found in obs. Available: {available}"
            )

        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape[0] < 7:
            raise ValueError(
                f"State '{self.state_key}' must contain at least 7 values, got {state.shape[0]}"
            )

        gripper = float(obs["gripper_state"])
        if self.input_gripper_mode == "normalized":
            gripper = self._normalized_to_width(gripper)
        elif self.input_gripper_mode == "binary":
            gripper = self._width_to_binary_closed(gripper)
        elif self.input_gripper_mode != "width":
            raise ValueError(
                f"Unsupported input_gripper_mode: {self.input_gripper_mode}"
            )

        qpos = np.zeros(8, dtype=np.float32)
        qpos[:7] = state[:7]
        qpos[7] = gripper
        return qpos

    def _build_image_tensor(self, obs: dict) -> torch.Tensor:
        frames = []
        stacked_frames = []
        images = obs["images"]

        for cam_name in self.camera_sources:
            img = images.get(cam_name)
            if img is None:
                raise ValueError(
                    f"Camera '{cam_name}' not found in obs. Available: {list(images.keys())}"
                )
            if img.shape[:2] != (self.image_height, self.image_width):
                img = cv2.resize(img, (self.image_width, self.image_height))
            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)
            frames.append(img.copy())
            tensor = torch.from_numpy(img).float() / 255.0
            stacked_frames.append(tensor.permute(2, 0, 1))

        if len(frames) >= 2:
            self.query_frames.append((frames[0], frames[1]))

        image_tensor = torch.stack(stacked_frames, dim=0).unsqueeze(0)
        return image_tensor.to(self.device)

    def _convert_pose_output(self, action: np.ndarray, qpos: np.ndarray) -> np.ndarray:
        mode = self.action_pose_mode
        if mode in ("absolute", "absolute_pose"):
            return action
        if mode in ("relative", "relative_pose"):
            out = action.copy()
            out[:3] = qpos[:3] + action[:3]
            out[3:7] = self._quat_multiply(qpos[3:7], action[3:7])
            return out
        raise ValueError(f"Unsupported action_pose_mode: {self.action_pose_mode}")

    def _convert_gripper_output(self, action: np.ndarray) -> np.ndarray:
        if action.shape[0] < 8:
            action = np.pad(action, (0, 8 - action.shape[0]), mode="constant")

        if self.output_gripper_mode == "normalized":
            action[7] = self._width_to_normalized(action[7])
        elif self.output_gripper_mode == "binary":
            action[7] = self._binary_closed_to_width(action[7])
        elif self.output_gripper_mode == "width":
            action[7] = float(
                np.clip(action[7], self.gripper_width_min, self.gripper_width_max)
            )
        else:
            raise ValueError(
                f"Unsupported output_gripper_mode: {self.output_gripper_mode}"
            )
        return action

    def _infer_raw_action_sequence(self, obs: dict) -> tuple[np.ndarray, np.ndarray]:
        """Run the original ACT model and return raw normalized actions."""
        qpos = self._build_qpos(obs)
        qpos_norm = self._normalize_qpos(qpos)
        qpos_tensor = torch.from_numpy(qpos_norm).float().unsqueeze(0).to(self.device)
        image_tensor = self._build_image_tensor(obs)

        with torch.inference_mode():
            action_seq = self.policy(qpos_tensor, image_tensor)

        if torch.is_tensor(action_seq):
            action_seq = action_seq.detach().cpu().numpy()
        action_seq = np.asarray(action_seq, dtype=np.float32)
        if action_seq.ndim == 3:
            action_seq = action_seq[0]
        if action_seq.ndim == 1:
            action_seq = action_seq.reshape(1, -1)
        return action_seq, qpos

    def _postprocess_action(self, raw_action: np.ndarray, qpos: np.ndarray) -> np.ndarray:
        """Convert one raw ACT action into the bridge action representation."""
        rel_action = self._denormalize_action(raw_action).astype(np.float32, copy=False)
        abs_action = self._convert_pose_output(rel_action, qpos)
        out = self._convert_gripper_output(abs_action.astype(np.float32, copy=False))

        self._debug_action_count += 1
        if self.debug_actions and self._debug_action_count % self.debug_interval == 1:
            print(
                "[OriginalACTPolicy][debug] "
                f"qpos_xyz={np.array2string(qpos[:3], precision=4, suppress_small=True)} "
                f"rel_xyz={np.array2string(rel_action[:3], precision=5, suppress_small=True)} "
                f"target_xyz={np.array2string(out[:3], precision=4, suppress_small=True)} "
                f"rel_gripper={float(rel_action[7]):.4f} target_gripper={float(out[7]):.4f}"
            )
        return out

    def _temporal_ensemble_action(self, action_seq: np.ndarray) -> np.ndarray:
        """Mirror the temporal aggregation logic from act_kuka_inference2.py."""
        self._action_history.append(action_seq.copy())

        actions_for_curr_step = []
        for query_step, seq in enumerate(self._action_history):
            offset = self._step - query_step
            if 0 <= offset < len(seq):
                actions_for_curr_step.append(seq[offset])
        if not actions_for_curr_step:
            actions_for_curr_step = [action_seq[0]]

        stacked = np.asarray(actions_for_curr_step, dtype=np.float32)
        weights = np.exp(
            -self.ensemble_decay * np.arange(len(stacked), dtype=np.float32)
        )
        weights = weights / max(weights.sum(), 1e-6)
        blended = (stacked * weights[:, None]).sum(axis=0)
        self._step += 1
        return blended

    def predict_chunk(self, obs: dict) -> np.ndarray:
        if self.policy is None or self.stats is None:
            raise RuntimeError("Policy not loaded. Call load() first.")

        if self.temporal_agg and self.n_action_steps != 1:
            raise ValueError(
                "temporal_agg=True requires n_action_steps=1 to match the original ACT loop"
            )

        action_seq, qpos = self._infer_raw_action_sequence(obs)

        if self.temporal_agg:
            out = np.zeros((1, 8), dtype=np.float32)
            out[0] = self._postprocess_action(
                self._temporal_ensemble_action(action_seq), qpos
            )
            return out

        chunk = action_seq[: self.n_action_steps]
        if len(chunk) == 0:
            chunk = np.zeros((1, 8), dtype=np.float32)
        if len(chunk) < self.n_action_steps:
            pad = np.repeat(chunk[-1:], self.n_action_steps - len(chunk), axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)

        out = np.zeros((self.n_action_steps, 8), dtype=np.float32)
        for i in range(self.n_action_steps):
            out[i] = self._postprocess_action(chunk[i], qpos)
        return out

    def predict(self, obs: dict) -> np.ndarray:
        return self.predict_chunk(obs)[0]

    def reset(self):
        self.query_frames = []
        self._step = 0
        self._action_history = []
        self._debug_action_count = 0
