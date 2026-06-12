"""Original ACT adapter for the 10 Hz qpos + binary-gripper dataset."""

from __future__ import annotations

import numpy as np

from .original_act_policy import OriginalACTPolicy


class QPos10HzBinaryOriginalACTPolicy(OriginalACTPolicy):
    """Match inference preprocessing to bag2hdf5_qpos_10hz_binary_strict.py."""

    def __init__(
        self,
        *args,
        gripper_closed_width_threshold: float = 0.068,
        binary_action_threshold: float = 0.5,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gripper_closed_width_threshold = float(
            gripper_closed_width_threshold
        )
        self.binary_action_threshold = float(binary_action_threshold)

        if not (
            self.gripper_width_min
            <= self.gripper_closed_width_threshold
            <= self.gripper_width_max
        ):
            raise ValueError(
                "gripper_closed_width_threshold must be inside the configured "
                "gripper width range"
            )
        if not 0.0 <= self.binary_action_threshold <= 1.0:
            raise ValueError("binary_action_threshold must be in [0, 1]")

    def load(self, checkpoint_path: str, device: str = "cuda"):
        try:
            super().load(checkpoint_path, device)
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load the 10 Hz qpos ACT checkpoint. Verify that "
                "act_chunk_size, camera names, state_dim=8, and the model "
                "architecture match the training command."
            ) from exc

        expected_shape = (8,)
        for key in ("qpos_mean", "qpos_std", "action_mean", "action_std"):
            value = np.asarray(self.stats.get(key))
            if value.shape != expected_shape:
                raise ValueError(
                    f"{key} must have shape {expected_shape} for "
                    f"[x,y,z,qx,qy,qz,qw,gripper], got {value.shape}"
                )

        print(
            "[QPos10HzBinaryOriginalACTPolicy] data contract: "
            "10 Hz, absolute Cartesian action, binary gripper "
            f"(width < {self.gripper_closed_width_threshold:.3f} m -> closed, "
            f"model output >= {self.binary_action_threshold:.2f} -> closed)"
        )
        print(
            "[QPos10HzBinaryOriginalACTPolicy] checkpoint gripper stats: "
            f"qpos_mean={float(self.stats['qpos_mean'][7]):.4f}, "
            f"qpos_std={float(self.stats['qpos_std'][7]):.4f}, "
            f"action_mean={float(self.stats['action_mean'][7]):.4f}, "
            f"action_std={float(self.stats['action_std'][7]):.4f}"
        )

    def _width_to_binary_closed(self, value: float) -> float:
        width = float(
            np.clip(value, self.gripper_width_min, self.gripper_width_max)
        )
        return 1.0 if width < self.gripper_closed_width_threshold else 0.0

    def _binary_closed_to_width(self, value: float) -> float:
        closed = float(value) >= self.binary_action_threshold
        return self.gripper_width_min if closed else self.gripper_width_max
