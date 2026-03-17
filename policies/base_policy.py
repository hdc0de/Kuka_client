"""Abstract base class for all policies."""

from abc import ABC, abstractmethod
import numpy as np


class BasePolicy(ABC):
    """Base interface for robot control policies.

    Subclasses must implement load() and predict().
    Optionally override reset() for stateful policies (e.g., temporal ensemble).
    """

    @abstractmethod
    def load(self, checkpoint_path: str, device: str = "cuda"):
        """Load model weights from checkpoint."""
        ...

    @abstractmethod
    def predict(self, obs: dict) -> np.ndarray:
        """Predict action from observation.

        Args:
            obs: {
                "images": {cam_name: np.ndarray (H,W,3) uint8, ...},
                "joint_positions": np.ndarray (7,) float32,
                "gripper_state": float,
            }

        Returns:
            action: np.ndarray (8,) — 7 joint targets + 1 gripper
        """
        ...

    def reset(self):
        """Reset internal state at episode start. Override for stateful policies."""
        pass


class RandomPolicy(BasePolicy):
    """Random policy for testing communication pipeline."""

    def __init__(self, action_dim: int = 8):
        self.action_dim = action_dim

    def load(self, checkpoint_path: str, device: str = "cuda"):
        pass  # nothing to load

    def predict(self, obs: dict) -> np.ndarray:
        return np.random.uniform(-0.05, 0.05, size=(self.action_dim,)).astype(np.float32)


class GoHomePolicy(BasePolicy):
    """Move to home position, then back to initial position."""

    def __init__(self, home_position=None, gripper_cycles=3, steps_per_grip=15):
        self._home = np.array(
            home_position or [0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, 0.0],
            dtype=np.float32,
        )
        self._initial_pos = None
        self._phase = "go_home"  # go_home → grip → return
        self._pos_threshold = 0.05  # rad
        self._gripper_cycles = gripper_cycles      # number of open/close cycles
        self._steps_per_grip = steps_per_grip      # steps per half-cycle
        self._grip_step = 0
        self._grip_cycle = 0

    def load(self, checkpoint_path: str, device: str = "cuda"):
        pass

    def reset(self):
        self._initial_pos = None
        self._phase = "go_home"
        self._grip_step = 0
        self._grip_cycle = 0

    def predict(self, obs: dict) -> np.ndarray:
        joints = obs["joint_positions"][:7].astype(np.float32)

        # Record initial position on first call
        if self._initial_pos is None:
            self._initial_pos = joints.copy()
            print(f"[GoHomePolicy] Initial position: {self._initial_pos}")

        if self._phase == "go_home":
            err = np.max(np.abs(joints - self._home))
            if err < self._pos_threshold:
                self._phase = "grip"
                self._grip_step = 0
                self._grip_cycle = 0
                print(f"[GoHomePolicy] Reached home, starting gripper test "
                      f"({self._gripper_cycles} cycles)")
            return np.concatenate([self._home, [-1.0]])  # open

        elif self._phase == "grip":
            # Alternate close(1.0) / open(-1.0) every steps_per_grip steps
            half = self._grip_step // self._steps_per_grip  # 0,1,2,3,4,5...
            grip_cmd = 1.0 if half % 2 == 0 else -1.0      # close first
            self._grip_step += 1

            # Check if all cycles done (each cycle = close + open = 2 halves)
            if half >= self._gripper_cycles * 2:
                self._phase = "return"
                print(f"[GoHomePolicy] Gripper test done, returning to initial")
                return np.concatenate([self._initial_pos, [-1.0]])

            if self._grip_step % self._steps_per_grip == 1:
                action = "CLOSE" if grip_cmd > 0 else "OPEN"
                print(f"[GoHomePolicy] Cycle {half // 2 + 1}/{self._gripper_cycles} "
                      f"— {action}")

            return np.concatenate([self._home, [grip_cmd]])

        else:  # return
            return np.concatenate([self._initial_pos, [-1.0]])
