"""Robot interface abstraction layer.

- RobotInterface: ABC for all robots
- DummyRobot: for testing without hardware
- KukaIIWAInterface: TCP socket client → talks to ros_bridge.py on the ROS 1 machine
"""

import json
import socket
import struct
from abc import ABC, abstractmethod

import numpy as np


class RobotInterface(ABC):
    """Abstract base class for robot control."""

    @abstractmethod
    def get_joint_positions(self) -> np.ndarray:
        """Return current joint positions (7,) float32."""
        ...

    @abstractmethod
    def get_gripper_state(self) -> float:
        """Return gripper state as a scalar."""
        ...

    @abstractmethod
    def execute_action(self, action: np.ndarray):
        """Execute action (8,): 7 joint targets + 1 gripper."""
        ...

    @abstractmethod
    def go_home(self):
        """Move robot to home position."""
        ...

    @abstractmethod
    def stop(self):
        """Emergency stop / clean shutdown."""
        ...


class DummyRobot(RobotInterface):
    """Fake robot for testing the communication pipeline."""

    def __init__(self, home_position: list[float] | None = None):
        self._joints = np.zeros(7, dtype=np.float32)
        self._gripper = 0.0
        self._home = np.array(
            home_position or [0, 0, 0, -1.57, 0, 1.57, 0],
            dtype=np.float32
        )
        print("[DummyRobot] Initialized")

    def get_joint_positions(self) -> np.ndarray:
        return self._joints.copy()

    def get_gripper_state(self) -> float:
        return self._gripper

    def execute_action(self, action: np.ndarray):
        self._joints = action[:7].astype(np.float32)
        self._gripper = float(action[7])

    def go_home(self):
        self._joints = self._home.copy()
        self._gripper = 0.0
        print("[DummyRobot] Moved to home position")

    def stop(self):
        print("[DummyRobot] Stopped")


# ---------------------------------------------------------------------------
# TCP protocol helpers (shared with ros_bridge.py)
# Message format: 4-byte length prefix (big-endian) + JSON body
# ---------------------------------------------------------------------------

def _send_msg(sock: socket.socket, data: dict):
    """Send length-prefixed JSON message."""
    body = json.dumps(data).encode()
    sock.sendall(struct.pack(">I", len(body)) + body)


def _recv_msg(sock: socket.socket) -> dict:
    """Receive length-prefixed JSON message."""
    header = _recv_exact(sock, 4)
    length = struct.unpack(">I", header)[0]
    body = _recv_exact(sock, length)
    return json.loads(body.decode())


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly n bytes from socket."""
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed unexpectedly")
        buf += chunk
    return buf


class KukaIIWAInterface(RobotInterface):
    """KUKA iiwa interface via TCP socket to ros_bridge.py.

    This runs on your Ubuntu 24.04 machine (no ROS needed).
    It connects to ros_bridge.py running on the ROS 1 machine.

    Protocol (JSON over TCP, length-prefixed):
        Request:  {"cmd": "get_state"}
        Response: {"joint_positions": [7 floats], "gripper_state": float}

        Request:  {"cmd": "execute", "action": [8 floats]}
        Response: {"ok": true}

        Request:  {"cmd": "go_home"}
        Response: {"ok": true}

        Request:  {"cmd": "stop"}
        Response: {"ok": true}
    """

    def __init__(self, config: dict):
        self._home = np.array(
            config.get("home_position", [0, 0, 0, -1.57, 0, 1.57, 0]),
            dtype=np.float32
        )
        host = config.get("bridge_host", "192.168.1.100")
        port = config.get("bridge_port", 6000)

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(5.0)
        self._sock.connect((host, port))
        print(f"[KukaIIWA] Connected to ros_bridge at {host}:{port}")

    def _request(self, msg: dict) -> dict:
        _send_msg(self._sock, msg)
        return _recv_msg(self._sock)

    def get_joint_positions(self) -> np.ndarray:
        resp = self._request({"cmd": "get_state"})
        return np.array(resp["joint_positions"], dtype=np.float32)

    def get_gripper_state(self) -> float:
        resp = self._request({"cmd": "get_state"})
        return float(resp["gripper_state"])

    def execute_action(self, action: np.ndarray):
        self._request({"cmd": "execute", "action": action.tolist()})

    def go_home(self):
        self._request({"cmd": "go_home"})
        print("[KukaIIWA] Moved to home position")

    def stop(self):
        try:
            self._request({"cmd": "stop"})
        except Exception:
            pass
        self._sock.close()
        print("[KukaIIWA] Stopped and disconnected")
