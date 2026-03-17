"""Robot control client.

Collects observations (cameras + robot joints), sends to server, executes returned actions.

Usage:
    python client.py --config config.yaml
    python client.py --config config.yaml --dummy   # test without real hardware
"""

import argparse
import time

import numpy as np
import yaml
import zmq

from camera import CameraConfig, CameraManager, DummyCamera
from msg_utils import pack_observation, unpack_action
from robot_interface import DummyRobot, KukaIIWAInterface


def build_cameras(config: dict, dummy: bool = False):
    """Build camera manager from config."""
    cam_configs = [
        CameraConfig(
            name=c["name"],
            serial=c.get("serial", ""),
            width=c.get("width", 640),
            height=c.get("height", 480),
            fps=c.get("fps", 30),
            mode=c.get("mode", "rgb"),
        )
        for c in config["cameras"]
    ]
    if dummy:
        return DummyCamera(cam_configs)
    return CameraManager(cam_configs)


def build_robot(config: dict, dummy: bool = False):
    """Build robot interface from config."""
    robot_cfg = config["robot"]
    if dummy:
        return DummyRobot(home_position=robot_cfg.get("home_position"))
    return KukaIIWAInterface(robot_cfg)


def main():
    parser = argparse.ArgumentParser(description="Robot control client")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--dummy", action="store_true",
                        help="Use DummyRobot + DummyCamera for testing")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    control_cfg = config.get("control", {})
    hz = control_cfg.get("hz", 10)
    max_steps = control_cfg.get("max_steps", 300)
    num_episodes = control_cfg.get("num_episodes", 10)
    dt = 1.0 / hz

    # Setup hardware
    cameras = build_cameras(config, dummy=args.dummy)
    robot = build_robot(config, dummy=args.dummy)

    # Setup ZMQ
    address = config["server"].get("connect_address", config["server"]["address"])
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REQ)
    socket.connect(address)
    print(f"[Client] Connected to {address}")

    try:
        for ep in range(num_episodes):
            input(f"\n[Client] Episode {ep+1}/{num_episodes} — Press Enter to start...")
            robot.go_home()

            # Send reset signal
            images = cameras.capture()
            joints = robot.get_joint_positions()
            gripper = robot.get_gripper_state()
            socket.send(pack_observation(images, joints, gripper, reset=True))
            socket.recv()  # consume reset ack

            print(f"[Client] Running episode {ep+1} (max {max_steps} steps, {hz} Hz)")
            success_steps = 0

            for step in range(max_steps):
                t0 = time.time()

                # 1. Capture observations
                images = cameras.capture()
                joints = robot.get_joint_positions()
                gripper = robot.get_gripper_state()

                # 2. Send to server
                socket.send(pack_observation(images, joints, gripper))

                # 3. Receive action
                action_data = socket.recv()
                action, done = unpack_action(action_data)

                # 4. Execute on robot
                robot.execute_action(action)

                # 5. Rate limiting
                elapsed = time.time() - t0
                sleep_time = dt - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                actual_hz = 1.0 / (time.time() - t0)
                if step % 50 == 0:
                    print(f"  [step {step:4d}] joints[:3]={joints[:3]}, "
                          f"hz={actual_hz:.1f}")

                if done:
                    print(f"  [step {step}] Policy signaled done")
                    break

            print(f"[Client] Episode {ep+1} finished ({step+1} steps)")

    except KeyboardInterrupt:
        print("\n[Client] Interrupted")
    finally:
        robot.stop()
        cameras.stop()
        socket.close()
        ctx.term()
        print("[Client] Shutdown complete")


if __name__ == "__main__":
    main()
