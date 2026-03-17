"""Stage 2 test: verify ACT policy loads and inference works end-to-end.

Runs server + client in a single process with DummyRobot + DummyCamera.
No interactive prompts — just runs 10 steps and prints results.
"""

import time
import yaml
import numpy as np
import zmq
import threading

from camera import CameraConfig, DummyCamera
from robot_interface import DummyRobot
from msg_utils import pack_observation, unpack_observation, pack_action, unpack_action


def run_server(config, ready_event, stop_event):
    """Server thread: load policy, respond to observations."""
    from policies.act_policy import ACTPolicy
    from policies.base_policy import RandomPolicy

    policy_name = config["server"]["policy"]
    checkpoint = config["server"].get("checkpoint", "")
    device = config["server"].get("device", "cuda")

    if policy_name == "act":
        camera_map = config["server"].get("camera_to_feature_map", None)
        policy = ACTPolicy(camera_to_feature_map=camera_map)
        policy.load(checkpoint, device)
    else:
        policy = RandomPolicy()
        policy.load(checkpoint, device)

    address = config["server"]["address"]
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(address)
    print(f"[Server] Listening on {address}")
    ready_event.set()

    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)

    step = 0
    try:
        while not stop_event.is_set():
            events = dict(poller.poll(timeout=500))
            if sock not in events:
                continue

            obs_data = sock.recv()
            obs = unpack_observation(obs_data)

            if obs.get("reset", False):
                policy.reset()
                print("[Server] Episode reset")
                sock.send(pack_action(np.zeros(8, dtype=np.float32), done=False))
                step = 0
                continue

            t0 = time.time()
            action = policy.predict(obs)
            dt = time.time() - t0
            step += 1

            if step <= 3 or step % 5 == 0:
                print(f"  [Server step {step:3d}] inference: {dt*1000:.1f}ms, "
                      f"action[:4]={action[:4]}, grip={action[7]:.3f}")

            sock.send(pack_action(action, done=False))
    finally:
        sock.close()
        ctx.term()
        print("[Server] Shut down")


def main():
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # Setup
    cam_configs = [
        CameraConfig(name=c["name"], width=c.get("width", 640),
                     height=c.get("height", 480), mode=c.get("mode", "rgb"))
        for c in config["cameras"]
    ]
    cameras = DummyCamera(cam_configs)
    robot = DummyRobot(home_position=config["robot"].get("home_position"))

    # Start server in background thread
    ready = threading.Event()
    stop = threading.Event()
    server_thread = threading.Thread(target=run_server, args=(config, ready, stop), daemon=True)
    server_thread.start()
    ready.wait(timeout=60)

    # ZMQ client
    address = config["server"]["address"]
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.connect(address)
    print(f"[Client] Connected to {address}")

    hz = config.get("control", {}).get("hz", 10)
    dt = 1.0 / hz
    n_steps = 20  # just 20 steps to verify

    # Send reset
    robot.go_home()
    images = cameras.capture()
    joints = robot.get_joint_positions()
    gripper = robot.get_gripper_state()
    sock.send(pack_observation(images, joints, gripper, reset=True))
    sock.recv()

    print(f"\n[Client] Running {n_steps} steps at {hz} Hz...")
    latencies = []

    for step in range(n_steps):
        t0 = time.time()

        images = cameras.capture()
        joints = robot.get_joint_positions()
        gripper = robot.get_gripper_state()

        sock.send(pack_observation(images, joints, gripper))
        action_data = sock.recv()
        action, done = unpack_action(action_data)

        robot.execute_action(action)

        elapsed = time.time() - t0
        latencies.append(elapsed)
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    # Summary
    latencies = np.array(latencies)
    print(f"\n{'='*50}")
    print(f"Stage 2 Test PASSED")
    print(f"{'='*50}")
    print(f"Policy: {config['server']['policy']}")
    print(f"Checkpoint: {config['server']['checkpoint']}")
    print(f"Steps: {n_steps}")
    print(f"Latency: mean={latencies.mean()*1000:.1f}ms, "
          f"max={latencies.max()*1000:.1f}ms, "
          f"min={latencies.min()*1000:.1f}ms")
    print(f"Effective Hz: {1.0/latencies.mean():.1f}")
    print(f"Action shape: {action.shape}, dtype: {action.dtype}")
    print(f"Last action: {action}")
    print(f"{'='*50}")

    # Cleanup
    stop.set()
    sock.close()
    ctx.term()
    cameras.stop()
    robot.stop()
    server_thread.join(timeout=5)


if __name__ == "__main__":
    main()
