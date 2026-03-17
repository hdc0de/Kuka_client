"""ZMQ message serialization utilities using msgpack + numpy + JPEG."""

import time
import struct
import cv2
import numpy as np
import msgpack


def encode_image(image: np.ndarray, quality: int = 90) -> bytes:
    """Encode RGB image to JPEG bytes."""
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                          [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()


def decode_image(data: bytes) -> np.ndarray:
    """Decode JPEG bytes to RGB numpy array."""
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def encode_depth(depth: np.ndarray) -> bytes:
    """Encode uint16 depth image to PNG bytes."""
    _, buf = cv2.imencode(".png", depth)
    return buf.tobytes()


def decode_depth(data: bytes) -> np.ndarray:
    """Decode PNG bytes to uint16 depth array."""
    buf = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)


def _np_to_bytes(arr: np.ndarray) -> bytes:
    """Serialize numpy array: 1-byte dtype_len + dtype + 4-byte ndim + shape + data."""
    dtype_str = arr.dtype.str.encode()
    header = struct.pack("B", len(dtype_str)) + dtype_str
    header += struct.pack("<I", arr.ndim)
    for s in arr.shape:
        header += struct.pack("<I", s)
    return header + arr.tobytes()


def _bytes_to_np(data: bytes) -> np.ndarray:
    """Deserialize numpy array from bytes."""
    offset = 0
    dtype_len = struct.unpack("B", data[offset:offset + 1])[0]
    offset += 1
    dtype_str = data[offset:offset + dtype_len].decode()
    offset += dtype_len
    ndim = struct.unpack("<I", data[offset:offset + 4])[0]
    offset += 4
    shape = []
    for _ in range(ndim):
        shape.append(struct.unpack("<I", data[offset:offset + 4])[0])
        offset += 4
    return np.frombuffer(data[offset:], dtype=np.dtype(dtype_str)).reshape(shape).copy()


def pack_observation(images: dict, joint_positions: np.ndarray,
                     gripper_state: float, reset: bool = False) -> bytes:
    """Pack observation into msgpack bytes.

    Args:
        images: {cam_name: rgb_array, cam_name_depth: depth_array}
        joint_positions: (7,) float32
        gripper_state: scalar
        reset: if True, signals episode reset
    """
    encoded_images = {}
    for name, img in images.items():
        if name.endswith("_depth"):
            encoded_images[name] = encode_depth(img)
        else:
            encoded_images[name] = encode_image(img)

    msg = {
        "images": encoded_images,
        "joint_positions": _np_to_bytes(joint_positions.astype(np.float32)),
        "gripper_state": float(gripper_state),
        "timestamp": time.time(),
        "reset": reset,
    }
    return msgpack.packb(msg, use_bin_type=True)


def unpack_observation(data: bytes) -> dict:
    """Unpack msgpack bytes into observation dict with decoded numpy arrays."""
    msg = msgpack.unpackb(data, raw=False)

    images = {}
    for name, img_bytes in msg["images"].items():
        if name.endswith("_depth"):
            images[name] = decode_depth(img_bytes)
        else:
            images[name] = decode_image(img_bytes)

    return {
        "images": images,
        "joint_positions": _bytes_to_np(msg["joint_positions"]),
        "gripper_state": msg["gripper_state"],
        "timestamp": msg["timestamp"],
        "reset": msg.get("reset", False),
    }


def pack_action(action: np.ndarray, done: bool = False) -> bytes:
    """Pack single action into msgpack bytes."""
    msg = {
        "action": _np_to_bytes(action.astype(np.float32)),
        "done": done,
    }
    return msgpack.packb(msg, use_bin_type=True)


def unpack_action(data: bytes) -> tuple:
    """Unpack msgpack bytes into (action_array, done_bool)."""
    msg = msgpack.unpackb(data, raw=False)
    return _bytes_to_np(msg["action"]), msg["done"]


def pack_chunk(actions: np.ndarray, done: bool = False) -> bytes:
    """Pack action chunk (N, 8) into msgpack bytes."""
    msg = {
        "actions": _np_to_bytes(actions.astype(np.float32)),
        "done": done,
    }
    return msgpack.packb(msg, use_bin_type=True)


def unpack_chunk(data: bytes) -> tuple:
    """Unpack msgpack bytes into (actions_array (N, 8), done_bool)."""
    msg = msgpack.unpackb(data, raw=False)
    return _bytes_to_np(msg["actions"]), msg["done"]
