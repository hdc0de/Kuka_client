"""RealSense multi-camera manager with configurable RGB/RGBD modes."""

from dataclasses import dataclass

import numpy as np


@dataclass
class CameraConfig:
    name: str           # e.g. "base_camera"
    serial: str = ""    # RealSense serial number (empty = any available)
    width: int = 640
    height: int = 480
    fps: int = 30
    mode: str = "rgb"   # "rgb" | "rgbd"


class CameraManager:
    """Manages multiple RealSense cameras."""

    def __init__(self, configs: list[CameraConfig]):
        import pyrealsense2 as rs
        self.rs = rs
        self.configs = configs
        self.pipelines = []
        self._start_cameras()

    def _start_cameras(self):
        rs = self.rs
        for cfg in self.configs:
            pipeline = rs.pipeline()
            rs_config = rs.config()

            if cfg.serial:
                rs_config.enable_device(cfg.serial)

            rs_config.enable_stream(
                rs.stream.color, cfg.width, cfg.height, rs.format.rgb8, cfg.fps
            )
            if cfg.mode == "rgbd":
                rs_config.enable_stream(
                    rs.stream.depth, cfg.width, cfg.height, rs.format.z16, cfg.fps
                )

            pipeline.start(rs_config)
            self.pipelines.append(pipeline)
            print(f"[Camera] Started '{cfg.name}' (serial={cfg.serial or 'auto'}, "
                  f"{cfg.width}x{cfg.height}, {cfg.mode})")

    def capture(self) -> dict[str, np.ndarray]:
        """Capture from all cameras.

        Returns:
            {cam_name: (H,W,3) uint8 RGB, cam_name_depth: (H,W) uint16} for RGBD cameras.
        """
        result = {}
        for cfg, pipeline in zip(self.configs, self.pipelines):
            frames = pipeline.wait_for_frames()

            color_frame = frames.get_color_frame()
            result[cfg.name] = np.asarray(color_frame.get_data())

            if cfg.mode == "rgbd":
                depth_frame = frames.get_depth_frame()
                result[f"{cfg.name}_depth"] = np.asarray(depth_frame.get_data())

        return result

    def stop(self):
        for pipeline in self.pipelines:
            pipeline.stop()
        self.pipelines.clear()
        print("[Camera] All cameras stopped.")


class DummyCamera:
    """Fake camera for testing without hardware."""

    def __init__(self, configs: list[CameraConfig]):
        self.configs = configs
        print(f"[DummyCamera] Initialized {len(configs)} virtual cameras")

    def capture(self) -> dict[str, np.ndarray]:
        result = {}
        for cfg in self.configs:
            result[cfg.name] = np.random.randint(
                0, 255, (cfg.height, cfg.width, 3), dtype=np.uint8
            )
            if cfg.mode == "rgbd":
                result[f"{cfg.name}_depth"] = np.random.randint(
                    0, 5000, (cfg.height, cfg.width), dtype=np.uint16
                )
        return result

    def stop(self):
        print("[DummyCamera] Stopped.")
