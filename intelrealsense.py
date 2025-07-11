# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file contains utilities for recording frames from Intel Realsense cameras.
"""

import argparse
import concurrent.futures
import logging
import math
import shutil
import threading
import time
import traceback
from collections import Counter
from pathlib import Path
from threading import Thread

import numpy as np
from PIL import Image
import abc
from dataclasses import dataclass

import draccus # type: ignore

SERIAL_NUMBER_INDEX = 1

@dataclass
class CameraConfig(draccus.ChoiceRegistry, abc.ABC):
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)


@CameraConfig.register_subclass("intelrealsense")
@dataclass
class IntelRealSenseCameraConfig(CameraConfig):
    """
    Example of tested options for Intel Real Sense D405:

    ```python
    IntelRealSenseCameraConfig(128422271347, 30, 640, 480)
    IntelRealSenseCameraConfig(128422271347, 60, 640, 480)
    IntelRealSenseCameraConfig(128422271347, 90, 640, 480)
    IntelRealSenseCameraConfig(128422271347, 30, 1280, 720)
    IntelRealSenseCameraConfig(128422271347, 30, 640, 480, use_depth=True)
    IntelRealSenseCameraConfig(128422271347, 30, 640, 480, rotation=90)
    ```
    """

    name: str | None = None
    serial_number: str | None = None
    fps: int | None = None
    width: int | None = None
    height: int | None = None
    color_mode: str = "rgb"
    channels: int | None = None
    use_depth: bool = False
    force_hardware_reset: bool = True
    rotation: int | None = None
    mock: bool = False

    def __post_init__(self):
        # bool is stronger than is None, since it works with empty strings
        if bool(self.name) and bool(self.serial_number):
            raise ValueError(
                f"One of them must be set: name or serial_number, but {self.name=} and {self.serial_number=} provided."
            )

        if self.color_mode not in ["rgb", "bgr"]:
            raise ValueError(
                f"`color_mode` is expected to be 'rgb' or 'bgr', but {self.color_mode} is provided."
            )

        self.channels = 3

        at_least_one_is_not_none = self.fps is not None or self.width is not None or self.height is not None
        at_least_one_is_none = self.fps is None or self.width is None or self.height is None
        if at_least_one_is_not_none and at_least_one_is_none:
            raise ValueError(
                "For `fps`, `width` and `height`, either all of them need to be set, or none of them, "
                f"but {self.fps=}, {self.width=}, {self.height=} were provided."
            )

        if self.rotation not in [-90, None, 90, 180]:
            raise ValueError(f"`rotation` must be in [-90, None, 90, 180] (got {self.rotation})")




def find_cameras(raise_when_empty=True, mock=False) -> list[dict]:
    """
    Find the names and the serial numbers of the Intel RealSense cameras
    connected to the computer.
    """
    if mock:
        import tests.cameras.mock_pyrealsense2 as rs # type: ignore
    else:
        import pyrealsense2 as rs # type: ignore

    cameras = []
    for device in rs.context().query_devices():
        serial_number = str(device.get_info(rs.camera_info(SERIAL_NUMBER_INDEX)))
        name = device.get_info(rs.camera_info.name)
        cameras.append(
            {
                "serial_number": serial_number,
                "name": name,
            }
        )

    if raise_when_empty and len(cameras) == 0:
        raise OSError(
            "Not a single camera was detected. Try re-plugging, or re-installing `librealsense` and its python wrapper `pyrealsense2`, or updating the firmware."
        )

    return cameras


def save_image(img_array, serial_number, frame_index, images_dir):
    try:
        img = Image.fromarray(img_array)
        path = images_dir / f"camera_{serial_number}_frame_{frame_index:06d}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(path), quality=100)
        logging.info(f"Saved image: {path}")
    except Exception as e:
        logging.error(f"Failed to save image for camera {serial_number} frame {frame_index}: {e}")


def save_images_from_cameras(
    images_dir: Path,
    serial_numbers: list[str] | None = None,
    fps=None,
    width=None,
    height=None,
    record_time_s=2,
    mock=False,
):
    """
    Initializes all the cameras and saves images to the directory. Useful to visually identify the camera
    associated to a given serial number.
    """
    if serial_numbers is None or len(serial_numbers) == 0:
        camera_infos = find_cameras(mock=mock)
        serial_numbers = [str(cam["serial_number"]) for cam in camera_infos]

    if mock:
        import tests.cameras.mock_cv2 as cv2 # type: ignore
    else:
        import cv2 # type: ignore

    print("Connecting cameras")
    cameras = []
    for cam_sn in serial_numbers:
        cam_sn = str(cam_sn)
        print(f"{cam_sn=}")
        config = IntelRealSenseCameraConfig(
            serial_number=cam_sn, fps=fps, width=width, height=height, mock=mock
        )
        camera = IntelRealSenseCamera(config)
        camera.connect()
        print(
            f"IntelRealSenseCamera({camera.serial_number}, fps={camera.fps}, width={camera.capture_width}, height={camera.capture_height}, color_mode={camera.color_mode})"
        )
        cameras.append(camera)

    images_dir = Path(images_dir)
    if images_dir.exists():
        shutil.rmtree(
            images_dir,
        )
    images_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving images to {images_dir}")
    frame_index = 0
    start_time = time.perf_counter()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            while True:
                now = time.perf_counter()

                for camera in cameras:
                    # If we use async_read when fps is None, the loop will go full speed, and we will end up
                    # saving the same images from the cameras multiple times until the RAM/disk is full.
                    image = camera.read() if fps is None else camera.async_read()
                    if image is None:
                        print("No Frame")

                    bgr_converted_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    executor.submit(
                        save_image,
                        bgr_converted_image,
                        camera.serial_number,
                        frame_index,
                        images_dir,
                    )

                if fps is not None:
                    dt_s = time.perf_counter() - now
                    time.sleep(1 / fps - dt_s)

                if time.perf_counter() - start_time > record_time_s:
                    break

                print(f"Frame: {frame_index:04d}\tLatency (ms): {(time.perf_counter() - now) * 1000:.2f}")

                frame_index += 1
    finally:
        print(f"Images have been saved to {images_dir}")
        for camera in cameras:
            camera.disconnect()


class IntelRealSenseCamera:
    
    def __init__(
        self,
        config: IntelRealSenseCameraConfig,
    ):
        self.config = config
        if config.name is not None:
            self.serial_number = str(self.find_serial_number_from_name(config.name))
        else:
            self.serial_number = str(config.serial_number) if config.serial_number is not None else None

        # Store the raw (capture) resolution from the config.
        self.capture_width = config.width
        self.capture_height = config.height

        # If rotated by ±90, swap width and height.
        if config.rotation in [-90, 90]:
            self.width = config.height
            self.height = config.width
        else:
            self.width = config.width
            self.height = config.height

        self.fps = config.fps
        self.channels = config.channels
        self.color_mode = config.color_mode
        self.use_depth = config.use_depth
        self.force_hardware_reset = config.force_hardware_reset
        self.mock = config.mock

        self.camera = None
        self.is_connected = False
        self.thread = None
        self.stop_event = None
        self.color_image = None
        self.depth_map = None
        self.logs = {}

        if self.mock:
            import tests.cameras.mock_cv2 as cv2 # type: ignore
        else:
            import cv2 # type: ignore

        self.rotation = None
        if config.rotation == -90:
            self.rotation = cv2.ROTATE_90_COUNTERCLOCKWISE
        elif config.rotation == 90:
            self.rotation = cv2.ROTATE_90_CLOCKWISE
        elif config.rotation == 180:
            self.rotation = cv2.ROTATE_180

    def find_serial_number_from_name(self, name):
        camera_infos = find_cameras()
        camera_names = [cam["name"] for cam in camera_infos]
        this_name_count = Counter(camera_names)[name]
        if this_name_count > 1:
            # TODO(aliberts): Test this with multiple identical cameras (Aloha)
            raise ValueError(
                f"Multiple {name} cameras have been detected. Please use their serial number to instantiate them."
            )

        name_to_serial_dict = {cam["name"]: str(cam["serial_number"]) for cam in camera_infos}
        cam_sn = name_to_serial_dict[name]

        return cam_sn

    def connect(self):

        if self.mock:
            import tests.cameras.mock_pyrealsense2 as rs # type: ignore
        else:
            import pyrealsense2 as rs # type: ignore

        config = rs.config()
        config.enable_device(str(self.serial_number))

        if self.fps and self.capture_width and self.capture_height:
            # TODO(rcadene): can we set rgb8 directly?
            config.enable_stream(
                rs.stream.color, self.capture_width, self.capture_height, rs.format.rgb8, self.fps
            )
        else:
            config.enable_stream(rs.stream.color)

        if self.use_depth:
            if self.fps and self.capture_width and self.capture_height:
                config.enable_stream(
                    rs.stream.depth, self.capture_width, self.capture_height, rs.format.z16, self.fps
                )
            else:
                config.enable_stream(rs.stream.depth)

        self.camera = rs.pipeline()
        try:
            profile = self.camera.start(config)
            is_camera_open = True
        except RuntimeError:
            is_camera_open = False
            traceback.print_exc()

        # If the camera doesn't work, display the camera indices corresponding to
        # valid cameras.
        if not is_camera_open:
            # Verify that the provided `serial_number` is valid before printing the traceback
            camera_infos = find_cameras()
            serial_numbers = [cam["serial_number"] for cam in camera_infos]
            if self.serial_number not in serial_numbers:
                raise ValueError(
                    f"`serial_number` is expected to be one of these available cameras {serial_numbers}, but {self.serial_number} is provided instead. "
                    "To find the serial number you should use, run `python lerobot/common/robot_devices/cameras/intelrealsense.py`."
                )

            raise OSError(f"Can't access IntelRealSenseCamera({self.serial_number}).")

        color_stream = profile.get_stream(rs.stream.color)
        color_profile = color_stream.as_video_stream_profile()
        actual_fps = color_profile.fps()
        actual_width = color_profile.width()
        actual_height = color_profile.height()

        # Using `math.isclose` since actual fps can be a float (e.g. 29.9 instead of 30)
        if self.fps is not None and not math.isclose(self.fps, actual_fps, rel_tol=1e-3):
            # Using `OSError` since it's a broad that encompasses issues related to device communication
            raise OSError(
                f"Can't set {self.fps=} for IntelRealSenseCamera({self.serial_number}). Actual value is {actual_fps}."
            )
        if self.capture_width is not None and self.capture_width != actual_width:
            raise OSError(
                f"Can't set {self.capture_width=} for IntelRealSenseCamera({self.serial_number}). Actual value is {actual_width}."
            )
        if self.capture_height is not None and self.capture_height != actual_height:
            raise OSError(
                f"Can't set {self.capture_height=} for IntelRealSenseCamera({self.serial_number}). Actual value is {actual_height}."
            )

        self.fps = round(actual_fps)
        self.capture_width = round(actual_width)
        self.capture_height = round(actual_height)

        self.is_connected = True

    def read(self, temporary_color: str | None = None) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Read a frame from the camera returned in the format height x width x channels (e.g. 480 x 640 x 3)
        of type `np.uint8`, contrarily to the pytorch format which is float channel first.

        When `use_depth=True`, returns a tuple `(color_image, depth_map)` with a depth map in the format
        height x width (e.g. 480 x 640) of type np.uint16.

        Note: Reading a frame is done every `camera.fps` times per second, and it is blocking.
        If you are reading data from other sensors, we advise to use `camera.async_read()` which is non blocking version of `camera.read()`.
        """

        if self.mock:
            import tests.cameras.mock_cv2 as cv2 # type: ignore
        else:
            import cv2 # type: ignore

        start_time = time.perf_counter()

        frame = self.camera.wait_for_frames(timeout_ms=5000)

        color_frame = frame.get_color_frame()

        if not color_frame:
            raise OSError(f"Can't capture color image from IntelRealSenseCamera({self.serial_number}).")

        color_image = np.asanyarray(color_frame.get_data())

        requested_color_mode = self.color_mode if temporary_color is None else temporary_color
        if requested_color_mode not in ["rgb", "bgr"]:
            raise ValueError(
                f"Expected color values are 'rgb' or 'bgr', but {requested_color_mode} is provided."
            )

        # IntelRealSense uses RGB format as default (red, green, blue).
        if requested_color_mode == "bgr":
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

        h, w, _ = color_image.shape
        if h != self.capture_height or w != self.capture_width:
            raise OSError(
                f"Can't capture color image with expected height and width ({self.height} x {self.width}). ({h} x {w}) returned instead."
            )

        if self.rotation is not None:
            color_image = cv2.rotate(color_image, self.rotation)

        # log the number of seconds it took to read the image
        self.logs["delta_timestamp_s"] = time.perf_counter() - start_time

        # log the utc time at which the image was received
        # self.logs["timestamp_utc"] = capture_timestamp_utc()

        if self.use_depth:
            depth_frame = frame.get_depth_frame()
            if not depth_frame:
                raise OSError(f"Can't capture depth image from IntelRealSenseCamera({self.serial_number}).")

            depth_map = np.asanyarray(depth_frame.get_data())

            h, w = depth_map.shape
            if h != self.capture_height or w != self.capture_width:
                raise OSError(
                    f"Can't capture depth map with expected height and width ({self.height} x {self.width}). ({h} x {w}) returned instead."
                )

            if self.rotation is not None:
                depth_map = cv2.rotate(depth_map, self.rotation)

            return color_image, depth_map
        else:
            return color_image

    def read_loop(self):
        while not self.stop_event.is_set():
            if self.use_depth:
                self.color_image, self.depth_map = self.read()
            else:
                self.color_image = self.read()

    def async_read(self):
        """Access the latest color image"""

        if self.thread is None:
            self.stop_event = threading.Event()
            self.thread = Thread(target=self.read_loop, args=())
            self.thread.daemon = True
            self.thread.start()

        num_tries = 0
        while self.color_image is None:
            # TODO(rcadene, aliberts): intelrealsense has diverged compared to opencv over here
            num_tries += 1
            time.sleep(1 / self.fps)
            if num_tries > self.fps and (self.thread.ident is None or not self.thread.is_alive()):
                raise Exception(
                    "The thread responsible for `self.async_read()` took too much time to start. There might be an issue. Verify that `self.thread.start()` has been called."
                )

        if self.use_depth:
            return self.color_image, self.depth_map
        else:
            return self.color_image

    def disconnect(self):


        if self.thread is not None and self.thread.is_alive():
            # wait for the thread to finish
            self.stop_event.set()
            self.thread.join()
            self.thread = None
            self.stop_event = None

        self.camera.stop()
        self.camera = None

        self.is_connected = False

    def __del__(self):
        if getattr(self, "is_connected", False):
            self.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Save a few frames using `IntelRealSenseCamera` for all cameras connected to the computer, or a selected subset."
    )
    parser.add_argument(
        "--serial-numbers",
        type=int,
        nargs="*",
        default=None,
        help="List of serial numbers used to instantiate the `IntelRealSenseCamera`. If not provided, find and use all available camera indices.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Set the number of frames recorded per seconds for all cameras. If not provided, use the default fps of each camera.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Set the width for all cameras. If not provided, use the default width of each camera.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Set the height for all cameras. If not provided, use the default height of each camera.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default="outputs/images_from_intelrealsense_cameras",
        help="Set directory to save a few frames for each camera.",
    )
    parser.add_argument(
        "--record-time-s",
        type=float,
        default=2.0,
        help="Set the number of seconds used to record the frames. By default, 2 seconds.",
    )
    args = parser.parse_args()
    save_images_from_cameras(**vars(args))
