#!/usr/bin/env python3
"""
Phase 1b.3 validation — exercise the CameraManager DeepStream reconcile path
without the full FastAPI/DB/S3 stack.

Simulates what orchestration does: two /camera/start calls (a burst that should
coalesce into ONE pipeline build via the debounce), then reads the per-camera
view's get_latest() — the exact object /detection/detect consumes.

RUN INSIDE THE DeepStream CONTAINER, from the service root:
  cd /workspace/services/camera-detection
  pip install opencv-python-headless pydantic        # only deps this test needs
  python3 deepstream/test_camera_manager.py

Uses the bundled test clips. Because they're short files, they hit EOS after
~20s (that's fine — get_latest() keeps serving the last cached frame).
"""
import os
import sys
import time
import asyncio

# Must be set BEFORE importing camera.service (CameraManager reads the backend
# at construction). NVINFER_CONFIG_PATH defaults to the staged config already.
os.environ["INFERENCE_BACKEND"] = "deepstream"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from camera.service import camera_manager, DEEPSTREAM_AVAILABLE  # noqa: E402
from camera.schemas import RTSPConfig  # noqa: E402

VIDEOS = "/workspace/services/camera-detection/test_videos"
CAMS = [
    ("cam0", f"file://{VIDEOS}/gun_front.mp4"),
    ("cam1", f"file://{VIDEOS}/car2.mp4"),
]


async def main():
    assert DEEPSTREAM_AVAILABLE, "DeepStream pipeline module failed to import (pyds/gi?)"
    print(f"backend = {camera_manager.backend}  deepstream_mode = {camera_manager._deepstream_mode}")

    # Burst of starts — should debounce into a single pipeline build.
    for cam_id, uri in CAMS:
        await camera_manager.start_single_camera(RTSPConfig(camera_id=cam_id, rtsp_url=uri, fps=5))
        print(f"queued {cam_id}")

    print("waiting for debounce + pipeline build (b2 engine is cached, ~seconds)...")
    for _ in range(15):
        time.sleep(2)
        line = []
        for cam_id, _uri in CAMS:
            view = camera_manager.get_camera_stream(cam_id)
            st = view.get_latest() if view else None
            if st is None:
                line.append(f"{cam_id}=<none>")
            else:
                labels = ",".join(f"{d['class_name']}:{d['confidence']:.2f}" for d in st.detections) or "none"
                line.append(f"{cam_id}=frame#{st.frame_count}[{labels}]")
        print("  ", "  ".join(line))

    print("stopping all...")
    await camera_manager.stop_all()
    print("done")


if __name__ == "__main__":
    asyncio.run(main())
