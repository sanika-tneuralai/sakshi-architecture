"""
DeepStream pipeline — decode + batched TensorRT inference for N streams in one
persistent GStreamer graph, caching the latest {frame, detections} per camera.

    nvurisrcbin/uridecodebin(src0..N) -> nvstreammux(batch=N) -> nvinfer(TensorRT)
        -> nvvideoconvert(RGBA) -> fakesink
                    │
        probe on the sink pad reads NvDsObjectMeta (detections) and, throttled to
        the publish fps, pulls the frame via pyds.get_nvds_buf_surface, storing
        both together in LatestState[camera_id].

The HTTP layer (/detection/detect) then just reads get_latest(camera_id) instead
of running inference inline — preserving the orchestration pull contract while the
heavy work happens continuously on the GPU.

Validated mechanics (Phase 1b.1): NvDsObjectMeta read + get_nvds_buf_surface on the
T4 with CUDA-unified nvbuf memory. This module builds the reusable class around them.

Runtime source add/remove is NOT yet supported (Phase 1b.4) — sources are added
before start().
"""
import os
import sys
import time
import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

# pyds / DeepStream libs aren't always on the default import path.
_DS_LIB = os.getenv("DEEPSTREAM_PATH", "/opt/nvidia/deepstream/deepstream/lib")
if _DS_LIB not in sys.path:
    sys.path.insert(0, _DS_LIB)

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib
import numpy as np
import pyds

logger = logging.getLogger(__name__)

# Must match the engine's class order (labels.txt): 0=motorcycle 1=car 2=gun.
DEFAULT_LABELS = ["motorcycle", "car", "gun"]


@dataclass
class LatestState:
    """Most recent frame + detections for one camera, produced by the probe."""
    frame: Optional[np.ndarray]  # BGR HxWx3 copy, or None in metadata-only mode
    detections: List[dict]       # [{class_id, class_name, confidence, x1,y1,x2,y2}, ...]
    frame_count: int
    ts: datetime                 # capture time (tz-aware, UTC)


class DeepStreamPipeline:
    """
    One persistent DeepStream pipeline handling all assigned streams.

    Thread model: a GLib MainLoop runs in a daemon thread; the nvinfer-src probe
    (streaming thread) writes LatestState under a lock. get_latest() reads it from
    the FastAPI thread. Detection coords are in streammux space (width x height),
    which is also the resolution of the extracted frame.
    """

    def __init__(
        self,
        config_path: str,
        width: int = 1920,
        height: int = 1080,
        labels: Optional[List[str]] = None,
        publish_fps: int = 5,
        max_batch: int = 8,
        extract_frames: bool = True,
    ):
        self.config_path = config_path
        self.width = width
        self.height = height
        self.labels = labels or DEFAULT_LABELS
        self.publish_interval = 1.0 / max(1, publish_fps)
        # When False, the probe stores detections only (frame=None) and never
        # calls get_nvds_buf_surface — metadata-only mode.
        self.extract_frames = extract_frames
        # nvinfer runs at a FIXED batch (engine built once); nvstreammux carries
        # the actual current source count. So reconciles never rebuild the engine.
        self.max_batch = max_batch

        self._sources: Dict[str, dict] = {}     # camera_id -> {index, uri, fps}
        self._by_index: Dict[int, str] = {}      # streammux source_id -> camera_id
        self._state: Dict[str, LatestState] = {}
        self._last_pub: Dict[str, float] = {}    # camera_id -> monotonic ts of last frame copy
        self._lock = threading.Lock()

        self._pipeline: Optional[Gst.Pipeline] = None
        self._loop: Optional[GLib.MainLoop] = None
        self._thread: Optional[threading.Thread] = None
        self.is_running = False

        Gst.init(None)
        # dGPU (T4): CUDA-unified memory so get_nvds_buf_surface is CPU-mappable.
        self._mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)

    # ------------------------------------------------------------------
    # Configuration (before start)
    # ------------------------------------------------------------------
    @staticmethod
    def _to_gst_uri(source: str) -> str:
        """Normalise a source to a GStreamer URI for uridecodebin.

        Accepts rtsp(s)://, http(s)://, file:// as-is; converts a plain local
        path to file://<abspath>. The API validates plain paths / rtsp URLs, so
        this bridges both forms to what uridecodebin needs.
        """
        if source.startswith(("rtsp://", "rtsps://", "http://", "https://", "file://")):
            return source
        return "file://" + os.path.abspath(source)

    def add_source(self, camera_id: str, uri: str, fps: int = 5) -> None:
        if self.is_running:
            raise RuntimeError("Runtime source add is not supported yet (Phase 1b.4)")
        if camera_id in self._sources:
            raise ValueError(f"camera {camera_id} already added")
        uri = self._to_gst_uri(uri)
        index = len(self._sources)
        self._sources[camera_id] = {"index": index, "uri": uri, "fps": fps}
        self._by_index[index] = camera_id
        logger.info("Added source %s at index %d: %s", camera_id, index, uri)

    # ------------------------------------------------------------------
    # Pipeline build
    # ------------------------------------------------------------------
    def _make(self, factory: str, name: str) -> Gst.Element:
        el = Gst.ElementFactory.make(factory, name)
        if not el:
            raise RuntimeError(f"Failed to create GStreamer element: {factory}")
        return el

    def _cb_newpad(self, decodebin, pad, mux_sinkpad):
        caps = pad.get_current_caps() or pad.query_caps()
        name = caps.get_structure(0).get_name()
        if name.startswith("video"):
            if pad.link(mux_sinkpad) != Gst.PadLinkReturn.OK:
                logger.error("Failed to link a decoder video pad to streammux")

    def _build(self) -> None:
        num = len(self._sources)
        if num == 0:
            raise RuntimeError("No sources added")

        live = any(s["uri"].startswith(("rtsp://", "rtsps://")) for s in self._sources.values())

        self._pipeline = Gst.Pipeline.new("goec-ds-pipeline")

        streammux = self._make("nvstreammux", "mux")
        streammux.set_property("batch-size", num)
        streammux.set_property("width", self.width)
        streammux.set_property("height", self.height)
        streammux.set_property("batched-push-timeout", 40000)
        streammux.set_property("live-source", 1 if live else 0)
        streammux.set_property("nvbuf-memory-type", self._mem_type)
        self._pipeline.add(streammux)

        for camera_id, cfg in self._sources.items():
            idx = cfg["index"]
            src = self._make("uridecodebin", f"src-{idx}")
            src.set_property("uri", cfg["uri"])
            self._pipeline.add(src)
            mux_sink = streammux.get_request_pad(f"sink_{idx}")
            src.connect("pad-added", self._cb_newpad, mux_sink)

        pgie = self._make("nvinfer", "pgie")
        pgie.set_property("config-file-path", self.config_path)
        # Fixed engine batch (>= num). Built once, reused across reconciles.
        pgie.set_property("batch-size", self.max_batch)
        self._pipeline.add(pgie)

        conv = self._make("nvvideoconvert", "conv")
        conv.set_property("nvbuf-memory-type", self._mem_type)
        self._pipeline.add(conv)

        capsf = self._make("capsfilter", "capsf")
        capsf.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))
        self._pipeline.add(capsf)

        sink = self._make("fakesink", "sink")
        sink.set_property("sync", 0)
        sink.set_property("async", 0)
        self._pipeline.add(sink)

        streammux.link(pgie)
        pgie.link(conv)
        conv.link(capsf)
        capsf.link(sink)

        sink.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, self._probe)

        bus = self._pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        logger.info("Built DeepStream pipeline for %d source(s), live=%s", num, live)

    # ------------------------------------------------------------------
    # Probe: detections + throttled frame extraction -> LatestState
    # ------------------------------------------------------------------
    def _probe(self, pad, info):
        buf = info.get_buffer()
        if not buf:
            return Gst.PadProbeReturn.OK
        batch = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        if not batch:
            return Gst.PadProbeReturn.OK

        now = time.monotonic()
        l_frame = batch.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            camera_id = self._by_index.get(frame_meta.source_id)
            if camera_id is not None:
                # Throttle the (relatively expensive) frame copy to publish_fps.
                # Detections are cheap, but we keep frame+dets paired for the same
                # frame, so we only build state on a throttle tick.
                last = self._last_pub.get(camera_id, 0.0)
                if now - last >= self.publish_interval:
                    dets = self._read_detections(frame_meta)
                    frame = None
                    if self.extract_frames:
                        frame = self._extract_frame(buf, frame_meta.batch_id)
                        if frame is None:
                            # extraction failed this tick — keep last good state
                            try:
                                l_frame = l_frame.next
                            except StopIteration:
                                break
                            continue
                    state = LatestState(
                        frame=frame,
                        detections=dets,
                        frame_count=frame_meta.frame_num,
                        ts=datetime.now(timezone.utc),
                    )
                    with self._lock:
                        self._state[camera_id] = state
                    self._last_pub[camera_id] = now

            try:
                l_frame = l_frame.next
            except StopIteration:
                break
        return Gst.PadProbeReturn.OK

    def _read_detections(self, frame_meta) -> List[dict]:
        dets: List[dict] = []
        l_obj = frame_meta.obj_meta_list
        while l_obj is not None:
            try:
                obj = pyds.NvDsObjectMeta.cast(l_obj.data)
            except StopIteration:
                break
            r = obj.rect_params
            cid = obj.class_id
            dets.append({
                "class_id": cid,
                "class_name": self.labels[cid] if cid < len(self.labels) else str(cid),
                "confidence": float(obj.confidence),
                "x1": float(r.left),
                "y1": float(r.top),
                "x2": float(r.left + r.width),
                "y2": float(r.top + r.height),
            })
            try:
                l_obj = l_obj.next
            except StopIteration:
                break
        return dets

    def _extract_frame(self, buf, batch_id: int) -> Optional[np.ndarray]:
        try:
            surf = pyds.get_nvds_buf_surface(hash(buf), batch_id)  # RGBA view
            # Copy out of mapped memory + drop alpha, RGBA->BGR (no cv2 dependency).
            bgr = np.ascontiguousarray(np.array(surf, copy=True)[:, :, [2, 1, 0]])
            return bgr
        except Exception as e:  # noqa: BLE001
            logger.error("get_nvds_buf_surface failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def _on_bus_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            logger.info("Pipeline EOS")
            self.is_running = False
        elif t == Gst.MessageType.ERROR:
            err, dbg = message.parse_error()
            logger.error("Pipeline ERROR: %s | %s", err, dbg)
        elif t == Gst.MessageType.WARNING:
            warn, dbg = message.parse_warning()
            logger.warning("Pipeline WARNING: %s | %s", warn, dbg)

    def _run_loop(self):
        self._loop = GLib.MainLoop()
        try:
            self._loop.run()
        except Exception as e:  # noqa: BLE001
            logger.error("GLib loop error: %s", e)

    def start(self) -> None:
        if self.is_running:
            logger.warning("Pipeline already running")
            return
        self._build()
        if self._pipeline.set_state(Gst.State.PLAYING) == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Failed to set pipeline to PLAYING")
        self.is_running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="DS-GLib")
        self._thread.start()
        logger.info("DeepStream pipeline started")

    def stop(self) -> None:
        logger.info("Stopping DeepStream pipeline")
        self.is_running = False
        if self._pipeline:
            self._pipeline.set_state(Gst.State.NULL)
        if self._loop and self._loop.is_running():
            self._loop.quit()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        with self._lock:
            self._state.clear()
        self._pipeline = self._loop = self._thread = None

    # ------------------------------------------------------------------
    # Read API (used by the detect endpoint in 1b.3)
    # ------------------------------------------------------------------
    def get_latest(self, camera_id: str) -> Optional[LatestState]:
        with self._lock:
            return self._state.get(camera_id)

    def get_status(self, camera_id: str) -> Optional[dict]:
        if camera_id not in self._sources:
            return None
        st = self.get_latest(camera_id)
        return {
            "camera_id": camera_id,
            "is_running": self.is_running,
            "frame_count": st.frame_count if st else 0,
            "last_frame_time": st.ts.timestamp() if st else 0,
            "stream_index": self._sources[camera_id]["index"],
            "rtsp_url": self._sources[camera_id]["uri"],
            "fps": self._sources[camera_id]["fps"],
            "backend": "deepstream",
        }

    def list_cameras(self) -> List[str]:
        return list(self._sources.keys())


class DeepStreamCameraView:
    """
    Per-camera adapter over the shared DeepStreamPipeline, returned by
    CameraManager.get_camera_stream() so the detection API can treat a DeepStream
    camera like the OpenCV one. `is_deepstream` lets the API skip inline inference
    and read the pipeline's cached detections instead.
    """
    is_deepstream = True

    def __init__(self, pipeline: "DeepStreamPipeline", camera_id: str):
        self._p = pipeline
        self.camera_id = camera_id

    def get_latest(self) -> Optional[LatestState]:
        return self._p.get_latest(self.camera_id)

    def get_frame(self) -> Optional[np.ndarray]:
        st = self._p.get_latest(self.camera_id)
        return st.frame if st else None

    async def get_preprocessed_frame(self) -> Optional[dict]:
        st = self._p.get_latest(self.camera_id)
        if st is None:
            return None
        return {
            "frame": st.frame,
            "timestamp": st.ts.timestamp(),
            "shape": st.frame.shape,
            "frame_count": st.frame_count,
        }

    def get_status(self) -> dict:
        return self._p.get_status(self.camera_id) or {
            "camera_id": self.camera_id,
            "is_running": False,
            "backend": "deepstream",
        }


# ---------------------------------------------------------------------------
# Standalone validation (Phase 1b.2): run the class on one/more URIs and poll
# get_latest() from the main thread — proves the cache is populated & readable
# cross-thread, exactly as the service will use it.
#
#   python3 camera/streams/deepstream_pipeline.py \
#       --config deepstream/DeepStream-Yolo/config_infer_goec.txt \
#       --uri file:///workspace/services/camera-detection/test_videos/gun_front.mp4
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--uri", action="append", required=True, help="repeatable; one per camera")
    ap.add_argument("--seconds", type=int, default=20)
    ap.add_argument("--max-batch", type=int, default=8,
                    help="fixed nvinfer batch; first run builds the b<N> engine (~minutes)")
    args = ap.parse_args()

    pipe = DeepStreamPipeline(config_path=args.config, max_batch=args.max_batch)
    for i, uri in enumerate(args.uri):
        pipe.add_source(f"cam{i}", uri, fps=5)
    pipe.start()

    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < args.seconds:
            time.sleep(2)
            for cam in pipe.list_cameras():
                st = pipe.get_latest(cam)
                if st is None:
                    print(f"{cam}: <no frame yet>")
                else:
                    labels = ", ".join(f"{d['class_name']}:{d['confidence']:.2f}" for d in st.detections) or "none"
                    print(f"{cam}: frame#{st.frame_count} {st.frame.shape} dets=[{labels}] ts={st.ts.isoformat()}")
    finally:
        pipe.stop()
