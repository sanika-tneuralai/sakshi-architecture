#!/usr/bin/env python3
"""
Phase 1b.1 prototype — validate the two mechanics the live pipeline depends on:
  (1) reading detection metadata (NvDsObjectMeta) off the nvinfer output, and
  (2) pulling the decoded frame as a numpy array via pyds.get_nvds_buf_surface
      (the exact path that will feed /detection/detect and the S3 upload).

Pipeline:  uridecodebin -> nvstreammux -> nvinfer -> nvvideoconvert(RGBA) -> fakesink
A probe on the fakesink pad prints per-frame detections (label/conf/bbox) and dumps
every Nth frame (that has detections) to a JPEG so you can eyeball frame extraction.

RUN INSIDE THE DeepStream CONTAINER (engine already built in Phase 1a):
  cd /workspace/services/camera-detection/deepstream
  python3 ds_probe_prototype.py \
      --uri file:///workspace/services/camera-detection/test_videos/gun_front.mp4 \
      --config DeepStream-Yolo/config_infer_goec.txt

  # RTSP works too:
  #   --uri rtsp://user:pass@host:554/stream   (add --live)

This is a throwaway validation harness — the real class lands in
camera/streams/deepstream_pipeline.py (Phase 1b.2). Needs no torch/ultralytics.
"""
import os
import sys
import argparse
from pathlib import Path

# pyds ships as a .so/wheel under the DeepStream lib dir and isn't always on the
# default import path. Mirror what the service's main.py does before importing.
_DS_LIB = os.getenv("DEEPSTREAM_PATH", "/opt/nvidia/deepstream/deepstream/lib")
if _DS_LIB not in sys.path:
    sys.path.insert(0, _DS_LIB)

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

try:
    import pyds
except ImportError:
    sys.exit(
        "ERROR: cannot import pyds (DeepStream Python bindings).\n"
        f"  Looked in: {_DS_LIB}\n"
        "  Locate it:   find /opt/nvidia -iname 'pyds*'\n"
        "  If a wheel:  pip3 install /opt/nvidia/deepstream/deepstream/lib/pyds-*.whl\n"
        "  Or set DEEPSTREAM_PATH to the dir containing pyds.so and re-run."
    )
import numpy as np

# cv2 is only needed to write the JPEG dumps; keep it optional so the core
# validation (metadata + get_nvds_buf_surface) still runs on a bare container.
try:
    import cv2
except ImportError:
    cv2 = None

# Must match labels.txt / the engine's class order.
LABELS = ["motorcycle", "car", "gun"]


def bus_call(bus, message, loop):
    t = message.type
    if t == Gst.MessageType.EOS:
        print("[bus] EOS")
        loop.quit()
    elif t == Gst.MessageType.WARNING:
        w, d = message.parse_warning()
        print(f"[bus] WARN: {w}: {d}")
    elif t == Gst.MessageType.ERROR:
        e, d = message.parse_error()
        print(f"[bus] ERROR: {e}: {d}")
        loop.quit()
    return True


def make_probe(dump_dir: Path, dump_every: int):
    dump_dir.mkdir(parents=True, exist_ok=True)

    def probe(pad, info, _u):
        buf = info.get_buffer()
        if not buf:
            return Gst.PadProbeReturn.OK

        batch = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
        if not batch:
            return Gst.PadProbeReturn.OK

        l_frame = batch.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            # --- (1) detection metadata ---
            dets = []
            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break
                r = obj.rect_params
                dets.append(
                    (
                        obj.class_id,
                        LABELS[obj.class_id] if obj.class_id < len(LABELS) else str(obj.class_id),
                        round(obj.confidence, 3),
                        int(r.left), int(r.top), int(r.width), int(r.height),
                    )
                )
                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            print(
                f"[src {frame_meta.source_id}] frame {frame_meta.frame_num}: "
                f"{len(dets)} det(s)"
            )
            for d in dets:
                print(f"    class_id={d[0]} {d[1]:<11} conf={d[2]:<5} "
                      f"bbox=({d[3]},{d[4]},{d[5]},{d[6]})")

            # --- (2) frame extraction to numpy (the dGPU-sensitive part) ---
            if dets and dump_every > 0 and frame_meta.frame_num % dump_every == 0:
                try:
                    surf = pyds.get_nvds_buf_surface(hash(buf), frame_meta.batch_id)
                    # surf is a view into mapped memory (RGBA) — copy before the
                    # buffer is recycled downstream.
                    rgba = np.array(surf, copy=True, order="C")
                    print(f"    -> get_nvds_buf_surface OK: shape={rgba.shape} dtype={rgba.dtype}")
                    if cv2 is not None:
                        bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
                        out = dump_dir / f"probe_src{frame_meta.source_id}_f{frame_meta.frame_num}.jpg"
                        cv2.imwrite(str(out), bgr)
                        print(f"    -> dumped frame to {out}")
                    else:
                        print("    (cv2 not installed — skipping JPEG dump; surface extraction still verified)")
                except Exception as e:
                    print(f"    !! get_nvds_buf_surface FAILED: {e}")

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        return Gst.PadProbeReturn.OK

    return probe


def cb_newpad(decodebin, decoder_src_pad, streammux_sinkpad):
    caps = decoder_src_pad.get_current_caps() or decoder_src_pad.query_caps()
    name = caps.get_structure(0).get_name()
    # Only link the video pad (ignore the audio pad — mp4 test clips have AAC).
    if name.startswith("video"):
        if decoder_src_pad.link(streammux_sinkpad) != Gst.PadLinkReturn.OK:
            print("!! failed to link decoder video pad -> streammux")
        else:
            print(f"[link] decoder video pad linked ({name})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uri", required=True, help="file:///... or rtsp://...")
    ap.add_argument("--config", required=True, help="nvinfer config file (relative to CWD)")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--live", action="store_true", help="set for RTSP/live sources")
    ap.add_argument("--dump-dir", default="probe_frames")
    ap.add_argument("--dump-every", type=int, default=30, help="dump every Nth detected frame (0=off)")
    args = ap.parse_args()

    Gst.init(None)
    pipeline = Gst.Pipeline.new("probe-proto")

    # dGPU (T4): CUDA-unified memory so get_nvds_buf_surface is CPU-mappable.
    mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)

    # --- source: uridecodebin (handles file + rtsp, dynamic video pad) ---
    src = Gst.ElementFactory.make("uridecodebin", "src")
    src.set_property("uri", args.uri)

    streammux = Gst.ElementFactory.make("nvstreammux", "mux")
    streammux.set_property("batch-size", 1)
    streammux.set_property("width", args.width)
    streammux.set_property("height", args.height)
    streammux.set_property("batched-push-timeout", 40000)
    streammux.set_property("live-source", 1 if args.live else 0)
    streammux.set_property("nvbuf-memory-type", mem_type)

    pgie = Gst.ElementFactory.make("nvinfer", "pgie")
    pgie.set_property("config-file-path", args.config)

    conv = Gst.ElementFactory.make("nvvideoconvert", "conv")
    conv.set_property("nvbuf-memory-type", mem_type)

    capsf = Gst.ElementFactory.make("capsfilter", "capsf")
    capsf.set_property("caps", Gst.Caps.from_string("video/x-raw(memory:NVMM), format=RGBA"))

    sink = Gst.ElementFactory.make("fakesink", "sink")
    sink.set_property("sync", 0)

    for el in (src, streammux, pgie, conv, capsf, sink):
        if not el:
            print("!! failed to create an element"); sys.exit(1)
        pipeline.add(el)

    # uridecodebin's video pad appears dynamically -> link to mux sink_0
    mux_sink = streammux.get_request_pad("sink_0")
    src.connect("pad-added", cb_newpad, mux_sink)

    streammux.link(pgie)
    pgie.link(conv)
    conv.link(capsf)
    capsf.link(sink)

    # Probe on the fakesink sink pad — metadata + RGBA surface both available here.
    sink.get_static_pad("sink").add_probe(
        Gst.PadProbeType.BUFFER, make_probe(Path(args.dump_dir), args.dump_every)
    )

    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    print(f"[run] uri={args.uri} config={args.config} live={args.live}")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except KeyboardInterrupt:
        pass
    finally:
        pipeline.set_state(Gst.State.NULL)
    print("[done]")


if __name__ == "__main__":
    main()
