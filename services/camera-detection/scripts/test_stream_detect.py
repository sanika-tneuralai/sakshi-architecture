"""Quick smoke test: pull live frames from the edge RTSP restream and run the
goec YOLO model against them, reporting detections per frame."""
import os
import time
import cv2
from ultralytics import YOLO

RTSP = os.environ.get("RTSP_URL", "rtsp://100.123.244.59:8554/cam")
MODEL = os.environ.get("MODEL_PATH", "./models/goec_N_v1.pt")
CONF = float(os.environ.get("CONF", "0.5"))
N_FRAMES = int(os.environ.get("N_FRAMES", "10"))
OUT_DIR = "./screenshots/_stream_test"

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
os.makedirs(OUT_DIR, exist_ok=True)

print(f"[*] Loading model: {MODEL}")
model = YOLO(MODEL)
print(f"[*] Classes: {model.names}")

print(f"[*] Opening stream: {RTSP}")
cap = cv2.VideoCapture(RTSP, cv2.CAP_FFMPEG)
if not cap.isOpened():
    raise SystemExit("[!] FAILED to open RTSP stream")

# warm up / let buffer fill
ok, frame = cap.read()
if not ok:
    raise SystemExit("[!] Stream opened but no frames decoded")
print(f"[*] First frame OK: {frame.shape[1]}x{frame.shape[0]}")

grabbed = 0
total_dets = 0
t0 = time.time()
for i in range(N_FRAMES):
    ok, frame = cap.read()
    if not ok:
        print(f"  frame {i}: read FAILED")
        continue
    grabbed += 1
    res = model.predict(frame, conf=CONF, verbose=False)[0]
    n = len(res.boxes)
    total_dets += n
    counts = {}
    for c in res.boxes.cls.tolist():
        name = model.names[int(c)]
        counts[name] = counts.get(name, 0) + 1
    confs = [round(float(x), 2) for x in res.boxes.conf.tolist()]
    print(f"  frame {i:2d}: {n} det {counts if counts else ''} conf={confs}")
    out = f"{OUT_DIR}/frame_{i:02d}.jpg"
    cv2.imwrite(out, res.plot())

cap.release()
dt = time.time() - t0
print(f"\n[=] {grabbed}/{N_FRAMES} frames decoded, {total_dets} total detections "
      f"in {dt:.1f}s ({grabbed/dt:.1f} fps inference). Annotated frames -> {OUT_DIR}")
