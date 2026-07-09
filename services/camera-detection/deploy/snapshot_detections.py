#!/usr/bin/env python3
"""
Save an annotated snapshot (boxes + labels) for a camera, to eyeball live
detections from the running service. Writes into /app/logs (bind-mounted to the
host's services/camera-detection/logs/) so you can scp it out to view.

Run against the running service container:
  docker exec -it goec-camera-detection python3 /app/deploy/snapshot_detections.py cam0

Env: HOST (default http://localhost:8004), CONF (default 0.25), OUTDIR (/app/logs)
"""
import os
import sys
import json
import base64
import urllib.request

import numpy as np
import cv2

HOST = os.getenv("HOST", "http://localhost:8004")
CONF = float(os.getenv("CONF", "0.25"))
OUTDIR = os.getenv("OUTDIR", "/app/logs")


def _post(path, body):
    req = urllib.request.Request(
        HOST + path, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    return json.load(urllib.request.urlopen(req, timeout=15))


def _get(path):
    return json.load(urllib.request.urlopen(HOST + path, timeout=15))


def main():
    cam = sys.argv[1] if len(sys.argv) > 1 else "cam0"
    det = _post("/detection/detect", {"camera_id": cam, "confidence_threshold": CONF})
    frm = _get(f"/camera/frame/{cam}")

    img = cv2.imdecode(np.frombuffer(base64.b64decode(frm["frame"]), np.uint8), cv2.IMREAD_COLOR)
    for d in det["detections"]:
        b = d["bbox"]
        p1, p2 = (int(b["x1"]), int(b["y1"])), (int(b["x2"]), int(b["y2"]))
        cv2.rectangle(img, p1, p2, (0, 0, 255), 3)
        cv2.putText(img, f'{d["class_name"]} {d["confidence"]:.2f}',
                    (p1[0], max(0, p1[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, f"{cam}_annotated.jpg")
    cv2.imwrite(out, img)
    print(f"{cam}: {len(det['detections'])} detection(s) -> {out} "
          f"(frame {det.get('frame_count')})")


if __name__ == "__main__":
    main()
