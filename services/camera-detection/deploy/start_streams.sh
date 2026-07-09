#!/usr/bin/env bash
# Start N camera pipelines and report per-camera detections — for validating the
# batched multi-stream path. By default it duplicates ONE RTSP URL N times
# (cam0..camN-1), which is enough to load-test decode+batched-inference when you
# only have a single source. Swap in a real per-camera list for production.
#
# Usage:
#   ./start_streams.sh [N] [RTSP_URL] [HOST]
# Env:
#   FPS   (default 5)   WAIT (default 30s; bump to ~330 on the FIRST run so the
#                       one-time b8 TensorRT engine build finishes before detect)
#
# Examples:
#   ./start_streams.sh 5 rtsp://100.123.244.59:8554/cam
#   WAIT=340 ./start_streams.sh 5           # first ever run (engine not cached yet)
set -euo pipefail

N="${1:-5}"
URL="${2:-rtsp://100.123.244.59:8554/cam}"
HOST="${3:-localhost:8004}"
FPS="${FPS:-5}"
WAIT="${WAIT:-30}"

echo "Starting $N camera(s) -> $URL  (host=$HOST fps=$FPS)"
for i in $(seq 0 $((N-1))); do
  printf 'cam%s: ' "$i"
  curl -s -X POST "http://$HOST/camera/start" -H 'Content-Type: application/json' \
    -d "{\"camera_id\":\"cam$i\",\"rtsp_url\":\"$URL\",\"fps\":$FPS}"
  echo
done

echo
echo "Waiting ${WAIT}s for debounced reconcile + warmup"
echo "(FIRST run also builds the b8 engine ~5 min — rerun with WAIT=340 or watch"
echo " 'docker compose logs -f' for 'serialize cuda engine ... b8 ... successfully')"
sleep "$WAIT"

echo
echo "=== /camera/list ==="
curl -s "http://$HOST/camera/list" | python3 -m json.tool

for i in $(seq 0 $((N-1))); do
  echo
  echo "=== cam$i /detection/detect ==="
  curl -s -X POST "http://$HOST/detection/detect" -H 'Content-Type: application/json' \
    -d "{\"camera_id\":\"cam$i\",\"confidence_threshold\":0.25}" \
    | python3 -c 'import sys,json; d=json.load(sys.stdin); print("  dets:", [(x["class_name"], round(x["confidence"],2)) for x in d.get("detections",[])], "| frame", d.get("frame_count"), "| ts", d.get("timestamp"))' 2>/dev/null \
    || echo "  (no detection / camera not ready)"
done
