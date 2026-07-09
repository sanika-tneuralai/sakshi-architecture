#!/usr/bin/env bash
# Seed the engine/artifact volume, then launch the service.
#
# The image bakes the built DeepStream-Yolo artifacts (parser .so, ONNX, labels)
# in /opt/goec-ds. In compose we mount a named volume over
# /app/deepstream/DeepStream-Yolo so the TensorRT engine — built by nvinfer on
# first run — persists across container recreation. On a fresh (empty) volume we
# copy the baked artifacts in so nvinfer has the ONNX + parser to build from.
set -euo pipefail

DEST=/app/deepstream/DeepStream-Yolo
PARSER="$DEST/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so"

if [ ! -f "$PARSER" ]; then
  echo "[entrypoint] seeding $DEST from baked artifacts (/opt/goec-ds)"
  mkdir -p "$DEST"
  cp -r /opt/goec-ds/. "$DEST/"
fi

if ! ls "$DEST"/*_b*_gpu0_fp16.engine >/dev/null 2>&1; then
  echo "[entrypoint] NOTE: no TensorRT engine cached yet — nvinfer will build the"
  echo "[entrypoint] b8 engine on the first /camera/start (~5 min, one time), then"
  echo "[entrypoint] cache it in the mounted volume for fast subsequent starts."
fi

exec python3 main.py
