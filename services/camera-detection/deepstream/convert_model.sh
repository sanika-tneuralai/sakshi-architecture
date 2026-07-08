#!/usr/bin/env bash
# =============================================================================
# Phase 1a — convert goec_N_v1.pt (YOLO11n, 3-class) to a DeepStream/TensorRT
# nvinfer model, and build the custom YOLO bbox parser.
#
# RUN THIS ON THE T4 SERVER (DeepStream 6.4 + CUDA 12.2 + TensorRT 8.6).
# It cannot run on a machine without the DeepStream SDK + GPU.
#
# What it does:
#   1. Clones marcoslucianops/DeepStream-Yolo (battle-tested export + parser).
#   2. Exports the .pt -> ONNX with the output layout NvDsInferParseYolo expects.
#   3. Builds libnvdsinfer_custom_impl_Yolo.so against this DeepStream install.
#   4. Stages ONNX + labels + config next to the parser for validation.
#
# The TensorRT .engine is NOT built here — nvinfer builds it automatically on the
# first deepstream-app run (see README step 4), because the engine is specific to
# this exact GPU + TRT version and must be generated on the target box.
#
# Usage:
#   ./convert_model.sh [/path/to/goec_N_v1.pt] [imgsz]
# Env overrides:
#   DS_PATH   (default /opt/nvidia/deepstream/deepstream-6.4)
#   CUDA_VER  (default 12.2  — must match the DeepStream release; 6.4 => 12.2)
# =============================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_PT="${1:-$HERE/../models/goec_N_v1.pt}"
IMGSZ="${2:-640}"
DS_PATH="${DS_PATH:-/opt/nvidia/deepstream/deepstream-6.4}"
CUDA_VER="${CUDA_VER:-12.2}"

REPO_DIR="$HERE/DeepStream-Yolo"
WORK_DIR="$REPO_DIR"   # export + parser + config all live here for validation

echo "=== Phase 1a: goec_N_v1 -> DeepStream ==="
echo "  model   : $MODEL_PT"
echo "  imgsz   : $IMGSZ"
echo "  DS_PATH : $DS_PATH"
echo "  CUDA_VER: $CUDA_VER"
echo

[ -f "$MODEL_PT" ] || { echo "ERROR: model not found: $MODEL_PT"; exit 1; }
[ -d "$DS_PATH" ]  || { echo "ERROR: DeepStream not found at $DS_PATH (run on the T4 server)"; exit 1; }

# --- 1. Clone DeepStream-Yolo -------------------------------------------------
if [ ! -d "$REPO_DIR" ]; then
  echo "[1/4] Cloning DeepStream-Yolo ..."
  git clone https://github.com/marcoslucianops/DeepStream-Yolo.git "$REPO_DIR"
else
  echo "[1/4] DeepStream-Yolo already present, skipping clone."
fi

# --- 2. Export .pt -> ONNX ----------------------------------------------------
# Needs ultralytics + onnx in the current python env. Use an isolated venv so we
# don't disturb the service env. ultralytics must be >= the version that trained
# the checkpoint (8.4.60) to load it cleanly.
echo "[2/4] Exporting ONNX ..."
if ! python3 -c "import ultralytics, onnx, onnxslim" 2>/dev/null; then
  echo "  installing export deps (ultralytics, onnx, onnxslim, onnxruntime) ..."
  pip3 install --quiet "ultralytics>=8.4.60" onnx onnxslim onnxruntime
fi
cp -f "$MODEL_PT" "$WORK_DIR/"
PT_NAME="$(basename "$MODEL_PT")"
pushd "$WORK_DIR" >/dev/null
# --dynamic => one ONNX serves any batch size; nvinfer picks batch from the config.
python3 utils/export_yoloV8.py -w "$PT_NAME" -s "$IMGSZ" --opset 16 --simplify --dynamic
popd >/dev/null

# export_yoloV8.py writes "<stem>.onnx" and a labels.txt. Normalise the name.
ONNX_SRC="$(ls -t "$WORK_DIR"/*.onnx | head -1)"
cp -f "$ONNX_SRC" "$WORK_DIR/goec_N_v1.onnx"
echo "  ONNX -> $WORK_DIR/goec_N_v1.onnx"

# Use OUR verified labels (do not trust the auto-generated order blindly).
cp -f "$HERE/labels.txt" "$WORK_DIR/labels.txt"
cp -f "$HERE/config_infer_goec.txt" "$WORK_DIR/config_infer_goec.txt"

# --- 3. Build the custom bbox parser -----------------------------------------
echo "[3/4] Building libnvdsinfer_custom_impl_Yolo.so (CUDA_VER=$CUDA_VER) ..."
pushd "$REPO_DIR" >/dev/null
CUDA_VER="$CUDA_VER" make -C nvdsinfer_custom_impl_Yolo
popd >/dev/null
ls -lh "$REPO_DIR/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so"

# --- 4. Done ------------------------------------------------------------------
echo
echo "[4/4] Staged for validation in: $WORK_DIR"
echo "  - goec_N_v1.onnx"
echo "  - labels.txt (motorcycle/car/gun)"
echo "  - config_infer_goec.txt"
echo "  - nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so"
echo
echo "NEXT: build the engine + visually verify boxes (README step 4):"
echo "  cd $WORK_DIR"
echo "  deepstream-app -c $HERE/deepstream_validate.txt"
echo "(first run builds goec_N_v1.onnx_b1_gpu0_fp16.engine — takes a few minutes)"
