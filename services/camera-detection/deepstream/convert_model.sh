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
# --- Auto-detect DeepStream install (override with DS_PATH=...) ---------------
# Order: explicit env -> the 'deepstream' symlink -> highest deepstream-* dir.
if [ -z "${DS_PATH:-}" ]; then
  if [ -d /opt/nvidia/deepstream/deepstream ]; then
    DS_PATH="$(readlink -f /opt/nvidia/deepstream/deepstream)"
  else
    DS_PATH="$(ls -d /opt/nvidia/deepstream/deepstream-* 2>/dev/null | sort -V | tail -1 || true)"
  fi
fi

if [ -z "${DS_PATH:-}" ] || [ ! -d "$DS_PATH" ]; then
  echo "ERROR: DeepStream SDK not found under /opt/nvidia/deepstream/."
  echo "  Found: $(ls -d /opt/nvidia/deepstream/* 2>/dev/null | tr '\n' ' ' || echo '(nothing)')"
  echo "  If DeepStream only exists inside a Docker container, run this script IN that"
  echo "  container. Otherwise install the SDK or set DS_PATH=/path/to/deepstream-X.Y."
  exit 1
fi

# --- Auto-detect CUDA_VER (override with CUDA_VER=...) ------------------------
# The parser Makefile needs the CUDA major.minor that matches this DeepStream build.
if [ -z "${CUDA_VER:-}" ]; then
  if [ -d /usr/local/cuda ]; then
    CUDA_VER="$(readlink -f /usr/local/cuda | sed -n 's/.*cuda-\([0-9]\+\.[0-9]\+\).*/\1/p')"
  fi
  # Fall back to the CUDA that ships with the detected DeepStream release.
  if [ -z "${CUDA_VER:-}" ]; then
    case "$(basename "$DS_PATH")" in
      *7.1*) CUDA_VER=12.6 ;;
      *7.0*|*6.4*) CUDA_VER=12.2 ;;
      *6.3*|*6.2*|*6.1*) CUDA_VER=11.8 ;;
      *) CUDA_VER=12.2 ;;
    esac
  fi
fi

REPO_DIR="$HERE/DeepStream-Yolo"
WORK_DIR="$REPO_DIR"   # export + parser + config all live here for validation

echo "=== Phase 1a: goec_N_v1 -> DeepStream ==="
echo "  model   : $MODEL_PT"
echo "  imgsz   : $IMGSZ"
echo "  DS_PATH : $DS_PATH  (detected)"
echo "  CUDA_VER: $CUDA_VER  (detected — override with CUDA_VER=... if wrong)"
echo

[ -f "$MODEL_PT" ] || { echo "ERROR: model not found: $MODEL_PT"; exit 1; }

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
# torch is pinned < 2.6 ON PURPOSE. Newer torch defaults to the dynamo ONNX
# exporter, which mis-traces the DeepStream-Yolo output wrapper and produces a
# graph TensorRT 8.6 rejects at parse time ("node_cat: all concat input tensors
# must have the same dimensions ... [-1,8400,4] vs [1,1,1]"). The legacy
# TorchScript exporter in torch<2.6 produces the correct, TRT-parseable graph.
# onnxscript is only needed by the dynamo path but harmless to keep installed.
if ! python3 -c "import torch,ultralytics,onnx,onnxslim; assert tuple(map(int,torch.__version__.split('.')[:2]))<(2,6)" 2>/dev/null; then
  echo "  installing export deps (torch<2.6 legacy ONNX exporter, ultralytics, onnx, onnxslim) ..."
  pip3 install --quiet "torch==2.5.1" "torchvision==0.20.1" "ultralytics>=8.4.60" onnx onnxslim onnxruntime
fi
cp -f "$MODEL_PT" "$WORK_DIR/"
PT_NAME="$(basename "$MODEL_PT")"
pushd "$WORK_DIR" >/dev/null
# --dynamic => one ONNX serves any batch size; nvinfer picks batch from the config.
# opset 18: recent torch.onnx can't down-convert below 18, and TensorRT 8.6
# (DeepStream 6.4) supports up to opset 19, so 18 avoids a noisy failed
# down-conversion while staying TRT-compatible.
python3 utils/export_yoloV8.py -w "$PT_NAME" -s "$IMGSZ" --opset 18 --simplify --dynamic
popd >/dev/null

# export_yoloV8.py writes "<stem>.onnx" (== goec_N_v1.onnx here). Normalise the
# name only when the export produced something different — copying a file onto
# itself errors under `set -e`.
ONNX_SRC="$(ls -t "$WORK_DIR"/*.onnx | head -1)"
if [ ! "$ONNX_SRC" -ef "$WORK_DIR/goec_N_v1.onnx" ]; then
  cp -f "$ONNX_SRC" "$WORK_DIR/goec_N_v1.onnx"
fi
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
