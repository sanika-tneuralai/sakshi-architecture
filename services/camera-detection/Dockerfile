# =============================================================================
# Camera-Detection service — DeepStream/TensorRT backend (decode-detect-deepstream)
# Multi-stage: torch/ultralytics live ONLY in the throwaway builder (for the ONNX
# export). The runtime image is slim — no torch stack (nvinfer does inference).
# Base: NVIDIA DeepStream 6.4 (GStreamer + TensorRT 8.6 + CUDA 12.2 + nvcc).
# Model weights models/goec_N_v1.pt must be in the build context.
# The b8 TensorRT engine is built at first container run (docker build has no GPU)
# into a mounted volume; see docker-entrypoint.sh + docker-compose.yml.
# =============================================================================

# ---- Stage 1: builder — export ONNX + compile the bbox parser ---------------
FROM nvcr.io/nvidia/deepstream:6.4-gc-triton-devel AS builder
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /build
# CPU torch<2.6 (legacy, TRT-parseable ONNX exporter) — small, no CUDA wheels.
RUN pip3 install --no-cache-dir \
      "torch==2.5.1+cpu" "torchvision==0.20.1+cpu" --extra-index-url https://download.pytorch.org/whl/cpu \
 && pip3 install --no-cache-dir "ultralytics>=8.4.60" onnx onnxslim
COPY deepstream/ /build/deepstream/
COPY models/goec_N_v1.pt /build/models/goec_N_v1.pt
# clone DeepStream-Yolo + export goec_N_v1.onnx + build libnvdsinfer_custom_impl_Yolo.so
RUN cd /build/deepstream && chmod +x convert_model.sh && ./convert_model.sh

# ---- Stage 2: runtime — slim, no torch/ultralytics --------------------------
FROM nvcr.io/nvidia/deepstream:6.4-gc-triton-devel
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    DEEPSTREAM_PATH=/opt/nvidia/deepstream/deepstream/lib \
    INFERENCE_BACKEND=deepstream

WORKDIR /app

COPY requirements.deepstream.txt .
RUN pip3 install --no-cache-dir -r requirements.deepstream.txt \
 && pip3 install --no-cache-dir \
      https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.1.10/pyds-1.1.10-py3-none-linux_x86_64.whl

# App code (.dockerignore keeps out the host's DeepStream-Yolo/, venvs, videos).
COPY . /app/

# Built artifacts (ONNX + parser .so + labels) from the builder — stashed so the
# entrypoint can seed the (initially empty) engine volume mounted over
# /app/deepstream/DeepStream-Yolo. No torch/ultralytics come across.
COPY --from=builder /build/deepstream/DeepStream-Yolo /opt/goec-ds

RUN mkdir -p /app/logs /app/screenshots
EXPOSE 8004

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD curl -f http://localhost:8004/health || exit 1

COPY deploy/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
