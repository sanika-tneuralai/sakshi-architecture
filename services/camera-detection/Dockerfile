# =============================================================================
# Camera-Detection service — DeepStream/TensorRT backend (decode-detect-deepstream)
# =============================================================================
# Base: NVIDIA DeepStream 6.4 (GStreamer + TensorRT 8.6 + CUDA 12.2 + nvcc).
#
# What is baked at BUILD time (no GPU needed):
#   - Python deps (fastapi, opencv-headless, numpy 1.26, boto3, ...)
#   - the REAL NVIDIA pyds 1.1.10 wheel (DeepStream Python bindings)
#   - CPU torch + ultralytics ONLY to export the ONNX at build
#   - DeepStream-Yolo: clone + export goec_N_v1.onnx + compile the bbox parser .so
#
# What happens at first RUN (needs the GPU, so cannot be a build step):
#   - nvinfer builds the fixed-batch (b8) TensorRT engine from the ONNX, cached
#     into the mounted volume so subsequent starts just deserialize it (~seconds).
#
# The model weights (models/goec_N_v1.pt) must be present in the build context.
# =============================================================================
FROM nvcr.io/nvidia/deepstream:6.4-gc-triton-devel

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    DEEPSTREAM_PATH=/opt/nvidia/deepstream/deepstream/lib \
    INFERENCE_BACKEND=deepstream

WORKDIR /app

# --- Python runtime deps + real pyds bindings --------------------------------
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt \
 && pip3 install --no-cache-dir \
      https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.1.10/pyds-1.1.10-py3-none-linux_x86_64.whl

# --- Build-only export toolchain (CPU torch<2.6 => legacy TRT-parseable ONNX) -
RUN pip3 install --no-cache-dir \
      "torch==2.5.1+cpu" "torchvision==0.20.1+cpu" --extra-index-url https://download.pytorch.org/whl/cpu \
 && pip3 install --no-cache-dir "ultralytics>=8.4.60" onnx onnxslim

# --- App code (includes models/goec_N_v1.pt and the deepstream/ kit) ---------
COPY . /app/

# --- Build DeepStream-Yolo (clone + ONNX export + parser .so); NO engine yet --
# Stash the built artifacts in /opt so the entrypoint can seed the (initially
# empty) engine volume that gets mounted over deepstream/DeepStream-Yolo.
RUN cd /app/deepstream && chmod +x convert_model.sh && ./convert_model.sh \
 && mkdir -p /opt/goec-ds && cp -r /app/deepstream/DeepStream-Yolo/. /opt/goec-ds/

RUN mkdir -p /app/logs /app/screenshots
EXPOSE 8004

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD curl -f http://localhost:8004/health || exit 1

COPY deploy/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
