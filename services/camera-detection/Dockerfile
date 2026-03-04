# Use NVIDIA DeepStream base image
FROM nvcr.io/nvidia/deepstream:6.4-gc-triton-devel

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV DEEPSTREAM_PATH=/opt/nvidia/deepstream/deepstream-6.4/lib

# Install Python 3.10+ and system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-gi \
    python3-dev \
    python3-gst-1.0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    gstreamer1.0-rtsp \
    libgstrtspserver-1.0-0 \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libgirepository1.0-dev \
    gcc \
    g++ \
    make \
    cmake \
    pkg-config \
    libcairo2-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA support
RUN pip3 install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/screenshots /app/logs

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set DeepStream library path
ENV LD_LIBRARY_PATH=${DEEPSTREAM_PATH}:${LD_LIBRARY_PATH}
ENV PYTHONPATH=${DEEPSTREAM_PATH}:${PYTHONPATH}

# Run the application
CMD ["python3", "main.py"]
