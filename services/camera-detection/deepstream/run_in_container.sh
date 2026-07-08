#!/usr/bin/env bash
# =============================================================================
# Launch the DeepStream 6.4 container with the repo mounted, and drop into the
# deepstream/ working dir. From there run ./convert_model.sh (Phase 1a) and the
# deepstream-app validation.
#
# RUN ON THE T4 SERVER, after Docker + nvidia-container-toolkit are installed.
# The image (~20 GB) is pulled on first run — make sure you have the disk (50 GB
# volume recommended; see README).
#
# Env overrides:
#   DS_IMAGE  (default nvcr.io/nvidia/deepstream:6.4-gc-triton-devel)
# =============================================================================
set -euo pipefail

IMAGE="${DS_IMAGE:-nvcr.io/nvidia/deepstream:6.4-gc-triton-devel}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../.." && pwd)"   # deepstream -> camera-detection -> services -> repo root

echo "Image     : $IMAGE"
echo "Mounting  : $REPO_ROOT  ->  /workspace"
echo "Workdir   : /workspace/services/camera-detection/deepstream"
echo

# --gpus all       : expose the T4 (needs nvidia-container-toolkit)
# --network host   : simplest for pulling git/pip inside; drop if you lock networking
# -v ... :/workspace: repo is editable from host + container
docker run --gpus all -it --rm \
  --network host \
  -v "$REPO_ROOT":/workspace \
  -w /workspace/services/camera-detection/deepstream \
  "$IMAGE" bash
