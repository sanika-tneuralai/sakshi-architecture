# Phase 1a — goec_N_v1 → DeepStream / TensorRT (model conversion + validation gate)

This is the **first, gating** step of the DeepStream migration (`decode-detect-deepstream`
branch). Goal: convert the `.pt` detector into an `nvinfer` TensorRT model and **prove the
boxes + class labels are correct** before any pipeline code is written. Everything here runs
**on the T4 GPU server** — it cannot run on a box without the DeepStream SDK.

## Model facts (verified from the checkpoint)

| Property | Value |
|---|---|
| Architecture | **YOLO11n** (`yolo11n.yaml`, scale `n`) |
| Trained with | ultralytics **8.4.60** |
| Input size | 640×640 |
| Classes (index order) | **0=motorcycle, 1=car, 2=gun** |

⚠️ The class order is index-significant. A wrong `labels.txt` order silently mislabels every
detection (same trap noted for the Hailo backend). `labels.txt` here is the verified order —
`convert_model.sh` copies it over whatever the export auto-generates.

## Files in this directory

| File | Purpose |
|---|---|
| `convert_model.sh` | Clone DeepStream-Yolo, export ONNX, build the custom bbox parser. |
| `config_infer_goec.txt` | `nvinfer` config (FP16, 3 classes, low pre-cluster threshold). |
| `labels.txt` | Verified class order. |
| `deepstream_validate.txt` | `deepstream-app` config for the single-stream visual gate. |

Generated artifacts (`DeepStream-Yolo/`, `*.onnx`, `*.engine`, `*.so`) are gitignored.

---

## Step 0a — provision the environment (Docker, first time only)

DeepStream is **not** installed on the host; it runs in the NVIDIA container (also the
deployment target). One-time setup on the T4 server:

- **Disk:** the devel image is ~20 GB and building the parser needs `nvcc` (devel-only), so
  a `-base`/`-samples` image won't work. **Resize the EBS root volume to ~50 GB**:
  ```bash
  # AWS side:
  aws ec2 modify-volume --volume-id vol-XXXX --size 50
  # instance side:
  lsblk && sudo growpart /dev/nvme0n1 1 && sudo resize2fs /dev/nvme0n1p1 && df -h /
  ```
- **Docker + NVIDIA toolkit:**
  ```bash
  sudo apt-get update && sudo apt-get install -y docker.io
  sudo systemctl enable --now docker && sudo usermod -aG docker $USER   # re-login after
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker
  ```

## Step 0b — enter the DeepStream container

Everything below (Steps 0–4) runs **inside** the container. Launch it with the repo mounted:

```bash
cd services/camera-detection/deepstream
chmod +x run_in_container.sh
./run_in_container.sh          # pulls nvcr.io/nvidia/deepstream:6.4-gc-triton-devel (~20 GB) on first run
# ... now inside the container, at /workspace/services/camera-detection/deepstream
```

Sanity-check the GPU is visible in the container: `nvidia-smi` should list the T4.

## Step 0 — verify the DeepStream environment (inside the container)

```bash
# DeepStream version (expect 6.4)
cat /opt/nvidia/deepstream/deepstream/version 2>/dev/null || \
  deepstream-app --version-all

# GPU + driver (T4, driver 535+ for DS 6.4)
nvidia-smi

# TensorRT (expect 8.6.x for DS 6.4) and CUDA (expect 12.2)
dpkg -l | grep -i tensorrt | head
nvcc --version 2>/dev/null || cat /usr/local/cuda/version.json 2>/dev/null
```

If DeepStream is **not** 6.4, set `DS_PATH` and `CUDA_VER` accordingly before Step 2
(DS 6.4→CUDA 12.2, DS 7.0→12.2, DS 7.1→12.6). CUDA_VER **must** match or the parser build fails.

## Step 1 — get the model onto the server

```bash
# from your machine (models are gitignored, so copy it explicitly)
scp services/camera-detection/models/goec_N_v1.pt <user>@<t4-server>:/path/to/repo/services/camera-detection/models/
```

## Step 2 — convert + build the parser

```bash
cd services/camera-detection/deepstream
chmod +x convert_model.sh
./convert_model.sh                 # uses ../models/goec_N_v1.pt, imgsz 640
# or: DS_PATH=/opt/nvidia/deepstream/deepstream-7.1 CUDA_VER=12.6 ./convert_model.sh
```

Success looks like:
- `DeepStream-Yolo/goec_N_v1.onnx` exists
- `DeepStream-Yolo/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so` built
- `labels.txt` + `config_infer_goec.txt` staged in `DeepStream-Yolo/`

## Step 3 — build the TensorRT engine (automatic, on first run)

No separate step: `nvinfer` builds `goec_N_v1.onnx_b1_gpu0_fp16.engine` from the ONNX the
first time deepstream-app runs (Step 4). It takes a few minutes; subsequent runs reuse it.

## Step 4 — validation gate (visual check)

```bash
cd services/camera-detection/deepstream/DeepStream-Yolo

# point the validator at a real absolute path to a test clip
ABS=$(cd ../../.. && pwd)   # repo root
sed -i "s#file:///REPLACE/WITH/ABS/PATH#file://$ABS#" ../deepstream_validate.txt

deepstream-app -c ../deepstream_validate.txt
```

Then inspect the annotated output:

```bash
ls -lh /tmp/goec_validate_out.mp4
# copy it back and eyeball it, or dump a frame:
ffmpeg -y -i /tmp/goec_validate_out.mp4 -vf "select=eq(n\,50)" -vframes 1 /tmp/frame50.jpg
```

### ✅ Pass criteria (all must hold)
1. Engine builds without error; deepstream-app reaches `PLAYING` and runs to EOS.
2. Boxes are drawn on the expected objects (use `gun_front.mp4` to exercise the `gun` class,
   `car2.mp4` for `car`).
3. **Labels are correct** — a gun is labelled `gun`, not `motorcycle`/`car`. This is the whole
   point of the gate.
4. Detection counts/positions look sane vs. running the `.pt` in ultralytics on the same clip
   (optional cross-check: `yolo predict model=../../models/goec_N_v1.pt source=<clip>`).

**Do not start Phase 1b until this passes.** A bad export/parser here would surface later as
"the pipeline runs but detections are garbage," which is far harder to diagnose.

---

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Parser `make` fails | `CUDA_VER` doesn't match the install. Set it to the DS release's CUDA. |
| Engine build fails on `Unsupported ONNX opset` | Lower `--opset` in `convert_model.sh` (try 16→13) to match TRT 8.6. |
| deepstream-app: cannot load custom lib | `custom-lib-path` is relative to CWD — run from the `DeepStream-Yolo/` dir. |
| Boxes appear but labels are wrong | `labels.txt` order. Must be motorcycle/car/gun. |
| No boxes at all | Threshold too high (check `pre-cluster-threshold`), or model-color-format/net-scale mismatch. Confirm the `.pt` actually detects on the same clip first. |
| `export_yoloV8.py` errors loading the checkpoint | ultralytics too old; `pip install "ultralytics>=8.4.60"`. |

## Next (Phase 1b, not this step)

Once the gate passes, the same `config_infer_goec.txt` + parser `.so` + ONNX feed the live
service pipeline: `nvurisrcbin → nvstreammux(batch=N) → nvinfer → probe → per-camera cache`,
with `/detection/detect` reading the cache. The engine will be rebuilt at `batch = number of
streams` (the `_bN_` in the engine filename). See the migration plan.
