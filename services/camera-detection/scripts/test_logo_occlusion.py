"""
Standalone sanity check for the logo-occlusion detector.

Usage:
    cd services/camera-detection
    python scripts/test_logo_occlusion.py

What it does
------------
1. Loads every reference image under slot_refs/camera_01/ (skipping any
   file whose stem starts with ``_`` or ``test_``).
2. Builds the per-ROI logo mask: pixels bright in *every* reference.
3. Saves the logo masks under /tmp/logo_warps/ for visual inspection.
4. For every reference frame, prints the coverage score per ROI — these
   should all be ~1.0 since the logo is fully visible in each empty
   reference.
5. If you drop a frame at slot_refs/camera_01/test_occluded.jpg with at
   least one slot covered by a car, the script also runs the negative
   case — coverage should be << LOGO_COVERAGE_MIN (default 0.5) and the
   detector should report ``occluded=True`` for that slot.

The point: pick a coverage threshold that separates "empty under any
lighting" from "car covers the logo" with margin.
"""
import importlib.util
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_LOGO_PATH = _HERE.parent / "detection" / "logo_occlusion.py"
_spec = importlib.util.spec_from_file_location("logo_occlusion", _LOGO_PATH)
logo_occlusion = importlib.util.module_from_spec(_spec)
sys.modules["logo_occlusion"] = logo_occlusion
_spec.loader.exec_module(logo_occlusion)

import cv2  # noqa: E402

LogoOcclusionDetector = logo_occlusion.LogoOcclusionDetector
COVERAGE_MIN = logo_occlusion.COVERAGE_MIN
SLOT_REFS_DIR = logo_occlusion.SLOT_REFS_DIR
WARP_SIZE = logo_occlusion.WARP_SIZE
_warp_polygon = logo_occlusion._warp_polygon
_bright_mask = logo_occlusion._bright_mask
_coverage = logo_occlusion._coverage

CAMERA_ID = "camera_01"

# G-logo polygons (from the user). Original frame resolution 1920x1080.
LOGO_ROIS = {
    "ROI_1": [[233, 216], [822, 202], [773, 545], [122, 569]],
    "ROI_2": [[1175, 184], [1657, 216], [1795, 515], [1316, 564]],
}


def main() -> int:
    ref_dir = Path(SLOT_REFS_DIR) / CAMERA_ID
    if not ref_dir.is_dir():
        print(f"[FAIL] Reference dir missing: {ref_dir}")
        return 1

    refs = sorted(
        p for p in ref_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and not p.stem.lower().startswith(("_", "test_"))
        and p.stem.lower() not in {"test", "occluded", "test_occluded"}
    )
    if not refs:
        print(f"[FAIL] No reference images in {ref_dir}")
        return 1

    print(f"[INFO] Threshold (LOGO_COVERAGE_MIN) = {COVERAGE_MIN}")
    print(f"[INFO] Warp size = {WARP_SIZE}x{WARP_SIZE}")
    print(f"[INFO] References: {[p.name for p in refs]}\n")

    out_dir = Path(tempfile.gettempdir()) / "logo_warps"
    out_dir.mkdir(exist_ok=True)

    det = LogoOcclusionDetector()

    # Build + dump logo masks for visual inspection.
    print("Logo masks (intersection of bright pixels across all refs):")
    for roi_id, poly in LOGO_ROIS.items():
        mask = det._logo_mask(CAMERA_ID, roi_id, poly)
        if mask is None:
            print(f"  {roi_id}: FAILED")
            continue
        out = out_dir / f"logo_mask_{roi_id}.png"
        cv2.imwrite(str(out), mask)
        print(f"  {roi_id}: -> {out}")
    print()

    # Sanity 1: every empty reference should score coverage ~1.0 against
    # the logo mask (the logo is, by construction, the pixels bright in
    # every reference).
    print(f"{'reference':<25} {'roi':<8} {'coverage':>9}  expected: > {COVERAGE_MIN}")
    print("-" * 60)
    for ref_path in refs:
        img = cv2.imread(str(ref_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[WARN] Failed to read {ref_path}")
            continue
        for roi_id, poly in LOGO_ROIS.items():
            mask = det._logo_mask(CAMERA_ID, roi_id, poly)
            if mask is None:
                continue
            warp = _warp_polygon(img, poly)
            if warp is None:
                continue
            cov = _coverage(_bright_mask(warp), mask)
            verdict = "OK" if cov >= COVERAGE_MIN else "BELOW"
            print(f"{ref_path.name:<25} {roi_id:<8} {cov:>9.3f}  [{verdict}]")
    print()

    # Sanity 2: occluded sample. The user can drop a frame at
    # slot_refs/{camera_id}/test_occluded.jpg to validate the negative case.
    occluded_path = next(
        (p for p in (ref_dir / "test_occluded.jpg", ref_dir / "_test_occluded.jpg")
         if p.is_file()),
        ref_dir / "test_occluded.jpg",
    )
    if occluded_path.is_file():
        print(f"[INFO] Found {occluded_path.name} — running occlusion test")
        frame = cv2.imread(str(occluded_path), cv2.IMREAD_COLOR)
        result = det.evaluate(CAMERA_ID, frame, LOGO_ROIS)
        print(f"[INFO] Result: {result}  (expected: at least one True)")

        # Also print the raw coverage scores for tuning context.
        print()
        print(f"{'occluded frame':<25} {'roi':<8} {'coverage':>9}  expected: < {COVERAGE_MIN}")
        print("-" * 60)
        for roi_id, poly in LOGO_ROIS.items():
            mask = det._logo_mask(CAMERA_ID, roi_id, poly)
            warp = _warp_polygon(frame, poly)
            if mask is None or warp is None:
                continue
            cov = _coverage(_bright_mask(warp), mask)
            verdict = "OK" if cov < COVERAGE_MIN else "ABOVE"
            print(f"{occluded_path.name:<25} {roi_id:<8} {cov:>9.3f}  [{verdict}]")
    else:
        print(
            f"[HINT] Save a frame with at least one occupied slot to "
            f"{occluded_path} and rerun to test the negative case."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
