"""
Logo-Coverage Occupancy Detector
================================
YOLO-independent occupancy signal for parking slots.

Each parking slot has a painted "G logo" (white circle + plug icon) inside
it. When a car parks in the slot, the car body covers the logo. We use the
*absence* of the logo as an occupancy signal — independent of, and
complementary to, the YOLO ``car`` detector. This catches cases where YOLO
misses the car (bad parking angle, partial occlusion, low-confidence frame).

The check runs **independently for every ROI** in the request: a frame
showing two slots produces two booleans, computed in parallel from the same
frame against the same reference set, one per slot.

Why pixel coverage instead of SSIM
----------------------------------
SSIM scored too tightly on this deployment. The painted logo strokes are
white in every reference, but the surrounding asphalt changes drastically
between day and night (manhole reflectivity, shadows, water stains). SSIM
asked to score "same logo, totally different surroundings" lands in the
0.15–0.20 range for two empty references — barely separable from the 0.05–0.08
of a fully occluded slot.

The white logo strokes themselves, however, *are* invariant: bright in every
empty reference, dark when a car body covers them. So the better signal is
"what fraction of the known logo-pixel locations are still bright in the
current frame?" That ignores the noisy asphalt entirely and gives a
meaningful, well-separated number.

How it works
------------
1. At first use, for each ``slot_refs/{camera_id}/*.{jpg,png}`` reference
   image, warp the requested logo polygon to a fixed-size grayscale square
   and threshold to a "bright pixels" mask via Otsu. The per-camera, per-ROI
   **logo mask** is the **intersection** of all reference bright masks —
   pixels that are bright in every empty reference (i.e. genuinely the
   painted logo, not lighting artefacts).
2. At runtime, warp the live frame's ROI the same way, threshold the same
   way, and compute
       coverage = |current_bright ∩ logo_mask| / |logo_mask|
   The slot is occluded if coverage < LOGO_COVERAGE_MIN.

Cost: ~3-8 ms per ROI on commodity hardware. Cheap enough to run on every
detection request.

Tunables (env)
--------------
LOGO_COVERAGE_MIN       Minimum fraction of logo pixels that must remain
                        bright for the slot to be considered clear. Below
                        this, the slot is reported as occluded. Default 0.5.
LOGO_WARP_SIZE          Edge length (px) of the warped square crop. 256 is
                        a good balance of detail vs cost.
"""
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

COVERAGE_MIN = float(os.getenv("LOGO_COVERAGE_MIN", "0.5"))
WARP_SIZE = int(os.getenv("LOGO_WARP_SIZE", "256"))

SLOT_REFS_DIR = os.getenv(
    "SLOT_REFS_DIR",
    str(Path(__file__).resolve().parent.parent / "slot_refs"),
)


def _order_polygon(points: List[List[int]]) -> np.ndarray:
    """
    Order a 4-point polygon as [top-left, top-right, bottom-right, bottom-left]
    so cv2.getPerspectiveTransform always sees a consistent winding. Required
    because the polygons in the DB may be saved in any clockwise/counter-
    clockwise order from the ROI editor.
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.shape != (4, 2):
        raise ValueError(f"Expected a 4-point polygon, got shape {pts.shape}")

    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]   # TL
    ordered[2] = pts[np.argmax(s)]   # BR
    ordered[1] = pts[np.argmin(d)]   # TR
    ordered[3] = pts[np.argmax(d)]   # BL
    return ordered


def _warp_polygon(image: np.ndarray, polygon: List[List[int]]) -> Optional[np.ndarray]:
    """
    Warp the given polygon region of ``image`` to a fixed WARP_SIZE x WARP_SIZE
    grayscale square. Returns None if the warp fails (degenerate polygon).
    """
    try:
        src = _order_polygon(polygon)
    except ValueError as e:
        logger.warning("[LOGO] Polygon ordering failed: %s", e)
        return None

    dst = np.array(
        [[0, 0], [WARP_SIZE - 1, 0], [WARP_SIZE - 1, WARP_SIZE - 1], [0, WARP_SIZE - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(image, M, (WARP_SIZE, WARP_SIZE))
    if warped.size == 0:
        return None
    return cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)


def _bright_mask(gray: np.ndarray) -> np.ndarray:
    """
    Binary mask of "bright" pixels using Otsu thresholding. Otsu picks the
    brightness cutoff automatically per image, so the same code works under
    very different lighting (a bright cutoff at noon ≠ at night).

    Result is uint8 with 255 for bright pixels and 0 elsewhere.
    """
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def _coverage(current_bright: np.ndarray, logo_mask: np.ndarray) -> float:
    """
    Fraction of the logo's bright pixels that are still bright in the current
    frame. 1.0 = logo fully visible; 0.0 = logo fully covered.
    """
    logo_pixels = int(np.count_nonzero(logo_mask))
    if logo_pixels == 0:
        return 1.0  # No logo signal — fail-open (treat as visible / not occluded)
    overlap = int(np.count_nonzero(cv2.bitwise_and(current_bright, logo_mask)))
    return overlap / logo_pixels


class LogoOcclusionDetector:
    """
    Per-camera reference cache + occlusion check.

    References live at ``{SLOT_REFS_DIR}/{camera_id}/*.{jpg,jpeg,png}``. Files
    starting with ``_`` or ``test_`` are skipped so test fixtures cannot
    contaminate the empty-reference set.

    The detector is keyed by ``(camera_id, roi_id, polygon)`` so re-warping
    only happens when the polygon set changes. In practice the reference
    masks are computed once at boot and reused forever.
    """

    def __init__(self) -> None:
        self._frames: Dict[str, List[np.ndarray]] = {}
        self._mask_cache: Dict[tuple, Optional[np.ndarray]] = {}

    def _load_frames(self, camera_id: str) -> List[np.ndarray]:
        if camera_id in self._frames:
            return self._frames[camera_id]

        cam_dir = Path(SLOT_REFS_DIR) / camera_id
        if not cam_dir.is_dir():
            logger.warning("[LOGO] No reference dir for camera=%s at %s", camera_id, cam_dir)
            self._frames[camera_id] = []
            return []

        frames: List[np.ndarray] = []
        for path in sorted(cam_dir.iterdir()):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            stem = path.stem.lower()
            if stem.startswith(("_", "test_")) or stem in {"test", "occluded", "test_occluded"}:
                logger.info("[LOGO] Skipping non-reference file: %s", path.name)
                continue
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                logger.warning("[LOGO] Failed to read reference %s", path)
                continue
            frames.append(img)
            logger.info("[LOGO] Loaded reference camera=%s file=%s shape=%s",
                        camera_id, path.name, img.shape)

        self._frames[camera_id] = frames
        return frames

    def _logo_mask(
        self, camera_id: str, roi_id: str, polygon: List[List[int]]
    ) -> Optional[np.ndarray]:
        """
        Build (and cache) the logo mask for this (camera, roi, polygon):
        intersection of bright-pixel masks across every reference. Pixels
        that are bright in *every* empty reference are the painted logo;
        anything that varies between references is lighting noise.
        """
        key = (camera_id, roi_id, tuple(tuple(p) for p in polygon))
        if key in self._mask_cache:
            return self._mask_cache[key]

        ref_masks: List[np.ndarray] = []
        for frame in self._load_frames(camera_id):
            warp = _warp_polygon(frame, polygon)
            if warp is None:
                continue
            ref_masks.append(_bright_mask(warp))

        if not ref_masks:
            self._mask_cache[key] = None
            return None

        # Intersection of all bright masks: pixels bright in every reference.
        logo = ref_masks[0].copy()
        for m in ref_masks[1:]:
            logo = cv2.bitwise_and(logo, m)

        bright_count = int(np.count_nonzero(logo))
        logger.info(
            "[LOGO] Built logo mask camera=%s roi=%s pixels=%d (%.1f%% of warp)",
            camera_id, roi_id, bright_count,
            100.0 * bright_count / (WARP_SIZE * WARP_SIZE),
        )
        self._mask_cache[key] = logo
        return logo

    def evaluate(
        self,
        camera_id: str,
        frame: np.ndarray,
        logo_rois: Dict[str, List[List[int]]],
    ) -> Dict[str, bool]:
        """
        Return ``{roi_id: occluded}`` for every ROI in ``logo_rois``.

        Each ROI is evaluated independently — a frame with two slots produces
        two booleans, both computed from the same frame against the same
        reference set.

        ``occluded=True`` means: < LOGO_COVERAGE_MIN of the logo's bright
        pixels are still bright in the current frame, i.e. a car is covering
        the logo. ``False`` means it looks like an empty slot.

        If references are missing or warp fails, the ROI is reported as
        ``False`` (fail-open) so the YOLO path stays in charge.
        """
        result: Dict[str, bool] = {}
        if frame is None or not logo_rois:
            return result

        for roi_id, polygon in logo_rois.items():
            logo_mask = self._logo_mask(camera_id, roi_id, polygon)
            if logo_mask is None:
                result[roi_id] = False
                continue

            current_warp = _warp_polygon(frame, polygon)
            if current_warp is None:
                result[roi_id] = False
                continue

            current_bright = _bright_mask(current_warp)
            cov = _coverage(current_bright, logo_mask)
            occluded = cov < COVERAGE_MIN
            result[roi_id] = occluded
            logger.info(
                "[LOGO] camera=%s roi=%s coverage=%.3f thresh=%.2f -> %s",
                camera_id, roi_id, cov, COVERAGE_MIN,
                "OCCLUDED" if occluded else "clear",
            )

        return result


_detector: Optional[LogoOcclusionDetector] = None


def get_logo_occlusion_detector() -> LogoOcclusionDetector:
    global _detector
    if _detector is None:
        _detector = LogoOcclusionDetector()
        logger.info(
            "[LOGO] Detector initialised (refs_dir=%s, coverage_min=%.2f, warp_size=%d)",
            SLOT_REFS_DIR, COVERAGE_MIN, WARP_SIZE,
        )
    return _detector


print("✓ detection.logo_occlusion module loaded")
