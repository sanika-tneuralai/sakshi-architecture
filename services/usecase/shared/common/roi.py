"""
ROI (Region of Interest) utility functions.
Pure spatial math — no hardcoded camera geometry, no app imports, no side effects.

ROI polygons are always passed in by the caller (injected by the orchestrator).
"""
import logging
from typing import Dict, List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def bbox_centroid(bbox: dict) -> Tuple[float, float]:
    """Return (cx, cy) centroid of a bbox dict with x1/y1/x2/y2 keys."""
    cx = (bbox["x1"] + bbox["x2"]) / 2.0
    cy = (bbox["y1"] + bbox["y2"]) / 2.0
    return cx, cy


def is_point_in_roi(x: float, y: float, roi: List[Tuple[int, int]]) -> bool:
    """
    True if point (x, y) is inside or on the boundary of polygon *roi*.
    Uses cv2.pointPolygonTest (measureDist=False):
        +1 = inside, 0 = on boundary, -1 = outside.
    """
    if len(roi) < 3:
        return False
    contour = np.array(roi, dtype=np.float32)
    result = cv2.pointPolygonTest(contour, (float(x), float(y)), measureDist=False)
    return result >= 0


def is_bbox_in_roi(bbox: dict, roi: List[Tuple[int, int]]) -> bool:
    """True if the bbox centroid is inside *roi*."""
    cx, cy = bbox_centroid(bbox)
    return is_point_in_roi(cx, cy, roi)


def which_rois(bbox: dict, rois: Dict[str, List[Tuple[int, int]]]) -> List[str]:
    """
    Return the names of all ROIs whose polygon contains the bbox centroid.

    Args:
        bbox: dict with x1/y1/x2/y2
        rois: mapping of roi_name -> list of (x, y) polygon points

    Returns:
        List of roi names the centroid falls inside (may be empty).
    """
    cx, cy = bbox_centroid(bbox)
    return [
        name
        for name, polygon in rois.items()
        if is_point_in_roi(cx, cy, polygon)
    ]


def which_rois_bbox_overlap(
    bbox: dict,
    rois: Dict[str, List[Tuple[int, int]]],
    overlap_threshold: float = 0.40,
) -> List[str]:
    """
    Return the names of all ROIs whose polygon has significant area overlap
    with the bounding box.  This detects double-slot parking where the car
    straddles a boundary and the centroid falls in only one ROI.

    Strategy: sample a grid of points inside the bbox and count how many
    fall inside each ROI polygon. A ROI is considered matched when the
    fraction of sample points inside it exceeds *overlap_threshold*.

    Args:
        bbox: dict with x1/y1/x2/y2
        rois: mapping of roi_name -> polygon point list
        overlap_threshold: fraction of bbox area that must overlap an ROI
                           to count as a match (default 15 %).

    Returns:
        List of ROI names with significant overlap (may be empty).
    """
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    # Sample a 5×5 grid of points covering the bbox
    steps = 5
    xs = [x1 + (x2 - x1) * i / (steps - 1) for i in range(steps)]
    ys = [y1 + (y2 - y1) * i / (steps - 1) for i in range(steps)]
    sample_points = [(x, y) for x in xs for y in ys]
    total = len(sample_points)

    matched = []
    for name, polygon in rois.items():
        hits = sum(1 for x, y in sample_points if is_point_in_roi(x, y, polygon))
        if hits / total >= overlap_threshold:
            matched.append(name)
    return matched
