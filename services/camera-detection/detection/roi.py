"""
ROI intersection and geometric helpers for detection.
"""
import logging
import numpy as np
import cv2
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


def point_in_polygon(point: Tuple[float, float], polygon: List[List[float]]) -> bool:
    """
    Check if a point is inside a polygon using ray casting algorithm.
    
    Args:
        point: (x, y) coordinates
        polygon: List of [x, y] polygon vertices
    
    Returns:
        True if point is inside polygon
    """
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    print(f"✓ point_in_polygon completed: {inside}")
    return inside


def bbox_in_roi(bbox: Tuple[float, float, float, float], roi_points: List[List[float]]) -> bool:
    """
    Check if a bounding box overlaps with ROI polygon.
    
    Args:
        bbox: (x1, y1, x2, y2) bounding box coordinates
        roi_points: List of [x, y] polygon vertices
    
    Returns:
        True if bbox overlaps with ROI
    """
    x1, y1, x2, y2 = bbox
    
    # Check if any corner of the bbox is inside the ROI
    corners = [
        (x1, y1),  # top-left
        (x2, y1),  # top-right
        (x2, y2),  # bottom-right
        (x1, y2),  # bottom-left
    ]
    
    for corner in corners:
        if point_in_polygon(corner, roi_points):
            print(f"✓ bbox_in_roi completed: True (corner inside)")
            return True
    
    # Check if center is inside ROI
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    result = point_in_polygon((center_x, center_y), roi_points)
    
    print(f"✓ bbox_in_roi completed: {result}")
    return result


def bbox_roi_overlap_ratio(bbox: Tuple[float, float, float, float], roi_mask: np.ndarray) -> float:
    """
    Calculate the overlap ratio between a bounding box and ROI mask.
    
    Args:
        bbox: (x1, y1, x2, y2) bounding box coordinates
        roi_mask: Binary mask of ROI (255 inside ROI, 0 outside)
    
    Returns:
        Overlap ratio (0.0 to 1.0)
    """
    x1, y1, x2, y2 = map(int, bbox)
    
    # Ensure coordinates are within image bounds
    h, w = roi_mask.shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    
    if x2 <= x1 or y2 <= y1:
        print("✓ bbox_roi_overlap_ratio completed: 0.0 (invalid bbox)")
        return 0.0
    
    # Extract ROI region for this bbox
    bbox_region = roi_mask[y1:y2, x1:x2]
    
    # Calculate overlap ratio
    total_pixels = bbox_region.size
    roi_pixels = np.sum(bbox_region > 0)
    
    ratio = roi_pixels / total_pixels if total_pixels > 0 else 0.0
    
    print(f"✓ bbox_roi_overlap_ratio completed: {ratio:.3f}")
    return ratio


def filter_detections_by_roi(
    detections: List[dict],
    roi_points: Optional[List[List[float]]] = None,
    roi_mask: Optional[np.ndarray] = None,
    overlap_threshold: float = 0.5
) -> Tuple[List[dict], List[dict]]:
    """
    Filter detections based on ROI intersection.
    
    Args:
        detections: List of detection dictionaries with 'bbox' key
        roi_points: ROI polygon vertices (for point-in-polygon check)
        roi_mask: Binary ROI mask (for precise overlap calculation)
        overlap_threshold: Minimum overlap ratio to consider detection in ROI
    
    Returns:
        Tuple of (roi_detections, non_roi_detections)
    """
    if not roi_points and roi_mask is None:
        logger.warning("No ROI provided, returning all detections as non-ROI")
        print("✓ filter_detections_by_roi completed: no ROI")
        return [], detections
    
    roi_detections = []
    non_roi_detections = []
    
    for det in detections:
        bbox = det['bbox']
        
        if roi_mask is not None:
            # Use mask-based overlap ratio
            overlap = bbox_roi_overlap_ratio(bbox, roi_mask)
            in_roi = overlap >= overlap_threshold
        elif roi_points:
            # Use point-in-polygon check
            in_roi = bbox_in_roi(bbox, roi_points)
        else:
            in_roi = False
        
        det['in_roi'] = in_roi
        
        if in_roi:
            roi_detections.append(det)
        else:
            non_roi_detections.append(det)
    
    print(f"✓ filter_detections_by_roi completed: {len(roi_detections)} in ROI, {len(non_roi_detections)} outside")
    return roi_detections, non_roi_detections


print("✓ detection.roi module loaded")
