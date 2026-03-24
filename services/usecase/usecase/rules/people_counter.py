"""
People Counter usecase rule.

Rule: Count persons detected, excluding staff uniforms.
This is a dashboard/analytics rule — it never triggers alerts (triggered=False).

Staff filtering logic:
    The detection payload may contain both person detections (from YOLO pretrained)
    and uniform detections (from custom-trained YOLO). A person is considered staff
    if their bbox overlaps with any uniform bbox above an IoU threshold.
    Only non-staff persons are counted as customers.

Bbox format: {x1, y1, x2, y2} (slim payload standard).
"""

import logging
from typing import Dict, Any, List

from usecase.rules.base import BaseUsecaseRule

logger = logging.getLogger(__name__)

PERSON_CLASS_NAME = 'person'
STAFF_UNIFORM_CLASSES = {'grey_uniform', 'black_uniform', 'beige_uniform',
                         'blue_uniform', 'red_uniform'}
IOU_THRESHOLD = 0.3


def _bbox_iou(a: Dict, b: Dict) -> float:
    """IoU between two {x1, y1, x2, y2} bboxes."""
    ix1 = max(a['x1'], b['x1'])
    iy1 = max(a['y1'], b['y1'])
    ix2 = min(a['x2'], b['x2'])
    iy2 = min(a['y2'], b['y2'])

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection = (ix2 - ix1) * (iy2 - iy1)
    area_a = (a['x2'] - a['x1']) * (a['y2'] - a['y1'])
    area_b = (b['x2'] - b['x1']) * (b['y2'] - b['y1'])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def _is_staff(person_bbox: Dict, uniform_bboxes: List[Dict],
              iou_threshold: float = IOU_THRESHOLD) -> bool:
    """True if person bbox overlaps any uniform bbox above threshold."""
    return any(
        _bbox_iou(person_bbox, u['bbox']) >= iou_threshold
        for u in uniform_bboxes
    )


class PeopleCounterRule(BaseUsecaseRule):
    """
    Analytics rule: count customers (persons minus staff) in the frame.

    Never triggers alerts. The dashboard reads matched_objects and
    customer_count / staff_count from the result.
    """
    USECASE_ID = 'people_counter'

    def evaluate(self, detection_output: Dict[str, Any]) -> Dict[str, Any]:
        all_detections = self.get_detections(detection_output)

        persons = [d for d in all_detections
                   if d.get('class_name') == PERSON_CLASS_NAME]
        uniforms = [d for d in all_detections
                    if d.get('class_name') in STAFF_UNIFORM_CLASSES]

        # Partition persons into staff vs customers
        customers = []
        staff = []
        for p in persons:
            if uniforms and _is_staff(p['bbox'], uniforms):
                staff.append(p)
            else:
                customers.append(p)

        logger.info(
            f'[{self.USECASE_ID}] total_persons={len(persons)}, '
            f'customers={len(customers)}, staff={len(staff)}'
        )

        return {
            'triggered': False,  # analytics rule — never alerts
            'matched_objects': customers,
            'customer_count': len(customers),
            'staff_count': len(staff),
        }
