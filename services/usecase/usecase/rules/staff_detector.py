"""
Staff Detector usecase rule.

Rule: Trigger if any staff uniform is detected.
Condition: class_name in STAFF_UNIFORM_CLASSES

The custom-trained YOLO model detects uniform colours (grey, black, beige,
blue, red). This rule simply surfaces those detections so the dashboard
or alert service can act on them.
"""

import logging
from typing import Dict, Any

from usecase.rules.base import BaseUsecaseRule

logger = logging.getLogger(__name__)

STAFF_UNIFORM_CLASSES = {'grey_uniform', 'black_uniform', 'beige_uniform',
                         'blue_uniform', 'red_uniform'}


class StaffDetectorRule(BaseUsecaseRule):
    """
    Usecase: Staff uniform detection.

    Triggers when at least one staff uniform is detected in the frame.
    """
    USECASE_ID = 'staff_detector'

    def evaluate(self, detection_output: Dict[str, Any]) -> Dict[str, Any]:
        all_detections = self.get_detections(detection_output)
        matched = [d for d in all_detections
                   if d.get('class_name') in STAFF_UNIFORM_CLASSES]

        triggered = len(matched) > 0

        logger.info(
            f'[{self.USECASE_ID}] triggered={triggered}, '
            f'staff_count={len(matched)}'
        )

        return {'triggered': triggered, 'matched_objects': matched}
