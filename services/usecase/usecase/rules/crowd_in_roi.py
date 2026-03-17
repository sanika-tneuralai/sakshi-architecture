"""
Crowd in ROI usecase rule.

Rule: Trigger if 3 or more persons are detected inside ROI.
Condition: class_name == "person" AND in_roi == true AND count >= 3
"""

import logging 
from typing import Dict, Any
from usecase.rules.base import BaseUsecaseRule

logger = logging.getLogger(__name__)

class CrowdInROIRule(BaseUsecaseRule):
    USECASE_ID = 'crowd_in_roi'
    CROWD_THRESHOLD = 3

    def evaluate(self, detection_output: Dict[str, Any]) -> Dict[str, Any]:
        matched = self.get_in_roi_by_class(detection_output,['person'])
        triggered = len(matched) >= self.CROWD_THRESHOLD

        logger.info(
            f'[{self.USECASE_ID}] triggered={triggered}, count={len(matched)}, threshold={self.CROWD_THRESHOLD}'
        )
        return {'triggered': triggered, 'matched_objects': matched}