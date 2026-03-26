"""
Smoking Detection usecase rules.
"""

import logging
from typing import Dict, Any
from usecase.rules.base import BaseUsecaseRule

logger = logging.getLogger(__name__)
SMOKING_CLASS_NAME = 'smoking' # custom model class_id = 13

class SmokingDetectionRule(BaseUsecaseRule):
    USECASE_ID = 'smoking_detection'
    
    def evaluate(self, detection_output: Dict[str, Any]) -> Dict[str, Any]:
        all_detections = self.get_detections(detection_output)
        matched = [
            d for d in all_detections
            if d.get("class_name") == SMOKING_CLASS_NAME
        ]

        triggered = len(matched) > 0

        logger.info(
            f'[{self.USECASE_ID}] triggered={triggered}, smoking_count={len(matched)}, threshold=any'
        )
        return {'triggered': triggered, 'matched_objects': matched}