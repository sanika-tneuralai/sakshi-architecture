""" Mopping activity detection rule
"""

import logging
from typing import Dict, Any
from usecase.rules.base import BaseUsecaseRule

logger = logging.getLogger(__name__)

MOPPING_CLASS_NAME = 'mopping' # custom model class_id = 14

class MoppingDetectionRule(BaseUsecaseRule):
    USECASE_ID = 'mopping_detection'
    
    def evaluate(self, detection_output: Dict[str, Any]) -> Dict[str, Any]:
        all_detections = self.get_detections(detection_output)
        matched = [
            d for d in all_detections
            if d.get("class_name") == MOPPING_CLASS_NAME
        ]

        triggered = len(matched) > 0

        logger.info(
            f'[{self.USECASE_ID}] triggered={triggered}, matched={len(matched)}, threshold=any'
        )
        return {'triggered': triggered, 'matched_objects': matched}


