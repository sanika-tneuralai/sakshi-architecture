"""
Phone usage detection rule

"""
import logging
from typing import Dict, Any
from usecase.rules.base import BaseUsecaseRule

logger = logging.getLogger(__name__)

PHONE_CLASS_NAMES = 'phone_using'

class PhoneDetectionRule(BaseUsecaseRule):
    USECASE_ID = 'phone_detection'
    
    def evaluate(self, detection_output: Dict[str, Any]) -> Dict[str, Any]:
        all_detections = self.get_detections(detection_output)
        matched = [
            d for d in all_detections
            if d.get("class_name") == PHONE_CLASS_NAMES
        ]

        phone_count = len(matched)
        triggered = len(matched) > 0

        logger.info(
            f'[{self.USECASE_ID}] triggered={triggered}, phone_count={phone_count}'
        )
        
        return {'triggered': triggered, 'matched_objects': matched}