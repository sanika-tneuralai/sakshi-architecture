"""
Bag Detection usecase rules.

Rule: Trigger if ANY bag is detected inside ROI.
Condition: class_name == "bag" AND in_roi == true

"""

import logging
from typing import Dict, Any
from usecase.rules.base import BaseUsecaseRule

logger = logging.getLogger(__name__)

COCO_BAG_CLASSES = ['backpack', 'handbag', 'suitcase']
BAG_CONFIDENCE_THRESHOLD = 0.5

class BagDetectionRule(BaseUsecaseRule):
    """
    Usecase: Bag detection

    """
    USECASE_ID = 'bag_detection' #auto-discovery picks this up and nothing else to register

    def evaluate(self, detection_output: Dict[str, Any]) -> Dict[str, Any]:

        all_detections = self.get_detections(detection_output) # get_in_roi_by_class is the shared base helper 
        matched = [
            d for d in all_detections
            if d.get("class_name") in COCO_BAG_CLASSES
            and d.get('confidence', 0) >= BAG_CONFIDENCE_THRESHOLD
        ]

        triggered = len(matched) > 0

        logger.info(
            f'[{self.USECASE_ID}] triggered={triggered}, matched={len(matched)}, bag_count={len(matched)}, threshold={BAG_CONFIDENCE_THRESHOLD}')
        return {'triggered': triggered, 'matched_objects': matched}

