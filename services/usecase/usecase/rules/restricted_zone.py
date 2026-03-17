"""
usecase/rules/restricted_zone.py
"""
import logging
from typing import Dict, Any
from usecase.rules.base import BaseUsecaseRule

logger = logging.getLogger(__name__)


class RestrictedZoneRule(BaseUsecaseRule):
    USECASE_ID = "restricted_zone_breach"
    VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle", "bike"]

    def evaluate(self, detection_output: Dict[str, Any]) -> Dict[str, Any]:
        matched = self.get_in_roi_by_class(detection_output, self.VEHICLE_CLASSES)
        triggered = len(matched) > 0

        logger.info(f"[{self.USECASE_ID}] triggered={triggered}, matched={len(matched)}")
        return {"triggered": triggered, "matched_objects": matched}