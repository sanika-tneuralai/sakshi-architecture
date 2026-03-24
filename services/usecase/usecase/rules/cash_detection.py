"""
Cash and open drawer detection usecase rules.

This rule have 3 sub scenarios:
    -cash only
    -drawer only
    -cash +drawer

    All three trigger the rule. The alert_message and event_type tell the alert service which scenario occured.
    Alert service uses matched_objects to distinguish -it checks class_names present in matched objects.
"""

import logging
from typing import Dict, Any
from usecase.rules.base import BaseUsecaseRule

logger = logging.getLogger(__name__)

CASH_CLASS_NAME = 'cash'
DRAWER_CLASS_NAME = 'drawer'

class CashDetectionRule(BaseUsecaseRule):
    USECASE_ID = 'cash_detection'
    
    def evaluate(self, detection_output: Dict[str, Any]) -> Dict[str, Any]:
        
        all_detections = self.get_detections(detection_output)
        cash_matches = [d for d in all_detections if d.get("class_name") == CASH_CLASS_NAME]
        drawer_matches = [d for d in all_detections if d.get("class_name") == DRAWER_CLASS_NAME]

        cash_count = len(cash_matches)
        drawer_count = len(drawer_matches)

        triggered = cash_count > 0 or drawer_count > 0

        #determine scenario - alert service reads this to build the right alert message
        if cash_count > 0 and drawer_count > 0:
            event_type = 'cash_and_drawer'
            alert_message = 'Cash + open drawer detected'
        elif cash_count >0:
            event_type = 'cash_detected'
            alert_message = 'Cash detected'
        else:
            event_type = 'drawer_detected'
            alert_message = 'Open drawer detected'
    
        matched = cash_matches + drawer_matches

        logger.info(
            f'[{self.USECASE_ID}] triggered={triggered}, cash_count={cash_count}, drawer_count={drawer_count}, event_type = {event_type if triggered else "none"}'
        )
        
        return {
            'triggered': triggered, 
            'matched_objects': matched,
            'event_type': event_type if triggered else None,
            'alert_message': alert_message if triggered else None
        }