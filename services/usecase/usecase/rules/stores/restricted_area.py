"""
Uniform-based restricted area acccess control.

ROI check is already done by the detection service(in_roi flag)
class_id -> class_name(same as all other rules)
cooldown logic is removed, alert service handle this
stream_config / frame dimensions not needed - ROI is precomputed upstream
wrong uniform list is preserved in matched_objects - alert service reads it

"""

import logging
from typing import Dict, Any, List
from usecase.rules.base import BaseUsecaseRule

logger = logging.getLogger(__name__)

# class name sets - match what your custom model outputs
ALLOWED_UNIFORMS = {'uniform_grey', 'uniform_black'}
VIOLATION_UNIFORMS = {'uniform_beige', 'uniform_blue', 'uniform_red'}
ALL_UNIFORM_CLASSES = ALLOWED_UNIFORMS | VIOLATION_UNIFORMS

class RestrictedAreaRule(BaseUsecaseRule):
    USECASE_ID = 'restricted_area'

    def evaluate(self, detection_output: Dict[str, Any]) -> Dict[str, Any]:
        all_detections = self.get_detections(detection_output)

        allowed_matches = []
        violation_matches = []

        for d in all_detections:
            class_name = d.get('class_name','')

            if class_name  not in ALL_UNIFORM_CLASSES:
                continue

            if not d.get('in_roi', False):
                continue

            if class_name in VIOLATION_UNIFORMS:
                violation_matches.append(d)
            elif class_name in ALLOWED_UNIFORMS:
                allowed_matches.append(d)
        
        violation_count = len(violation_matches)
        allowed_count = len(allowed_matches)
        triggered = violation_count > 0

        # collect unique wrong uniform names - alert seervice uses this for the message
        # Restricted Area: unauthorised uniform detected(uniform_beige, uniform_red)"
        wrong_uniforms = list(set(d.get('class_name') for d in violation_matches))

        logger.info(
            f'[{self.USECASE_ID}] triggered={triggered}, violations = {violation_count}, allowed={allowed_count}, wrong_uniforms={wrong_uniforms}')
        
        return {
            'triggered': triggered,
            'matched_objects': violation_matches, # alert service reads this for details
            'violation_count': violation_count,
            'allowed_count': allowed_count,
            'wrong_uniforms': wrong_uniforms
        }


        
