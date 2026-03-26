"""
Uniform compliance / dress code monitoring rule.
1. class_id -> class_name (same as all other rules)
2. class_coutns dict - filter detections by class name
3. alert only on untucked shirts (same trigger logic as monolith)
4. frame_num % 300 cooldown -> alert service TTL
5. uniform_counts and compliance_counts preserved in return for alert service to build detailed message
"""

import logging
from typing import Dict, Any, List
from usecase.rules.base import BaseUsecaseRule

logger = logging.getLogger(__name__)

UNIFORM_CLASSES = {'uniform_grey', 'uniform_black', 'uniform_beige', 'uniform_blue', 'uniform_red'}

COMPLIANCE_CLASSES = {
    'shirt_tucked', 'shirt_untucked', 'id_visible', 'name_tag_visible', 'shoes_black'
}

ALL_CLASSES = UNIFORM_CLASSES | COMPLIANCE_CLASSES

class DressCodeRule(BaseUsecaseRule):
    USECASE_ID = 'dress_code'

    def evaluate(self, detection_output: Dict[str, Any]) -> Dict[str, Any]:
        all_detections = self.get_detections(detection_output)

        # seperate into uniform and compliance detections
        uniform_matches = [d for d in all_detections if d.get('class_name') in UNIFORM_CLASSES]
        compliance_matches = [d for d in all_detections if d.get('class_name') in COMPLIANCE_CLASSES]

        # Violations - only untucked shirts trigger alert

        untucked_matches = [d for d in compliance_matches if d.get('class_name') == 'shirt_untucked']
        untucked_count = len(untucked_matches)
        triggered = untucked_count > 0

        # count per class - alert service use these for the detailed messages
        uniform_counts = _count_by_class(uniform_matches)
        compliance_counts = _count_by_class(compliance_matches)

        logger.info(
            f'[{self.USECASE_ID}] triggered={triggered}, untucked_count={untucked_count}, uniform_counts={uniform_counts}, compliance_counts={compliance_counts}')
        
        return {
            'triggered': triggered,
            'matched_objects': untucked_matches, # alert service reads this for details
            'uniform_counts': uniform_counts,
            'compliance_counts': compliance_counts,
            'uniform_detected': bool(uniform_matches) # for dashboard to filter only uniform detections
        }

def _count_by_class(detections: List[Dict[str, Any]]) -> Dict[str, int]:
    """
    count detections per class_name. Replace the monolith class_count dict approach.
        e.g. [{"class_name": "shirt_untucked"}, {"class_name": "shirt_untucked"}]
         → {"shirt_untucked": 2}
    """
    counts: Dict[str, int] = {}
    for d in detections:
        name = d.get('class_name', 'unknown')
        counts[name] = counts.get(name, 0) + 1
    return counts