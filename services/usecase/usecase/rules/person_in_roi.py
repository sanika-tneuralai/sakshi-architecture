"""
Person in ROI usecase rule.

Rule: Trigger if ANY person is detected inside ROI.
Condition: class_name == "person" AND in_roi == true
"""
import logging
from typing import Dict, Any
from usecase.rules.base import BaseUsecaseRule

logger = logging.getLogger(__name__)



class PersonInROIRule(BaseUsecaseRule):

    """
    Usecase: Person detected inside Region of Interest.
    
    Triggers when at least one person is detected with in_roi == true.
    """
    USECASE_ID = 'person_in_roi' # auto dicovery key
    
    def evaluate(self, detection_output: Dict[str, Any]) -> Dict[str, Any]:
        matched = self.get_in_roi_by_class(detection_output, ['person'])
        triggered = len(matched) > 0

        logger.info(f'[{self.USECASE_ID}] triggered = {triggered}, matched={len(matched)}')
        return {'triggered':triggered, 'matched_objects': matched}
    

     
# ─────────────────────────────────────────────────────────────
# HOW TO ADD A NEW USECASE (template — copy this file):
#
# 1. Create usecase/rules/your_usecase.py
# 2. Copy this file's structure
# 3. Change USECASE_ID to your unique string
# 4. Change the class name
# 5. Implement evaluate() using the base helpers
# 6. Done — no other files need to change
# ─