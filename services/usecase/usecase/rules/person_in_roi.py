"""
Person in ROI usecase rule.

Rule: Trigger if ANY person is detected inside ROI.
Condition: class_name == "person" AND in_roi == true
"""
from typing import Dict, Any
from usecase.rules.base import BaseUsecaseRule


class PersonInROIRule(BaseUsecaseRule):
    """
    Usecase: Person detected inside Region of Interest.
    
    Triggers when at least one person is detected with in_roi == true.
    """
    
    def evaluate(self, detection_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate if any person is inside ROI.
        
        Args:
            detection_output: Detection API response
            
        Returns:
            Evaluation result with triggered status and matched objects
        """
        print(f"\\n[RULE:{self.usecase_id}] ========== EVALUATION START ==========")
        
        detections = self.get_detections(detection_output)
        print(f"[RULE:{self.usecase_id}] Processing {len(detections)} detections")
        
        matched_objects = []
        
        for idx, detection in enumerate(detections):
            class_name = detection.get("class_name", "")
            in_roi = detection.get("in_roi", False)
            confidence = detection.get("confidence", 0.0)
            
            print(f"[RULE:{self.usecase_id}] Detection {idx+1}: class='{class_name}', in_roi={in_roi}, conf={confidence:.2f}")
            
            # Rule: person AND in_roi
            if class_name == "person" and in_roi:
                print(f"[RULE:{self.usecase_id}] ✓✓✓ MATCH: Person in ROI (confidence: {confidence:.2f})")
                matched_objects.append(detection)
        
        triggered = len(matched_objects) > 0
        
        print(f"[RULE:{self.usecase_id}] Evaluation complete:")
        print(f"[RULE:{self.usecase_id}]   - Triggered: {triggered}")
        print(f"[RULE:{self.usecase_id}]   - Matched: {len(matched_objects)} person(s)")
        print(f"[RULE:{self.usecase_id}] ========== EVALUATION END ==========\\n")
        
        return {
            "triggered": triggered,
            "matched_objects": matched_objects
        }
