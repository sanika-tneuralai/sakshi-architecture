"""
Crowd in ROI usecase rule.

Rule: Trigger if 3 or more persons are detected inside ROI.
Condition: class_name == "person" AND in_roi == true AND count >= 3
"""
from typing import Dict, Any
from usecase.rules.base import BaseUsecaseRule


class CrowdInROIRule(BaseUsecaseRule):
    """
    Usecase: Crowd (3+ persons) detected inside Region of Interest.
    
    Triggers when 3 or more persons are detected with in_roi == true.
    """
    
    CROWD_THRESHOLD = 3  # Minimum number of persons to be considered a crowd
    
    def evaluate(self, detection_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate if a crowd (3+ persons) is inside ROI.
        
        Args:
            detection_output: Detection API response
            
        Returns:
            Evaluation result with triggered status and matched objects
        """
        print(f"\\n[RULE:{self.usecase_id}] ========== EVALUATION START ==========")
        print(f"[RULE:{self.usecase_id}] Crowd threshold: {self.CROWD_THRESHOLD} persons")
        
        detections = self.get_detections(detection_output)
        print(f"[RULE:{self.usecase_id}] Processing {len(detections)} detections")
        
        matched_objects = []
        
        for idx, detection in enumerate(detections):
            class_name = detection.get("class_name", "")
            in_roi = detection.get("in_roi", False)
            confidence = detection.get("confidence", 0.0)
            
            # Rule: person AND in_roi (collect all persons in ROI)
            if class_name == "person" and in_roi:
                print(f"[RULE:{self.usecase_id}] Detection {idx+1}: Person in ROI (conf={confidence:.2f})")
                matched_objects.append(detection)
        
        # Trigger only if crowd threshold is met
        triggered = len(matched_objects) >= self.CROWD_THRESHOLD
        
        if triggered:
            print(f"[RULE:{self.usecase_id}] ✓✓✓ CROWD DETECTED: {len(matched_objects)} persons >= {self.CROWD_THRESHOLD}")
        else:
            print(f"[RULE:{self.usecase_id}] ✗ No crowd: {len(matched_objects)} persons < {self.CROWD_THRESHOLD}")
        
        print(f"[RULE:{self.usecase_id}] Evaluation complete:")
        print(f"[RULE:{self.usecase_id}]   - Triggered: {triggered}")
        print(f"[RULE:{self.usecase_id}]   - Matched: {len(matched_objects)} person(s)")
        print(f"[RULE:{self.usecase_id}] ========== EVALUATION END ==========\\n")
        
        return {
            "triggered": triggered,
            "matched_objects": matched_objects
        }
