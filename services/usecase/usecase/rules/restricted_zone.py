"""
Restricted Zone Breach usecase rule.

Rule: Trigger if ANY vehicle (car, truck, bus, bike) is detected inside ROI.
Condition: class_name in ["car", "truck", "bus", "motorcycle", "bicycle"] AND in_roi == true
"""
from typing import Dict, Any
from usecase.rules.base import BaseUsecaseRule


class RestrictedZoneRule(BaseUsecaseRule):
    """
    Usecase: Vehicle detected in restricted zone (ROI).
    
    Triggers when any vehicle is detected with in_roi == true.
    Useful for monitoring no-vehicle zones, pedestrian areas, etc.
    """
    
    # Vehicle classes that trigger this usecase
    VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle", "bicycle", "bike"]
    
    def evaluate(self, detection_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate if any vehicle is inside restricted zone (ROI).
        
        Args:
            detection_output: Detection API response
            
        Returns:
            Evaluation result with triggered status and matched objects
        """
        print(f"\\n[RULE:{self.usecase_id}] ========== EVALUATION START ==========")
        print(f"[RULE:{self.usecase_id}] Restricted vehicle classes: {', '.join(self.VEHICLE_CLASSES)}")
        
        detections = self.get_detections(detection_output)
        print(f"[RULE:{self.usecase_id}] Processing {len(detections)} detections")
        
        matched_objects = []
        
        for idx, detection in enumerate(detections):
            class_name = detection.get("class_name", "")
            in_roi = detection.get("in_roi", False)
            confidence = detection.get("confidence", 0.0)
            
            print(f"[RULE:{self.usecase_id}] Detection {idx+1}: class='{class_name}', in_roi={in_roi}, conf={confidence:.2f}")
            
            # Rule: vehicle class AND in_roi
            if class_name in self.VEHICLE_CLASSES and in_roi:
                print(f"[RULE:{self.usecase_id}] ✓✓✓ BREACH: {class_name.upper()} in restricted zone (conf={confidence:.2f})")
                matched_objects.append(detection)
        
        triggered = len(matched_objects) > 0
        
        print(f"[RULE:{self.usecase_id}] Evaluation complete:")
        print(f"[RULE:{self.usecase_id}]   - Triggered: {triggered}")
        print(f"[RULE:{self.usecase_id}]   - Matched: {len(matched_objects)} vehicle(s)")
        print(f"[RULE:{self.usecase_id}] ========== EVALUATION END ==========\\n")
        
        return {
            "triggered": triggered,
            "matched_objects": matched_objects
        }
