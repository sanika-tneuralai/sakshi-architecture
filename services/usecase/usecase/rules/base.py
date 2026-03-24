"""
Abstract Base class for all usecase rules.
Added USECASE_ID class attribute.

If the ID is a class attribute, the registry scanner can read it Withour instantiating the class. This means:
-Discover has no side effects
-failed instantiation doesn't hide a valid registration
-The ID is part of the class contract, not runtime state
"""
from abc import ABC, abstractmethod
from typing import ClassVar, Dict, Any, List


class BaseUsecaseRule(ABC):
    """
    Abstract base for all usecase rules.
    CONTRACT (what every subcalss must provide):
    USECASE_ID: classVar[str] - uunique string ID for auto-discovery
    evaluate() - evaluation logic
    """
    # subclass must define thhis as a classs level string
    # example: USECASE_ID = 'person_in_roi'
    # the auto-discovery scanner reads this without instantiating the class

    USECASE_ID: ClassVar[str]

    def __init__(self, usecase_id: str):
        """
        Initialize the usecase rule.
        
        Args:
            usecase_id: Unique identifier for this usecase
        """
        self.usecase_id = usecase_id
        print(f"[RULE] Initialized rule: {usecase_id}")
    
    @abstractmethod
    def evaluate(self, detection_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the usecase rule against detection output.
        
        Args:
            detection_output: Detection API response containing detections with in_roi flags
            each detection has: class_name, confidence, in_roi,bbox
            
        Returns:
            Dictionary containing:
                - triggered (bool): Whether the usecase condition is met
                - matched_objects (list): List of objects that matched the rule
        """
        pass
    
    def get_detections(self, detection_output: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Helper method to extract detections from detection output.
        
        Args:
            detection_output: Detection API response
            
        Returns:
            List of detection objects
        """
        return detection_output.get("detections", [])

    def get_in_roi_detections(self, detection_output: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Helper: returns all detections.

        ROI has been removed from the detection service — all detections are
        now evaluated regardless of position. This method is kept for
        backward compatibility with existing rules.
        """
        return self.get_detections(detection_output)

    def get_in_roi_by_class(
            self,
            detection_output: Dict[str, Any],
            class_names: List[str],
            ) -> List[Dict[str, Any]]:
            """
            Helper: get detections matching specific class names.

            ROI filtering removed — returns all detections of the given classes.

            Args:
                detection_output: detection payload
                class_names: list of class names to match, e.g ['person'] or ['car', 'truck']
            Returns:
                List of matching detections
            """
            return [
                d for d in self.get_detections(detection_output)
                if d.get("class_name", "") in class_names
            ]