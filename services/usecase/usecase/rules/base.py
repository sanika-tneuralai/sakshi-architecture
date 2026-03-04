"""
Base class for all usecase rules.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseUsecaseRule(ABC):
    """
    Abstract base class for usecase rules.
    
    Each usecase rule must:
    - Accept detection_output
    - Return triggered (bool) and matched_objects (list)
    - Never access camera streams
    - Never access ROI geometry
    - Never call external APIs
    """
    
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
