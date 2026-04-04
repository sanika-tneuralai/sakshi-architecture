"""
Abstract base class for all usecase rules.

If USECASE_ID is a class attribute, the registry scanner can read it
without instantiating the class:
  - Discovery has no side effects
  - Failed instantiation doesn't hide a valid registration
  - The ID is part of the class contract, not runtime state
"""
import logging
from abc import ABC, abstractmethod
from typing import ClassVar, Dict, Any, List

logger = logging.getLogger(__name__)


class BaseUsecaseRule(ABC):
    """
    Abstract base for all usecase rules.

    Contract every subclass must satisfy:
      USECASE_ID: ClassVar[str]  — unique string ID for auto-discovery
      evaluate()                 — evaluation logic
    """

    USECASE_ID: ClassVar[str]

    def __init__(self, usecase_id: str):
        self.usecase_id = usecase_id
        logger.debug("[RULE] Initialized rule: %s", usecase_id)

    @abstractmethod
    def evaluate(self, detection_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate the usecase rule against detection output.

        Args:
            detection_output: slim payload with detections, snapshot_url,
                              camera_id, and optionally rois.

        Returns:
            {
                "triggered": bool,
                "matched_objects": list
            }
        """
        pass

    def get_detections(self, detection_output: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract the detections list from the payload."""
        return detection_output.get("detections", [])

    def get_in_roi_by_class(
        self,
        detection_output: Dict[str, Any],
        class_names: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Return all detections whose class_name is in *class_names*.

        Args:
            detection_output: detection payload
            class_names: e.g. ['person'] or ['car', 'truck']

        Returns:
            Filtered list of matching detections.
        """
        return [
            d for d in self.get_detections(detection_output)
            if d.get("class_name", "") in class_names
        ]