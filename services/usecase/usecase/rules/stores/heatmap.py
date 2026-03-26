"""
Track person positions for density visualization.
Architecural Note:
    - Bbox format is : {x1,y1,x2,y2}
    -Center point is calculated as ((x1+x2)/2 / frame_W
Also this rule never triggers an alert (return triggered = False always)
It exists purely to produce heatmap_points for the dashboard
The dashboard reads matched_objects to get the positions.

"""

import logging
from typing import Dict,Any
from usecase.rules.base import BaseUsecaseRule

logger = logging.getLogger(__name__)

PERSON_CLASS_NAME = 'person'
FRAME_W = 1280
FRAME_H = 720

class HeatmapRule(BaseUsecaseRule):
    USECASE_ID = 'heatmap'

    def evaluate(self, detection_output: Dict[str, Any]) -> Dict[str, Any]:
        all_detections = self.get_detections(detection_output)
        person_detections = [
            d for d in all_detections
            if d.get("class_name") == PERSON_CLASS_NAME
        ]
# compute normalized center points for each person (0.0 to 1.0)
#  Dashboard uses these directly for heatmap rendering.
        heatmap_points = []
        for d in person_detections:
            bbox = d.get("bbox", {})
            cx = ((bbox.get("x1", 0) + bbox.get("x2", 0)) / 2) / FRAME_W
            cy = ((bbox.get("y1", 0) + bbox.get("y2", 0)) / 2) / FRAME_H
            heatmap_points.append({'x': round(cx, 4), 'y': round(cy, 4), 'confidence': d.get('confidence', 0.0)})

        # Attach heatmap_points inside each matched object so the dashboard can read them from matched_objects in UsecaseResult

        matched = [
            {**d, 'heatmap_points': heatmap_points[i]}
            for i, d in enumerate(person_detections)
        ]
   
        logger.info(
            f'[{self.USECASE_ID}] heatmap_points_count={len(heatmap_points)}'
        )
        return {'triggered': False, 'matched_objects': matched}