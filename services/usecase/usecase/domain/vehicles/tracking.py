"""
Centroid-based object tracker.

Assigns stable string track_ids to detections across frames using
nearest-centroid matching. No external tracking libraries required.
Thread-safe — a single tracker instance can be shared across threads.
"""
import logging
import threading
import uuid
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _centroid(bbox: dict) -> Tuple[float, float]:
    return (bbox["x1"] + bbox["x2"]) / 2.0, (bbox["y1"] + bbox["y2"]) / 2.0


class CentroidTracker:
    """
    Tracks objects across frames by matching centroids frame-to-frame.

    Call update(detections) each frame with a list of detection dicts
    (each must contain a "bbox" dict with x1/y1/x2/y2).
    Returns the same list with "track_id" added to every detection.

    Usage:
        tracker = CentroidTracker(max_disappeared=10, max_distance=80)
        tracked = tracker.update(detections)
    """

    def __init__(self, max_disappeared: int = 10, max_distance: float = 80.0):
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self._next_id = 0
        # Short session prefix (8 hex chars) — changes on every restart so
        # track_ids like "a3f1b2c0_trk_0000" never collide with a previous run.
        self._session = uuid.uuid4().hex[:8]
        # OrderedDict: track_id (str) -> centroid np.ndarray shape (2,)
        self._centroids: OrderedDict = OrderedDict()
        # track_id -> consecutive frames without a match
        self._disappeared: Dict[str, int] = {}
        self._lock = threading.Lock()

    def _new_id(self) -> str:
        tid = f"{self._session}_trk_{self._next_id:04d}"
        self._next_id += 1
        return tid

    def _register(self, centroid: np.ndarray) -> str:
        tid = self._new_id()
        self._centroids[tid] = centroid
        self._disappeared[tid] = 0
        return tid

    def _deregister(self, tid: str) -> None:
        self._centroids.pop(tid, None)
        self._disappeared.pop(tid, None)

    def update(self, detections: List[dict]) -> List[dict]:
        """
        Match detections to existing tracks; return detections with track_id added.

        - New detections that don't match any existing track → new track_id
        - Existing tracks with no matching detection → increment disappeared counter
        - Tracks exceeding max_disappeared → deregistered
        """
        with self._lock:
            if not detections:
                for tid in list(self._disappeared):
                    self._disappeared[tid] += 1
                    if self._disappeared[tid] > self.max_disappeared:
                        self._deregister(tid)
                return detections

            input_centroids = np.array(
                [_centroid(d["bbox"]) for d in detections], dtype=np.float32
            )

            if not self._centroids:
                for i, det in enumerate(detections):
                    tid = self._register(input_centroids[i])
                    det["track_id"] = tid
                return detections

            existing_ids = list(self._centroids.keys())
            existing_centroids = np.array(
                [self._centroids[t] for t in existing_ids], dtype=np.float32
            )

            # Build distance matrix: shape (n_existing, n_new)
            try:
                from scipy.spatial.distance import cdist
                D = cdist(existing_centroids, input_centroids)
            except ImportError:
                # Pure-numpy fallback
                D = np.linalg.norm(
                    existing_centroids[:, None, :] - input_centroids[None, :, :],
                    axis=2,
                )

            # Greedy matching: sort existing tracks by their minimum distance to any detection
            row_inds = D.min(axis=1).argsort()
            col_inds = D.argmin(axis=1)[row_inds]

            matched_rows: set = set()
            matched_cols: set = set()
            assignment: Dict[int, str] = {}  # detection index -> track_id

            for row, col in zip(row_inds, col_inds):
                if row in matched_rows or col in matched_cols:
                    continue
                if D[row, col] > self.max_distance:
                    continue
                tid = existing_ids[row]
                self._centroids[tid] = input_centroids[col]
                self._disappeared[tid] = 0
                assignment[col] = tid
                matched_rows.add(row)
                matched_cols.add(col)

            # Age out unmatched existing tracks
            for row, tid in enumerate(existing_ids):
                if row not in matched_rows:
                    self._disappeared[tid] += 1
                    if self._disappeared[tid] > self.max_disappeared:
                        self._deregister(tid)

            # Register new detections with no match
            for col in range(len(detections)):
                if col not in matched_cols:
                    tid = self._register(input_centroids[col])
                    assignment[col] = tid

            for i, det in enumerate(detections):
                det["track_id"] = assignment.get(i, self._new_id())

            return detections

    def reset(self) -> None:
        with self._lock:
            self._centroids.clear()
            self._disappeared.clear()
            self._next_id = 0

    # ------------------------------------------------------------------
    # Redis serialization helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """
        Serialize tracker state to a plain JSON-compatible dict.

        numpy arrays stored in ``_centroids`` are converted to plain lists
        so the result can be safely passed to ``json.dumps``.
        """
        with self._lock:
            return {
                "session": self._session,
                "next_id": self._next_id,
                "centroids": {
                    tid: centroid.tolist()
                    for tid, centroid in self._centroids.items()
                },
                "disappeared": dict(self._disappeared),
            }

    def from_dict(self, data: dict) -> None:
        """
        Restore tracker state from a plain dict (as produced by ``to_dict``).

        Lists stored in ``data["centroids"]`` are converted back to
        numpy arrays so ``update()`` can operate on them unchanged.
        The session prefix is restored so IDs remain consistent within
        the same run.
        """
        with self._lock:
            self._session = data.get("session", self._session)
            self._next_id = data.get("next_id", 0)
            self._centroids = OrderedDict(
                (tid, np.array(centroid, dtype=np.float32))
                for tid, centroid in data.get("centroids", {}).items()
            )
            self._disappeared = dict(data.get("disappeared", {}))
