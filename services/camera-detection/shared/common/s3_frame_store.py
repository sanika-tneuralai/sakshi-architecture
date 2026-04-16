"""
S3-backed frame storage.
Uploads OpenCV BGR frames as JPEG and returns a public S3 URL.
Downloaded frames are returned as numpy arrays ready for cv2 processing.
"""
import logging
import time
from typing import Optional

import boto3
import cv2
import numpy as np

from shared.common.config import Config

logger = logging.getLogger(__name__)


class S3FrameStore:
    """Upload/download numpy frames (OpenCV BGR) to/from S3 as JPEG."""

    def __init__(self):
        bucket = Config.get_s3_bucket()
        if not bucket:
            raise ValueError("AWS_S3_BUCKET env var is required")
        self._bucket = bucket
        self._region = Config.get_s3_region()
        self._client = boto3.client(
            "s3",
            region_name=self._region,
            aws_access_key_id=Config.get_s3_access_key(),
            aws_secret_access_key=Config.get_s3_secret_key(),
        )

    def _key(self, camera_id: str, frame_id: str) -> str:
        """S3 object key: frames/<camera_id>/<frame_id>.jpg"""
        return f"frames/{camera_id}/{frame_id}.jpg"

    def _public_url(self, key: str) -> str:
        """Build the HTTPS URL for an S3 object."""
        return f"https://{self._bucket}.s3.{self._region}.amazonaws.com/{key}"

    def upload_frame(self, camera_id: str, frame: np.ndarray, frame_id: Optional[str] = None) -> str:
        """
        Encode frame as JPEG, upload to S3, and return the public HTTPS URL.

        Args:
            camera_id: camera identifier (used as S3 prefix)
            frame:     OpenCV BGR numpy array
            frame_id:  optional unique ID; defaults to millisecond timestamp

        Returns:
            Public S3 URL string
        """
        if frame_id is None:
            frame_id = str(int(time.time() * 1000))

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        key = self._key(camera_id, frame_id)

        self._client.put_object(
            Bucket=self._bucket,
            Key=key,
            Body=buffer.tobytes(),
            ContentType="image/jpeg",
        )
        url = self._public_url(key)
        logger.info(f"[S3] Uploaded frame: {url}")
        return url

    def download_frame(self, camera_id: str, frame_id: str) -> Optional[np.ndarray]:
        """
        Download a frame from S3 and decode it back to a numpy BGR array.
        Returns None if the object does not exist.
        """
        key = self._key(camera_id, frame_id)
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=key)
            data = response["Body"].read()
            arr = np.frombuffer(data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            logger.info(f"[S3] Downloaded frame: {key}")
            return frame
        except self._client.exceptions.NoSuchKey:
            logger.warning(f"[S3] Frame not found: {key}")
            return None

    def download_frame_from_url(self, url: str) -> Optional[np.ndarray]:
        """
        Download a frame using its full S3 HTTPS URL.
        Extracts the key from the URL and calls get_object.
        Returns None if not found.
        """
        prefix = f"https://{self._bucket}.s3.{self._region}.amazonaws.com/"
        if not url.startswith(prefix):
            logger.error(f"[S3] URL does not match configured bucket: {url}")
            return None
        key = url[len(prefix):]
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=key)
            data = response["Body"].read()
            arr = np.frombuffer(data, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        except self._client.exceptions.NoSuchKey:
            logger.warning(f"[S3] Frame not found at key: {key}")
            return None


# Lazy singleton — only created on first use
_store: Optional[S3FrameStore] = None


def get_s3_frame_store() -> S3FrameStore:
    global _store
    if _store is None:
        _store = S3FrameStore()
    return _store
