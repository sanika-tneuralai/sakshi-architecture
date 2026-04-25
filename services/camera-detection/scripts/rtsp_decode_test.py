"""
Standalone RTSP decode tester.

Reads the RTSP stream the same way camera/streams/opencv.py does, but with
NO detection / S3 / DB / Hailo overhead, and counts:
  - frames received
  - read failures (cap.read() returned False)
  - cascades (10 consecutive read failures, mirrors prod max_errors)
  - reconnects performed

It also captures FFmpeg's stderr (where the [h264 @ ...] decode warnings go)
and counts those lines per category.

Usage:
    python scripts/rtsp_decode_test.py "rtsp://admin:PASS@192.168.1.231:554/stream2" --duration 600

Outputs a JSON summary at the end and prints stats every 30 seconds.

Compare two runs to isolate where the problem is:
  1. Run this with your service stopped → if errors still happen, it's the camera/network.
  2. Run this with your service running on the same Pi → if errors are much worse, it's local resource contention.
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from datetime import datetime

import cv2


# Mirror the env vars from opencv.py so we test the same decode path
os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = (
    'rtsp_transport;tcp'
    '|max_delay;500000'
    '|stimeout;30000000'
    '|buffer_size;16777216'
    '|rtbufsize;100M'
    '|fflags;discardcorrupt'
)


H264_ERROR_PATTERNS = [
    (re.compile(r'error while decoding MB'), 'mb_decode_error'),
    (re.compile(r'left block unavailable'), 'left_block_unavailable'),
    (re.compile(r'cabac decode of qscale diff failed'), 'cabac_qscale_failed'),
    (re.compile(r'concealing \d+ DC, \d+ AC'), 'concealment'),
    (re.compile(r'method PLAY failed: 429'), 'rtsp_429_too_many_sessions'),
    (re.compile(r'method PLAY failed'), 'rtsp_play_failed_other'),
    (re.compile(r'no frame!'), 'no_frame'),
    (re.compile(r'invalid NAL unit'), 'invalid_nal'),
]


def categorize_stderr_line(line: str) -> str:
    for pat, name in H264_ERROR_PATTERNS:
        if pat.search(line):
            return name
    return None


class StderrSniffer:
    """
    Tee stderr to a pipe so we can count FFmpeg's decode warnings per category.
    OpenCV's FFmpeg backend writes these to fd 2 directly — Python can't see them
    via sys.stderr without redirecting at the OS level.
    """

    def __init__(self):
        self.counts = Counter()
        self.recent_lines = []  # last 50 lines for end-of-run dump
        self.stopped = False
        self._reader_thread = None
        self._original_stderr_fd = None
        self._read_fd = None
        self._write_fd = None

    def start(self):
        # Save original stderr fd
        self._original_stderr_fd = os.dup(2)
        # Create pipe; redirect fd 2 (stderr) into the write end
        self._read_fd, self._write_fd = os.pipe()
        os.dup2(self._write_fd, 2)
        os.close(self._write_fd)

        self._reader_thread = threading.Thread(target=self._reader, daemon=True)
        self._reader_thread.start()

    def _reader(self):
        with os.fdopen(self._read_fd, 'r', errors='replace', buffering=1) as f:
            for line in f:
                if self.stopped:
                    return
                line = line.rstrip()
                # Pass through to original stderr so the user still sees the messages
                os.write(self._original_stderr_fd, (line + "\n").encode(errors="replace"))
                cat = categorize_stderr_line(line)
                if cat:
                    self.counts[cat] += 1
                    self.recent_lines.append(line)
                    if len(self.recent_lines) > 50:
                        self.recent_lines.pop(0)

    def stop(self):
        self.stopped = True


def open_capture(url: str):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, [
        cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 30000,
        cv2.CAP_PROP_READ_TIMEOUT_MSEC, 30000,
    ])
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rtsp_url")
    ap.add_argument("--duration", type=int, default=600,
                    help="Test duration in seconds (default: 600 = 10 min)")
    ap.add_argument("--max-errors", type=int, default=10,
                    help="Consecutive read failures before counting a 'cascade' (default: 10)")
    ap.add_argument("--report-interval", type=int, default=30,
                    help="Print interim stats every N seconds (default: 30)")
    args = ap.parse_args()

    sniffer = StderrSniffer()
    sniffer.start()

    print(f"=== RTSP decode tester ===")
    print(f"URL:              {args.rtsp_url}")
    print(f"Duration:         {args.duration}s")
    print(f"Cascade threshold: {args.max_errors} consecutive read failures")
    print(f"Started:          {datetime.now().isoformat()}")
    print("---")

    cap = open_capture(args.rtsp_url)
    if not cap.isOpened():
        print("ERROR: Failed to open RTSP stream.")
        sys.exit(2)

    start_ts = time.monotonic()
    last_report_ts = start_ts
    frame_count = 0
    read_failure_count = 0
    cascade_count = 0
    reconnect_count = 0
    consecutive_failures = 0
    first_frame_ts = None

    try:
        while time.monotonic() - start_ts < args.duration:
            ret, frame = cap.read()

            if not ret:
                read_failure_count += 1
                consecutive_failures += 1
                if consecutive_failures >= args.max_errors:
                    cascade_count += 1
                    print(f"[{datetime.now().isoformat()}] CASCADE #{cascade_count} "
                          f"({consecutive_failures} consecutive failures) — reconnecting")
                    cap.release()
                    time.sleep(1.0)  # grace period for camera to free the session
                    cap = open_capture(args.rtsp_url)
                    reconnect_count += 1
                    consecutive_failures = 0
                    if not cap.isOpened():
                        print(f"[{datetime.now().isoformat()}] reconnect failed, retrying in 5s...")
                        time.sleep(5)
                        cap = open_capture(args.rtsp_url)
                else:
                    time.sleep(0.05)
                continue

            consecutive_failures = 0
            frame_count += 1
            if first_frame_ts is None:
                first_frame_ts = time.monotonic()

            # Periodic report
            now = time.monotonic()
            if now - last_report_ts >= args.report_interval:
                elapsed = now - start_ts
                eff_fps = frame_count / elapsed if elapsed > 0 else 0
                print(f"[{datetime.now().isoformat()}] "
                      f"elapsed={elapsed:.0f}s frames={frame_count} "
                      f"avg_fps={eff_fps:.1f} read_fails={read_failure_count} "
                      f"cascades={cascade_count} reconnects={reconnect_count} "
                      f"decode_warnings={dict(sniffer.counts)}")
                last_report_ts = now

    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        cap.release()
        sniffer.stop()
        # Small delay to flush sniffer
        time.sleep(0.2)

    elapsed = time.monotonic() - start_ts
    summary = {
        "url": args.rtsp_url,
        "duration_seconds": round(elapsed, 1),
        "frames_received": frame_count,
        "avg_fps": round(frame_count / elapsed, 2) if elapsed > 0 else 0,
        "read_failures": read_failure_count,
        "cascades": cascade_count,
        "reconnects": reconnect_count,
        "ffmpeg_warnings_by_type": dict(sniffer.counts),
        "cascades_per_hour": round(cascade_count / (elapsed / 3600), 2) if elapsed > 0 else 0,
        "warnings_per_hour": {
            k: round(v / (elapsed / 3600), 2) for k, v in sniffer.counts.items()
        } if elapsed > 0 else {},
    }

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    if sniffer.recent_lines:
        print("\n--- Last 10 FFmpeg warning lines ---")
        for line in sniffer.recent_lines[-10:]:
            print(line)


if __name__ == "__main__":
    main()
