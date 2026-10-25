"""Webcam capture and MediaPipe pose, off the UI thread."""

from __future__ import annotations

import os
import shutil
import time
import urllib.request

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QImage

from motionlane.pose.sample import BodySample, warning_for
from motionlane.profile import model_path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

_LEFT_SHOULDER = 11
_RIGHT_SHOULDER = 12
_LEFT_HIP = 23
_RIGHT_HIP = 24
_LEFT_ANKLE = 27
_RIGHT_ANKLE = 28
_NOSE = 0


def ensure_model(report) -> str:
    dest = model_path()
    if os.path.isfile(dest) and os.path.getsize(dest) > 500_000:
        return dest
    report("Downloading the pose model. This happens once.")
    part = dest + ".part"
    request = urllib.request.Request(MODEL_URL, headers={"User-Agent": "MotionKeys"})
    with urllib.request.urlopen(request, timeout=60) as response, open(part, "wb") as out:
        shutil.copyfileobj(response, out)
    os.replace(part, dest)
    return dest


def _visible(point, floor: float) -> bool:
    visibility = getattr(point, "visibility", None)
    if visibility is None:
        return True
    return visibility >= floor


def _read_sample(landmarks, brightness: float) -> BodySample:
    points: list[tuple[float, float, float] | None] = []
    for point in landmarks:
        visibility = getattr(point, "visibility", None)
        points.append((point.x, point.y, 1.0 if visibility is None else visibility))

    shoulders = landmarks[_LEFT_SHOULDER], landmarks[_RIGHT_SHOULDER]
    hips = landmarks[_LEFT_HIP], landmarks[_RIGHT_HIP]
    ankles = landmarks[_LEFT_ANKLE], landmarks[_RIGHT_ANKLE]
    if not all(_visible(point, 0.45) for point in (*shoulders, *hips)):
        sample = BodySample(present=False, brightness=brightness, points=points)
        sample.warning = warning_for(sample)
        return sample

    hip_x = (hips[0].x + hips[1].x) / 2
    hip_y = (hips[0].y + hips[1].y) / 2
    shoulder_y = (shoulders[0].y + shoulders[1].y) / 2
    shoulder_width = abs(shoulders[0].x - shoulders[1].x)
    torso = max(0.01, hip_y - shoulder_y)
    visible_ankles = [point for point in ankles if _visible(point, 0.35)]
    if visible_ankles:
        ankle_y = sum(point.y for point in visible_ankles) / len(visible_ankles)
        body = max(0.05, ankle_y - shoulder_y)
    else:
        body = torso / 0.38
    sample = BodySample(
        present=True,
        hip_x=hip_x,
        hip_y=hip_y,
        shoulder_y=shoulder_y,
        nose_y=landmarks[_NOSE].y,
        torso=torso,
        body=body,
        shoulder_width=shoulder_width,
        ankles_visible=all(_visible(point, 0.35) for point in ankles),
        brightness=brightness,
        points=points,
    )
    sample.warning = warning_for(sample)
    return sample


class PoseWorker(QThread):
    packet = Signal(object, object)
    failed = Signal(str)
    status = Signal(str)

    def run(self) -> None:
        cap = None
        landmarker = None
        try:
            model = ensure_model(self.status.emit)
            landmarker = self._landmarker(model)
            cap = self._open_camera()
            if cap is None:
                self.failed.emit("No camera was found.")
                return
            started = time.perf_counter()
            last_ts = 0
            while not self.isInterruptionRequested():
                ok, frame = cap.read()
                if not ok:
                    self.failed.emit("The camera stopped sending frames.")
                    return
                frame = cv2.flip(frame, 1)
                frame = cv2.resize(frame, (640, 480))
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb = np.ascontiguousarray(rgb)
                brightness = float(rgb.mean())
                timestamp = max(last_ts + 1, int((time.perf_counter() - started) * 1000))
                last_ts = timestamp
                image = self._mp_image(rgb)
                result = landmarker.detect_for_video(image, timestamp)
                if not result.pose_landmarks:
                    sample = BodySample(present=False, brightness=brightness)
                    sample.warning = warning_for(sample)
                else:
                    sample = _read_sample(result.pose_landmarks[0], brightness)
                qimage = QImage(
                    rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format_RGB888
                ).copy()
                self.packet.emit(sample, qimage)
        except Exception as exc:
            self.failed.emit(str(exc))
        finally:
            if landmarker is not None:
                landmarker.close()
            if cap is not None:
                cap.release()

    def _landmarker(self, model: str):
        import mediapipe as mp
        from mediapipe.tasks.python.core.base_options import BaseOptions
        from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions
        from mediapipe.tasks.python.vision.core.vision_task_running_mode import (
            VisionTaskRunningMode,
        )

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return PoseLandmarker.create_from_options(options)

    def _mp_image(self, rgb):
        import mediapipe as mp

        return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    def _open_camera(self):
        for index in (0, 1):
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
            cap.release()
        return None
