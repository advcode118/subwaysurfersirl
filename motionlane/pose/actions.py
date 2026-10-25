"""Turn a calibrated body into lane, jump, and duck."""

from __future__ import annotations

import math
from dataclasses import dataclass

from motionlane.pose.sample import BodySample
from motionlane.profile import Profile

_ZONE_NAMES = {-1: "Left", 0: "Center", 1: "Right"}


@dataclass
class Controls:
    zone: int | None
    zone_changed: bool
    jump: bool
    duck: bool
    label: str


def _ema(previous: float | None, value: float, alpha: float) -> float:
    if previous is None:
        return value
    return alpha * value + (1.0 - alpha) * previous


class ActionMapper:
    """Hips pick one of three camera areas. Crossing a line switches once."""

    def __init__(self, profile: Profile):
        self.profile = profile
        self.zone = 0
        self.ready = False
        self._x: float | None = None
        self._hip_y: float | None = None
        self._nose_y: float | None = None
        self._torso: float | None = None
        self._stand_hip = profile.hip_y
        self._stand_nose = profile.nose_y
        self._stand_torso = max(0.05, profile.torso)
        self._jump_ready = True

    def update(self, sample: BodySample) -> Controls:
        if not sample.present:
            label = sample.warning or "Step into the camera view."
            return Controls(None, False, False, False, label)

        self._x = _ema(self._x, sample.hip_x, 0.42)
        self._hip_y = _ema(self._hip_y, sample.hip_y, 0.45)
        self._nose_y = _ema(self._nose_y, sample.nose_y, 0.45)
        self._torso = _ema(self._torso, sample.torso, 0.35)

        rise = self._stand_hip - self._hip_y
        jump_line = self.profile.jump_threshold()
        if rise < jump_line * 0.4:
            self._stand_hip = 0.08 * self._hip_y + 0.92 * self._stand_hip
            rise = self._stand_hip - self._hip_y

        jump = False
        if self._jump_ready and rise >= jump_line:
            jump = True
            self._jump_ready = False
        elif rise < jump_line * 0.45:
            self._jump_ready = True

        drop = self._nose_y - self._stand_nose
        squeeze = 1.0 - (self._torso / self._stand_torso) if self._stand_torso else 0.0
        duck = (
            drop >= self.profile.duck_drop_threshold()
            or squeeze >= self.profile.duck_squeeze_threshold()
        )
        if not duck:
            self._stand_nose = 0.05 * self._nose_y + 0.95 * self._stand_nose
            self._stand_torso = 0.05 * self._torso + 0.95 * self._stand_torso

        zone = self._zone_at(self._x)
        changed = self.ready and zone != self.zone
        self.ready = True
        self.zone = zone

        if jump:
            label = "Jump"
        elif duck:
            label = "Duck"
        else:
            label = _ZONE_NAMES[zone]
        return Controls(zone, changed, jump, duck, label)

    def _zone_at(self, x: float) -> int:
        left, right = self.profile.lines()
        band = max(0.012, self.profile.shoulder_width * 0.08)
        if not self.ready:
            if x < left:
                return -1
            if x > right:
                return 1
            return 0
        if self.zone == 0:
            if x < left:
                return -1
            if x > right:
                return 1
            return 0
        if self.zone < 0:
            if x > left + band:
                return 1 if x > right else 0
            return -1
        if x < right - band:
            return -1 if x < left else 0
        return 1


def _point(points, index: int, floor: float = 0.4):
    if not points or index >= len(points):
        return None
    point = points[index]
    if point is None or point[2] < floor:
        return None
    return point[0], point[1]


def _joint_angle(hip, knee, ankle) -> float | None:
    ax, ay = hip[0] - knee[0], hip[1] - knee[1]
    bx, by = ankle[0] - knee[0], ankle[1] - knee[1]
    mag_a = math.hypot(ax, ay)
    mag_b = math.hypot(bx, by)
    if mag_a < 1e-4 or mag_b < 1e-4:
        return None
    cos = max(-1.0, min(1.0, (ax * bx + ay * by) / (mag_a * mag_b)))
    return math.degrees(math.acos(cos))


def knee_angle(points) -> float | None:
    angles = []
    for hip_i, knee_i, ankle_i in ((23, 25, 27), (24, 26, 28)):
        hip = _point(points, hip_i)
        knee = _point(points, knee_i, 0.35)
        ankle = _point(points, ankle_i, 0.3)
        if hip and knee and ankle:
            angle = _joint_angle(hip, knee, ankle)
            if angle is not None:
                angles.append(angle)
    if not angles:
        return None
    return sum(angles) / len(angles)


def _cross(origin, end, point) -> float:
    return (end[0] - origin[0]) * (point[1] - origin[1]) - (end[1] - origin[1]) * (point[0] - origin[0])


def forearms_crossed(points) -> bool:
    left_elbow = _point(points, 13, 0.45)
    left_wrist = _point(points, 15, 0.45)
    right_elbow = _point(points, 14, 0.45)
    right_wrist = _point(points, 16, 0.45)
    if not all((left_elbow, left_wrist, right_elbow, right_wrist)):
        return False
    left = _cross(left_elbow, left_wrist, right_elbow) * _cross(left_elbow, left_wrist, right_wrist)
    right = _cross(right_elbow, right_wrist, left_elbow) * _cross(right_elbow, right_wrist, left_wrist)
    return left < 0 and right < 0


class Stage:
    """Lanes come from the lines. A jump is a rise of the hips as a share of the torso. A squat is a drop of the shoulders."""

    def __init__(self):
        self.x = 0.5
        self.y = 0.55
        self.shoulder_y = 0.35
        self.zone = 0
        self.ready = False
        self.jump_ratio = 0.45
        self._jump_armed = True
        self._have_body = False
        self._stand_hip = 0.55
        self._stand_shoulder = 0.35
        self._stand_torso = 0.25
        self._squat = False

    def update(self, sample: BodySample, lines: tuple[float, float] | None) -> Controls:
        if not sample.present or lines is None:
            self._have_body = False
            self._jump_armed = True
            label = sample.warning or "Step into the camera view."
            return Controls(None, False, False, False, label)

        self.x = sample.hip_x if not self._have_body else _ema(self.x, sample.hip_x, 0.55)
        self.y = sample.hip_y if not self._have_body else _ema(self.y, sample.hip_y, 0.55)
        self.shoulder_y = sample.shoulder_y if not self._have_body else _ema(self.shoulder_y, sample.shoulder_y, 0.55)
        if not self._have_body:
            self._stand_hip = self.y
            self._stand_shoulder = self.shoulder_y
            self._stand_torso = max(0.05, sample.torso)
        self._have_body = True

        zone = self._zone_at(self.x, lines)
        changed = self.ready and zone != self.zone
        self.ready = True
        self.zone = zone

        torso = max(0.05, self._stand_torso)
        rise = (self._stand_hip - self.y) / torso
        drop = (self.shoulder_y - self._stand_shoulder) / torso
        ratio = min(0.8, max(0.2, self.jump_ratio))
        if drop >= 0.32:
            self._squat = True
        elif drop < 0.14:
            self._squat = False
        if not self._squat and rise < ratio * 0.5 and drop < 0.12:
            self._stand_hip = 0.1 * self.y + 0.9 * self._stand_hip
            self._stand_shoulder = 0.1 * self.shoulder_y + 0.9 * self._stand_shoulder
            self._stand_torso = 0.1 * max(0.05, sample.torso) + 0.9 * self._stand_torso
            torso = max(0.05, self._stand_torso)
            rise = (self._stand_hip - self.y) / torso
            drop = (self.shoulder_y - self._stand_shoulder) / torso

        jump = self._jump_armed and rise >= ratio
        if jump or rise >= ratio:
            self._jump_armed = False
            self._squat = False
        elif rise < ratio * 0.4:
            self._jump_armed = True

        duck = self._squat and not jump
        if jump:
            label = "Jump"
        elif duck:
            label = "Squat"
        else:
            label = _ZONE_NAMES[zone]
        return Controls(zone, changed, jump, duck, label)

    def _zone_at(self, x: float, lines: tuple[float, float]) -> int:
        left, right = lines
        band = 0.015
        if not self.ready:
            if x < left:
                return -1
            if x > right:
                return 1
            return 0
        if self.zone == 0:
            if x < left:
                return -1
            if x > right:
                return 1
            return 0
        if self.zone < 0:
            return -1 if x < left + band else (1 if x > right else 0)
        return 1 if x > right - band else (-1 if x < left else 0)
