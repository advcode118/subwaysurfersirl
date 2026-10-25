"""Saved calibration for one person and one camera setup."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass


def app_dir() -> str:
    base = os.environ.get("APPDATA") or os.path.expanduser("~")
    path = os.path.join(base, "MotionKeys")
    os.makedirs(path, exist_ok=True)
    return path


def profile_path() -> str:
    return os.path.join(app_dir(), "profile.json")


def model_path() -> str:
    return os.path.join(app_dir(), "pose_landmarker_lite.task")


@dataclass
class Profile:
    center_x: float
    hip_y: float
    nose_y: float
    torso: float
    shoulder_width: float
    jump_rise: float
    duck_drop: float
    duck_squeeze: float
    zone_half: float
    jump_scale: float = 1.0
    duck_scale: float = 1.0
    zone_scale: float = 1.0
    jump_line: float = 0.42
    crouch_line: float = 0.74
    void_speed: float = 1.0
    jump_percent: float = 0.45

    def lines(self) -> tuple[float, float]:
        half = self.zone_half * self.zone_scale
        left = min(max(self.center_x - half, 0.04), 0.84)
        right = max(min(self.center_x + half, 0.96), left + 0.08)
        return left, right

    def set_lines(self, left: float, right: float) -> None:
        left = min(max(left, 0.04), 0.84)
        right = max(min(right, 0.96), left + 0.08)
        self.center_x = (left + right) / 2
        self.zone_half = (right - left) / 2
        self.zone_scale = 1.0

    def jump_threshold(self) -> float:
        return max(0.02, self.jump_rise * 0.55 / self.jump_scale)

    def duck_drop_threshold(self) -> float:
        return max(0.015, self.duck_drop * 0.55 / self.duck_scale)

    def duck_squeeze_threshold(self) -> float:
        return max(0.06, self.duck_squeeze * 0.55 / self.duck_scale)


def save_profile(profile: Profile) -> None:
    with open(profile_path(), "w", encoding="utf-8") as handle:
        json.dump(asdict(profile), handle, indent=2)


def load_profile() -> Profile | None:
    path = profile_path()
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        fields = Profile.__dataclass_fields__
        kwargs = {key: data[key] for key in fields if key in data}
        return Profile(**kwargs)
    except (OSError, json.JSONDecodeError, TypeError, KeyError):
        return None
