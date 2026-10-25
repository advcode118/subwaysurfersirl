"""One frame of body measurements, already mirrored to match the preview."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BodySample:
    present: bool
    hip_x: float = 0.5
    hip_y: float = 0.5
    shoulder_y: float = 0.3
    nose_y: float = 0.3
    torso: float = 0.25
    body: float = 0.6
    shoulder_width: float = 0.2
    ankles_visible: bool = False
    brightness: float = 128.0
    warning: str = ""
    points: list[tuple[float, float, float] | None] | None = None


def warning_for(sample: BodySample) -> str:
    if not sample.present:
        return "Step into the camera view."
    if not sample.ankles_visible:
        return "Step back so your whole body is in frame."
    if sample.brightness < 45:
        return "The room is too dark."
    if sample.shoulder_width > 0.48:
        return "Step back. You fill the frame."
    if sample.shoulder_width < 0.09:
        return "Step closer so your body is easier to read."
    return ""
