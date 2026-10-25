"""Walk a person through standing, one jump, and one crouch."""

from __future__ import annotations

from statistics import pstdev

from motionlane.pose.sample import BodySample
from motionlane.profile import Profile


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


class Wizard:
    def __init__(self):
        self.step = "frame"
        self.message = "Stand where the camera sees your whole body."
        self.profile: Profile | None = None
        self._ok_frames = 0
        self._neutral: list[BodySample] = []
        self._center: dict[str, float] | None = None
        self._peak_rise = 0.0
        self._jump_latched = False
        self._peak_drop = 0.0
        self._peak_squeeze = 0.0
        self._duck_latched = False

    def lines(self) -> tuple[float, float] | None:
        if self.profile is not None:
            return self.profile.lines()
        if self._center is None:
            return None
        half = self._center["zone_half"]
        center = self._center["center_x"]
        left = min(max(center - half, 0.04), 0.72)
        right = max(min(center + half, 0.96), left + 0.12)
        return left, right

    def push(self, sample: BodySample) -> None:
        if self.step == "done":
            return
        if self.step == "frame":
            self._frame(sample)
        elif self.step == "neutral":
            self._hold_still(sample)
        elif self.step == "jump":
            self._jump(sample)
        elif self.step == "duck":
            self._duck(sample)

    def _frame(self, sample: BodySample) -> None:
        if sample.warning:
            self._ok_frames = 0
            self.message = sample.warning
            return
        self._ok_frames += 1
        self.message = "Stand where the camera sees your whole body."
        if self._ok_frames >= 12:
            self.step = "neutral"
            self._neutral.clear()
            self.message = "Stand in the middle and hold still."

    def _hold_still(self, sample: BodySample) -> None:
        if sample.warning or not sample.present:
            self._neutral.clear()
            self.message = sample.warning or "Stand in the middle and hold still."
            return
        self._neutral.append(sample)
        if len(self._neutral) > 24:
            self._neutral.pop(0)
        if len(self._neutral) < 18:
            self.message = "Stand in the middle and hold still."
            return
        xs = [item.hip_x for item in self._neutral]
        ys = [item.hip_y for item in self._neutral]
        if pstdev(xs) > 0.018 or pstdev(ys) > 0.018:
            self.message = "Hold still in the middle."
            return
        shoulder = _mean([item.shoulder_width for item in self._neutral])
        self._center = {
            "center_x": _mean(xs),
            "hip_y": _mean(ys),
            "nose_y": _mean([item.nose_y for item in self._neutral]),
            "torso": max(0.05, _mean([item.torso for item in self._neutral])),
            "shoulder_width": shoulder,
            "zone_half": min(0.26, max(0.11, shoulder * 1.15)),
        }
        self.step = "jump"
        self._peak_rise = 0.0
        self._jump_latched = False
        self.message = "Jump once, about as high as you will while playing."

    def _jump(self, sample: BodySample) -> None:
        if not sample.present or self._center is None:
            self.message = "Step back into frame, then jump once."
            return
        rise = self._center["hip_y"] - sample.hip_y
        if rise > self._peak_rise:
            self._peak_rise = rise
        if self._peak_rise >= 0.04:
            self._jump_latched = True
        if self._jump_latched and rise < self._peak_rise * 0.45:
            self.step = "duck"
            self._peak_drop = 0.0
            self._peak_squeeze = 0.0
            self._duck_latched = False
            self.message = "Crouch once, then stand back up."
            return
        self.message = "Jump once, about as high as you will while playing."

    def _duck(self, sample: BodySample) -> None:
        if not sample.present or self._center is None:
            self.message = "Step back into frame, then crouch once."
            return
        drop = sample.nose_y - self._center["nose_y"]
        squeeze = 1.0 - (sample.torso / self._center["torso"])
        self._peak_drop = max(self._peak_drop, drop)
        self._peak_squeeze = max(self._peak_squeeze, squeeze)
        if self._peak_drop >= 0.03 or self._peak_squeeze >= 0.1:
            self._duck_latched = True
        settled = drop < max(0.015, self._peak_drop * 0.4) and squeeze < max(
            0.04, self._peak_squeeze * 0.5
        )
        if self._duck_latched and settled:
            center = self._center
            self.profile = Profile(
                center_x=center["center_x"],
                hip_y=center["hip_y"],
                nose_y=center["nose_y"],
                torso=center["torso"],
                shoulder_width=center["shoulder_width"],
                jump_rise=self._peak_rise,
                duck_drop=max(self._peak_drop, 0.03),
                duck_squeeze=max(self._peak_squeeze, 0.1),
                zone_half=center["zone_half"],
            )
            self.step = "done"
            self.message = "Step across a line to change lane. Use camera when you are ready."
            return
        self.message = "Crouch once, then stand back up."
