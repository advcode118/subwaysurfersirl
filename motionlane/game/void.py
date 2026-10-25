"""A quiet rhythm field. Frames come toward you, and you pass through them."""

from __future__ import annotations

import math
from dataclasses import dataclass

from PySide6.QtCore import Qt, QPointF, QRectF
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

PLAYER_Z = 0.78
BEAT = 1.45
TRAVEL = 3.5
PATTERN = (
    (0, "open"),
    (1, "open"),
    (-1, "open"),
    (0, "open"),
    (1, "high"),
    (-1, "open"),
    (0, "low"),
    (-1, "open"),
    (1, "open"),
    (0, "open"),
)


@dataclass
class Gate:
    lane: int
    kind: str
    z: float
    judged: bool = False
    passed: bool = False
    flash: float = 0.0


class VoidGame:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.lane = 0
        self.lane_pos = 0.0
        self.jump_t = -1.0
        self.ducking = False
        self.gates: list[Gate] = []
        self.spawn_in = 1.1
        self.pattern = 0
        self.score = 0
        self.combo = 0
        self.lives = 3
        self.over = False
        self.time = 0.0
        self.ripples: list[float] = []
        self.speed = 1.0

    def jump(self) -> None:
        if self.jump_t < 0 and not self.ducking:
            self.jump_t = 0.0

    def set_duck(self, ducking: bool) -> None:
        self.ducking = False if self.jump_t >= 0 else ducking

    def update(self, dt: float, playing: bool) -> None:
        dt = min(max(dt, 0.0), 0.05)
        self.time += dt
        self.lane_pos += (self.lane - self.lane_pos) * min(1.0, dt * 9.0)
        if self.jump_t >= 0:
            self.jump_t += dt
            if self.jump_t > 0.52:
                self.jump_t = -1.0
        self.ripples = [age + dt for age in self.ripples if age < 0.7]
        if not playing or self.over:
            return
        self.spawn_in -= dt
        if self.spawn_in <= 0:
            self.spawn_in += BEAT / self.speed
            lane, kind = PATTERN[self.pattern % len(PATTERN)]
            self.pattern += 1
            self.gates.append(Gate(lane, kind, 0.0))
        step = dt / (TRAVEL / self.speed)
        for gate in self.gates:
            gate.z += step
            if gate.flash > 0:
                gate.flash = max(0.0, gate.flash - dt)
            if gate.judged:
                continue
            if abs(gate.z - PLAYER_Z) <= 0.075 and self._matches(gate):
                self._pass(gate)
            elif gate.z > PLAYER_Z + 0.075:
                gate.judged = True
                self.combo = 0
                self.lives -= 1
                if self.lives <= 0:
                    self.lives = 0
                    self.over = True
                    self.gates.clear()
                    return
        self.gates = [gate for gate in self.gates if gate.z < 1.2]

    def _matches(self, gate: Gate) -> bool:
        if self.lane != gate.lane:
            return False
        if gate.kind == "high":
            return self.jump_t >= 0
        if gate.kind == "low":
            return self.ducking
        return True

    def _pass(self, gate: Gate) -> None:
        gate.judged = True
        gate.passed = True
        gate.flash = 0.45
        self.combo += 1
        self.score += 1
        self.ripples.append(0.0)

    @property
    def pulse(self) -> float:
        return (self.time / (BEAT / self.speed)) % 1.0


def project(width: float, height: float, z: float, lane: float) -> tuple[float, float, float]:
    depth = z**0.86
    scale = 0.16 + 0.84 * depth
    x = width * 0.5 + lane * width * 0.24 * scale
    y = height * 0.2 + height * 0.64 * depth
    return x, y, scale


class VoidView(QWidget):
    def __init__(self, game: VoidGame):
        super().__init__()
        self.game = game
        self.started = False
        self.setMinimumWidth(420)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#07080c"))
        width = float(self.width())
        height = float(self.height())
        self._lines(painter, width, height)
        self._pulse(painter, width, height)
        for gate in sorted(self.game.gates, key=lambda item: item.z):
            self._gate(painter, width, height, gate)
        self._figure(painter, width, height)
        self._ripples(painter, width, height)
        painter.setPen(QColor("#f4efe4"))
        painter.setFont(QFont("Bahnschrift", 18))
        painter.drawText(24, 36, str(self.game.score))
        self._lives(painter)
        if self.game.combo > 1 and not self.game.over:
            painter.setPen(QColor("#e7d7a8"))
            painter.setFont(QFont("Bahnschrift", 13))
            painter.drawText(24, 58, str(self.game.combo))
        if self.game.over:
            painter.setPen(QColor("#f4efe4"))
            painter.setFont(QFont("Bahnschrift", 28))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, f"You scored {self.game.score}")
            painter.setFont(QFont("Segoe UI", 14))
            painter.drawText(
                self.rect().adjusted(0, 64, 0, 0),
                Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter,
                "Cross your forearms and hold them",
            )
        elif not self.started:
            painter.setPen(QColor("#f4efe4"))
            painter.setFont(QFont("Bahnschrift", 20))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Press Start")
        painter.end()

    def _lives(self, painter: QPainter) -> None:
        for index in range(3):
            color = QColor("#f4efe4") if index < self.game.lives else QColor("#3a342c")
            painter.fillRect(self.width() - 78 + index * 18, 24, 10, 10, color)

    def _lines(self, painter: QPainter, width: float, height: float) -> None:
        bright = self.game.pulse < 0.12
        color = QColor("#f4efe4") if bright else QColor("#e7d7a8")
        color.setAlpha(230 if bright else 150)
        pen = QPen(color)
        pen.setWidthF(1.6 if bright else 1.15)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        for lane in (-1, 0, 1):
            far = project(width, height, 0.02, lane)
            near = project(width, height, 1.0, lane)
            painter.drawLine(QPointF(far[0], far[1]), QPointF(near[0], near[1]))

    def _pulse(self, painter: QPainter, width: float, height: float) -> None:
        origin = project(width, height, 0.02, 0)
        radius = 8 + self.game.pulse * 28
        color = QColor("#e7d7a8")
        color.setAlpha(int(90 * (1.0 - self.game.pulse)))
        pen = QPen(color)
        pen.setWidthF(1.2)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(origin[0], origin[1]), radius, radius * 0.45)

    def _gate(self, painter: QPainter, width: float, height: float, gate: Gate) -> None:
        x, y, scale = project(width, height, gate.z, gate.lane)
        if gate.passed and gate.flash > 0:
            color = QColor("#f7f4ee")
        elif gate.judged:
            color = QColor("#5c564c")
        elif gate.kind == "low":
            color = QColor("#e7d7a8")
        else:
            color = QColor("#f4efe4")
        pen = QPen(color)
        pen.setWidthF(max(1.4, 2.6 * scale))
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        frame_w = max(28.0, 70 * scale)
        frame_h = max(40.0, 100 * scale)
        if gate.kind == "high":
            painter.drawRect(int(x - frame_w / 2), int(y - frame_h), int(frame_w), int(frame_h * 0.5))
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(int(x - frame_w / 2), int(y - frame_h * 0.38), int(frame_w), max(5, int(frame_h * 0.1)))
            self._arrow(painter, x, y - frame_h * 0.78, max(16.0, 30 * scale), True, color)
            self._gate_word(painter, x, y - frame_h - 4, "Jump", color, scale)
        elif gate.kind == "low":
            painter.drawRect(int(x - frame_w / 2), int(y - frame_h * 0.46), int(frame_w), int(frame_h * 0.46))
            painter.setBrush(color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(int(x - frame_w / 2), int(y - frame_h), int(frame_w), max(5, int(frame_h * 0.1)))
            self._arrow(painter, x, y - frame_h * 0.22, max(16.0, 30 * scale), False, color)
            self._gate_word(painter, x, y - frame_h - 4, "Squat", color, scale)
        else:
            painter.drawRect(int(x - frame_w / 2), int(y - frame_h), int(frame_w), int(frame_h))

    def _arrow(self, painter: QPainter, x: float, y: float, size: float, up: bool, color: QColor) -> None:
        painter.setBrush(color)
        painter.setPen(Qt.PenStyle.NoPen)
        tip = y - size if up else y + size
        base = y + size * 0.45 if up else y - size * 0.45
        painter.drawPolygon(
            [QPointF(x, tip), QPointF(x - size * 0.72, base), QPointF(x + size * 0.72, base)]
        )

    def _gate_word(self, painter: QPainter, x: float, y: float, word: str, color: QColor, scale: float) -> None:
        if scale < 0.38:
            return
        painter.setPen(color)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setFont(QFont("Bahnschrift", max(12, int(15 * scale))))
        painter.drawText(QRectF(x - 46, y - 18, 92, 20), Qt.AlignmentFlag.AlignCenter, word)

    def _figure(self, painter: QPainter, width: float, height: float) -> None:
        game = self.game
        lift = 0.0
        if game.jump_t >= 0:
            lift = 70 * math.sin(math.pi * min(game.jump_t, 0.52) / 0.52)
        x, y, scale = project(width, height, PLAYER_Z, game.lane_pos)
        painter.save()
        painter.translate(x, y - lift * scale)
        painter.rotate((game.lane - game.lane_pos) * 16)
        painter.scale(scale * (0.72 if game.ducking else 1.15), scale * (0.72 if game.ducking else 1.15))
        pen = QPen(QColor("#f4efe4"))
        pen.setWidthF(1.7)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(0, -62), 6, 6)
        painter.drawLine(QPointF(0, -56), QPointF(0, -20))
        painter.drawLine(QPointF(-14, -46), QPointF(14, -46))
        painter.drawLine(QPointF(-9, -22), QPointF(9, -22))
        painter.drawLine(QPointF(-9, -22), QPointF(-11, 0))
        painter.drawLine(QPointF(9, -22), QPointF(11, 0))
        painter.restore()

    def _ripples(self, painter: QPainter, width: float, height: float) -> None:
        x, y, _scale = project(width, height, PLAYER_Z, self.game.lane_pos)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for age in self.game.ripples:
            color = QColor("#f4efe4")
            color.setAlpha(int(160 * (1.0 - age / 0.7)))
            pen = QPen(color)
            pen.setWidthF(1.3)
            painter.setPen(pen)
            radius = 12 + age * 90
            painter.drawEllipse(QPointF(x, y - 30), radius, radius * 0.55)
