"""One window: camera areas on the left, practice track on the right."""

from __future__ import annotations

import sys
import time

from PySide6.QtCore import Qt, QRect, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from motionlane.game.runner import Runner, lane_centers, lane_width, obstacle_box
from motionlane.game.void import VoidGame, VoidView
from motionlane.pose.actions import ActionMapper, Stage, forearms_crossed
from motionlane.pose.calibrate import Wizard
from motionlane.pose.sample import BodySample
from motionlane.pose.tracker import PoseWorker
from motionlane.profile import load_profile, save_profile

BONES = (
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
    (0, 11),
    (0, 12),
)

INK = QColor("#1c2430")
PAPER = QColor("#e7eef6")
QUIET = QColor("#b7c3d1")
AMBER = QColor("#f2c14e")
LANE = QColor("#2a3648")
BLOCK = QColor("#d9785a")
BAR = QColor("#7ea2c4")


def fit_rect(image_w: int, image_h: int, bounds: QRect) -> QRect:
    if image_w <= 0 or image_h <= 0:
        return bounds
    scale = min(bounds.width() / image_w, bounds.height() / image_h)
    width = int(image_w * scale)
    height = int(image_h * scale)
    x = bounds.x() + (bounds.width() - width) // 2
    y = bounds.y() + (bounds.height() - height) // 2
    return QRect(x, y, width, height)


class CameraView(QWidget):
    lines_changed = Signal(float, float)

    def __init__(self):
        super().__init__()
        self.image = None
        self.sample: BodySample | None = None
        self.marker: tuple[float, float] | None = None
        self.lines: tuple[float, float] | None = (0.33, 0.67)
        self.active: int | None = None
        self.dragging: str | None = None
        self._target = QRect()
        self.setMinimumSize(360, 270)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.ArrowCursor)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), INK)
        if self.image is None:
            painter.setPen(PAPER)
            painter.setFont(QFont("Segoe UI", 12))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "Waiting for the camera")
            painter.end()
            return
        self._target = fit_rect(self.image.width(), self.image.height(), self.rect())
        painter.drawImage(self._target, self.image)
        if self.lines is not None:
            self._draw_zones(painter, self._target)
        if self.sample is not None and self.sample.points:
            self._draw_body(painter, self._target)
        painter.end()

    def _line_x(self, norm: float) -> int:
        return int(self._target.x() + norm * self._target.width())

    def _norm_x(self, pixel: float) -> float | None:
        if self._target.width() <= 0:
            return None
        return (pixel - self._target.x()) / self._target.width()

    def _near_line(self, x: float) -> str | None:
        if self.lines is None or self._target.width() <= 0:
            return None
        left = abs(x - self._line_x(self.lines[0]))
        right = abs(x - self._line_x(self.lines[1]))
        if left <= 16 and left <= right:
            return "left"
        if right <= 16:
            return "right"
        return None

    def mousePressEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        which = self._near_line(event.position().x())
        if which is None:
            return
        self.dragging = which
        self._move_lane(which, event.position().x())

    def mouseMoveEvent(self, event) -> None:
        which = self.dragging or self._near_line(event.position().x())
        self.setCursor(Qt.CursorShape.SizeHorCursor if which else Qt.CursorShape.ArrowCursor)
        if self.dragging:
            self._move_lane(self.dragging, event.position().x())

    def mouseReleaseEvent(self, event) -> None:
        if self.dragging:
            self._move_lane(self.dragging, event.position().x())
        self.dragging = None

    def _move_lane(self, which: str, pixel: float) -> None:
        norm = self._norm_x(pixel)
        if norm is None or self.lines is None:
            return
        left, right = self.lines
        if which == "left":
            left = min(max(norm, 0.04), right - 0.08)
        else:
            right = max(min(norm, 0.96), left + 0.08)
        self.lines = (left, right)
        self.lines_changed.emit(left, right)
        self.update()

    def _draw_zones(self, painter: QPainter, target: QRect) -> None:
        left, right = self.lines
        spans = ((0.0, left, -1, "Left"), (left, right, 0, "Center"), (right, 1.0, 1, "Right"))
        quiet = {
            -1: QColor(90, 140, 190, 40),
            0: QColor(242, 193, 78, 28),
            1: QColor(217, 120, 90, 40),
        }
        strong = {
            -1: QColor(90, 140, 190, 90),
            0: QColor(242, 193, 78, 70),
            1: QColor(217, 120, 90, 90),
        }
        painter.setFont(QFont("Bahnschrift", 13))
        for start, end, zone, name in spans:
            rect = QRect(
                int(target.x() + start * target.width()),
                target.y(),
                max(1, int((end - start) * target.width())),
                target.height(),
            )
            painter.fillRect(rect, strong[zone] if zone == self.active else quiet[zone])
            painter.setPen(PAPER)
            painter.drawText(rect.adjusted(0, 8, 0, 0), Qt.AlignmentFlag.AlignHCenter, name)
        for index, line in enumerate((left, right)):
            x = self._line_x(line)
            pen = QPen(AMBER if self.dragging == ("left", "right")[index] else QColor("#f4f7fb"))
            pen.setWidth(3 if self.dragging else 2)
            painter.setPen(pen)
            painter.drawLine(x, target.top(), x, target.bottom())
            painter.setBrush(AMBER)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(x - 6, target.center().y() - 18, 12, 36, 4, 4)
        if self.marker is not None:
            painter.setBrush(AMBER)
            painter.setPen(Qt.PenStyle.NoPen)
            x = target.x() + self.marker[0] * target.width()
            y = target.y() + self.marker[1] * target.height()
            painter.drawEllipse(int(x - 7), int(y - 7), 14, 14)

    def _draw_body(self, painter: QPainter, target: QRect) -> None:
        points = self.sample.points
        pen = QPen(QColor("#d7e6f2"))
        pen.setWidth(2)
        painter.setPen(pen)

        def at(index: int):
            point = points[index]
            if point is None or point[2] < 0.4:
                return None
            return (
                target.x() + point[0] * target.width(),
                target.y() + point[1] * target.height(),
            )

        for start, end in BONES:
            a = at(start)
            b = at(end)
            if a and b:
                painter.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))


class TrackView(QWidget):
    def __init__(self, runner: Runner):
        super().__init__()
        self.runner = runner
        self.started = False
        self.setMinimumWidth(420)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), INK)
        width = self.width()
        height = self.height()
        centers = lane_centers(width)
        slab = lane_width(width)
        for lane, center in centers.items():
            rect = QRect(int(center - slab / 2), 28, int(slab), height - 52)
            color = QColor("#33445c") if lane == self.runner.lane else LANE
            painter.fillRect(rect, color)
            painter.setPen(QPen(QColor("#41536b")))
            offset = int(self.runner.scroll) % 46
            y = 40 - offset
            while y < height - 30:
                painter.drawLine(int(center), y, int(center), y + 16)
                y += 46

        pose = self.runner.pose(height)
        foot = pose["foot"]
        for obstacle in self.runner.obstacles:
            box = obstacle_box(obstacle, width)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(BLOCK if obstacle.kind == "block" else BAR)
            painter.drawRoundedRect(QRect(int(box[0]), int(box[1]), int(box[2]), int(box[3])), 4, 4)

        center = centers[self.runner.lane]
        body = QRect(
            int(center - 18),
            int(pose["top"]),
            36,
            int(pose["body_h"]),
        )
        painter.setBrush(AMBER)
        painter.drawRoundedRect(body, 6, 6)

        painter.setPen(PAPER)
        painter.setFont(QFont("Bahnschrift", 18))
        painter.drawText(24, 36, str(int(self.runner.score)))
        if not self.runner.alive:
            self._center_text(painter, "That one hit.")
        elif not self.started:
            self._center_text(painter, "Press Start")
        painter.end()

    def _center_text(self, painter: QPainter, text: str) -> None:
        painter.setPen(PAPER)
        painter.setFont(QFont("Bahnschrift", 22))
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, text)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Motion Keys")
        self.resize(1180, 720)
        self.profile = load_profile()
        self.wizard = None if self.profile else Wizard()
        self.mapper = ActionMapper(self.profile) if self.profile else None
        self.runner = Runner()
        self.void = VoidGame()
        self.mode = "void"
        self.stage = Stage()
        self.running = False
        self.camera_on = True
        self.duck_key = False
        self.controls = None
        self.sample: BodySample | None = None
        self.pending_lines: tuple[float, float] | None = None
        self.crossed = False
        self._cross_time = 0.0
        self._last = time.perf_counter()

        self.camera = CameraView()
        if self.profile is not None:
            self.camera.lines = self.profile.lines()
        self.camera.lines_changed.connect(self.on_lines_moved)
        self.zone_word = QLabel("Camera")
        self.zone_word.setFont(QFont("Bahnschrift", 26))
        self.instruction = QLabel(self._instruction_text())
        self.instruction.setWordWrap(True)
        self.instruction.setFont(QFont("Segoe UI", 13))
        self.line_hint = QLabel("Drag the lane lines. Jump by leaving the ground. Squat by lowering your shoulders.")
        self.line_hint.setFont(QFont("Segoe UI", 11))
        self.line_hint.setStyleSheet("color: #b7c3d1;")
        self.start_button = QPushButton("Start")
        self.mode_button = QPushButton("Practice")
        self.camera_button = QPushButton("Use camera")
        self.calibrate_button = QPushButton("Calibrate again")
        self.start_button.clicked.connect(self.toggle_run)
        self.mode_button.clicked.connect(self.toggle_mode)
        self.camera_button.clicked.connect(self.toggle_camera)
        self.calibrate_button.clicked.connect(self.recalibrate)

        self.jump_slider = QSlider(Qt.Orientation.Horizontal)
        self.jump_slider.setRange(20, 80)
        self.jump_slider.setValue(self._saved_jump_percent())
        self.stage.jump_ratio = self.jump_slider.value() / 100
        self.jump_label = QLabel(f"Jump {self.jump_slider.value()}%")
        self.jump_label.setFixedWidth(110)
        self.zone_slider = self._slider()
        self.speed_slider = self._slider()
        self.jump_slider.valueChanged.connect(self._set_jump_percent)
        self.zone_slider.valueChanged.connect(lambda value: self._scale("zone_scale", value))
        self.speed_slider.valueChanged.connect(self._set_speed)
        self.sliders = QWidget()
        slider_layout = QVBoxLayout(self.sliders)
        slider_layout.setContentsMargins(0, 8, 0, 0)
        jump_row = QHBoxLayout()
        jump_row.addWidget(self.jump_label)
        jump_row.addWidget(self.jump_slider)
        slider_layout.addLayout(jump_row)
        slider_layout.addLayout(self._slider_row("Middle width", self.zone_slider))
        slider_layout.addLayout(self._slider_row("Speed", self.speed_slider))

        buttons = QHBoxLayout()
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.camera_button)
        buttons.addWidget(self.calibrate_button)

        side = QVBoxLayout()
        side.addWidget(self.camera, 1)
        side.addWidget(self.zone_word)
        side.addWidget(self.instruction)
        side.addWidget(self.line_hint)
        side.addWidget(self.mode_button)
        side.addLayout(buttons)
        side.addWidget(self.sliders)

        self.track = TrackView(self.runner)
        self.void_view = VoidView(self.void)
        self.stack = QStackedWidget()
        self.stack.addWidget(self.void_view)
        self.stack.addWidget(self.track)
        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(18)
        column = QWidget()
        column.setFixedWidth(400)
        column.setLayout(side)
        layout.addWidget(column)
        layout.addWidget(self.stack, 1)
        self.setCentralWidget(root)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._apply_style()
        self._sync_controls()

        self.worker = PoseWorker()
        self.worker.packet.connect(self.on_packet)
        self.worker.failed.connect(self.on_failed)
        self.worker.status.connect(self.on_status)
        self.worker.start()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(16)

    def _slider(self) -> QSlider:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(60, 180)
        slider.setValue(100)
        return slider

    def _slider_row(self, name: str, slider: QSlider) -> QHBoxLayout:
        row = QHBoxLayout()
        label = QLabel(name)
        label.setFixedWidth(110)
        row.addWidget(label)
        row.addWidget(slider)
        return row

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #1c2430; color: #e7eef6; }
            QLabel { background: transparent; }
            QPushButton {
                background: #f2c14e; color: #1c2430; border: none;
                padding: 8px 14px; font-family: Bahnschrift; font-size: 15px;
            }
            QPushButton:disabled { background: #3a4658; color: #93a0b0; }
            QSlider::groove:horizontal { height: 4px; background: #3a4658; }
            QSlider::handle:horizontal {
                background: #f2c14e; width: 14px; margin: -6px 0; border-radius: 7px;
            }
            """
        )

    def _instruction_text(self) -> str:
        if self.wizard is not None:
            return self.wizard.message
        if self.camera_on and self.mode == "void":
            return "Pass through the frame on your line. Jump by leaving the ground. Squat by lowering your shoulders."
        if self.camera_on:
            return "The player follows the yellow dot. Jump by leaving the ground. Squat by lowering your shoulders."
        if self.profile is not None:
            return "Arrow keys play the track. Use camera when you want your body to drive."
        return "Arrow keys play the track. Calibration starts when the camera sees you."

    def _sync_controls(self) -> None:
        ready = self.profile is not None and self.wizard is None
        self.camera_button.setEnabled(True)
        self.sliders.setVisible(True)
        self.zone_slider.setVisible(ready)
        for label in self.sliders.findChildren(QLabel):
            if label.text() == "Middle width":
                label.setVisible(ready)
        self.camera_button.setText("Stop camera" if self.camera_on else "Use camera")
        self.mode_button.setText("Practice" if self.mode == "void" else "Void")
        self.stack.setCurrentWidget(self.void_view if self.mode == "void" else self.track)
        if self.mode == "void" or self.runner.alive:
            self.start_button.setText("Stop" if self.running else "Start")
        elif not self.running:
            self.start_button.setText("Run again")
        else:
            self.start_button.setText("Stop")
        if ready:
            self._show_jump_percent(self._saved_jump_percent())
            self._set_slider(self.zone_slider, self.profile.zone_scale)
            self._set_slider(self.speed_slider, self.profile.void_speed)
            self.void.speed = max(0.6, min(1.8, self.profile.void_speed))
        self.instruction.setText(self._instruction_text())

    def _set_slider(self, slider: QSlider, scale: float) -> None:
        slider.blockSignals(True)
        slider.setValue(int(max(60, min(180, round(scale * 100)))))
        slider.blockSignals(False)

    def _saved_jump_percent(self) -> int:
        ratio = 0.45 if self.profile is None else self.profile.jump_percent
        return int(max(20, min(80, round(ratio * 100))))

    def _show_jump_percent(self, percent: int) -> None:
        percent = int(max(20, min(80, percent)))
        self.jump_slider.blockSignals(True)
        self.jump_slider.setValue(percent)
        self.jump_slider.blockSignals(False)
        self.jump_label.setText(f"Jump {percent}%")
        self.stage.jump_ratio = percent / 100

    def _set_jump_percent(self, value: int) -> None:
        percent = int(max(20, min(80, value)))
        self.jump_label.setText(f"Jump {percent}%")
        self.stage.jump_ratio = percent / 100
        if self.profile is None:
            return
        self.profile.jump_percent = self.stage.jump_ratio
        save_profile(self.profile)

    def _scale(self, field: str, value: int) -> None:
        if self.profile is None:
            return
        setattr(self.profile, field, value / 100)
        save_profile(self.profile)

    def _set_speed(self, value: int) -> None:
        self.void.speed = max(0.6, min(1.8, value / 100))
        if self.profile is None:
            return
        self.profile.void_speed = self.void.speed
        save_profile(self.profile)

    def on_lines_moved(self, left: float, right: float) -> None:
        self.pending_lines = (left, right)
        if self.profile is None:
            return
        self.profile.set_lines(left, right)
        save_profile(self.profile)
        self._set_slider(self.zone_slider, self.profile.zone_scale)

    def _show_lines(self) -> None:
        if self.camera.dragging:
            return
        if self.profile is not None:
            self.camera.lines = self.profile.lines()
        elif self.pending_lines is not None:
            self.camera.lines = self.pending_lines

    def on_status(self, text: str) -> None:
        self.zone_word.setText(text)

    def on_failed(self, text: str) -> None:
        self.zone_word.setText("Camera")
        self.instruction.setText(text)

    def on_packet(self, sample: BodySample, image) -> None:
        self.sample = sample
        self.camera.image = image
        self.camera.sample = sample
        if self.wizard is not None:
            self.wizard.push(sample)
            if self.wizard.profile is not None:
                self.profile = self.wizard.profile
                if self.camera.lines is not None:
                    self.profile.set_lines(*self.camera.lines)
                self.profile.jump_percent = self.stage.jump_ratio
                save_profile(self.profile)
                self.mapper = ActionMapper(self.profile)
                self.wizard = None
                self._sync_controls()
        elif self.mapper is not None:
            self.mapper.update(sample)
        self.crossed = bool(sample.points) and forearms_crossed(sample.points)
        self.controls = self.stage.update(sample, self.camera.lines)
        self.camera.marker = (self.stage.x, self.stage.y) if sample.present else None
        self._show_lines()

        if self.controls.zone is not None:
            self.camera.active = self.controls.zone
            self.zone_word.setText(self.controls.label)
        else:
            self.camera.active = None
            self.camera.marker = None
            if sample.warning and self.wizard is not None:
                self.zone_word.setText("Camera")
        if self.wizard is not None:
            self.instruction.setText(self.wizard.message)
        elif self.controls.zone is None and sample.warning:
            self.instruction.setText(sample.warning)
        else:
            self.instruction.setText(self._instruction_text())
        self.camera.update()

    def toggle_run(self) -> None:
        if self.running:
            self.running = False
        elif self.mode == "void":
            self.void.reset()
            self._apply_body()
            self.running = True
            self.void_view.started = True
        else:
            self.runner.reset()
            self._apply_body()
            self.running = True
            self.track.started = True
        self._sync_controls()
        self.setFocus()

    def toggle_mode(self) -> None:
        self.running = False
        self.mode = "practice" if self.mode == "void" else "void"
        self._sync_controls()
        self.setFocus()

    def toggle_camera(self) -> None:
        self.camera_on = not self.camera_on
        self._sync_controls()
        self.setFocus()

    def recalibrate(self) -> None:
        self.running = False
        self.wizard = Wizard()
        self.mapper = None
        self.controls = None
        self._sync_controls()
        self.setFocus()

    def tick(self) -> None:
        now = time.perf_counter()
        dt = now - self._last
        self._last = now
        if self.mode == "void":
            self._apply_body()
            self.void.update(dt, self.running and not self.void.over)
            if self.void.over:
                self.running = False
                if self.crossed:
                    self._cross_time += dt
                    if self._cross_time >= 0.45:
                        self.void.reset()
                        self._cross_time = 0.0
                        self._apply_body()
                        self.running = True
                        self.void_view.started = True
                else:
                    self._cross_time = 0.0
            else:
                self._cross_time = 0.0
            self.void_view.update()
            return
        if self.running and self.runner.alive:
            self._apply_body()
            self.runner.update(dt, self.track.width(), self.track.height())
            if not self.runner.alive:
                self.running = False
                self._sync_controls()
        else:
            self._apply_body()
            self.runner.tick_air(dt)
        self.track.update()

    def _body_driving(self) -> bool:
        return bool(self.camera_on and self.controls is not None and self.controls.zone is not None)

    def _apply_body(self) -> None:
        controls = self.controls if self.camera_on else None
        if controls is None or controls.zone is None:
            self.runner.set_duck(self.duck_key)
            self.void.set_duck(self.duck_key)
            return
        self.runner.lane = controls.zone
        self.void.lane = controls.zone
        if controls.jump:
            self.runner.ducking = False
            self.runner.jump()
            self.void.ducking = False
            self.void.jump()
        else:
            ducking = self.duck_key or controls.duck
            self.runner.set_duck(ducking)
            self.void.set_duck(ducking)

    def keyPressEvent(self, event) -> None:
        if event.isAutoRepeat():
            return
        if event.key() == Qt.Key.Key_F8:
            self.toggle_camera()
            return
        if not self.running:
            super().keyPressEvent(event)
            return
        if event.key() == Qt.Key.Key_Left and not self._body_driving():
            self.runner.lane = max(-1, self.runner.lane - 1)
            self.void.lane = max(-1, self.void.lane - 1)
        elif event.key() == Qt.Key.Key_Right and not self._body_driving():
            self.runner.lane = min(1, self.runner.lane + 1)
            self.void.lane = min(1, self.void.lane + 1)
        elif event.key() == Qt.Key.Key_Up:
            self.runner.jump()
            self.void.jump()
        elif event.key() == Qt.Key.Key_Down:
            self.duck_key = True
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.isAutoRepeat():
            return
        if event.key() == Qt.Key.Key_Down:
            self.duck_key = False
        else:
            super().keyReleaseEvent(event)

    def closeEvent(self, event) -> None:
        self.worker.requestInterruption()
        self.worker.wait(2000)
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
