"""Three lanes, a block to jump, and a bar to crouch. Nothing beyond that."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class Obstacle:
    lane: int
    kind: str
    y: float


class Runner:
    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.lane = 0
        self.jump_t = -1.0
        self.ducking = False
        self.obstacles: list[Obstacle] = []
        self.spawn_wait = 1.1
        self.score = 0.0
        self.alive = True
        self.scroll = 0.0
        self.speed = 340.0

    def jump(self) -> None:
        if self.alive and self.jump_t < 0 and not self.ducking:
            self.jump_t = 0.0

    def set_duck(self, ducking: bool) -> None:
        self.ducking = False if self.jump_t >= 0 else ducking

    def tick_air(self, dt: float) -> None:
        if self.jump_t < 0:
            return
        self.jump_t += min(dt, 0.05)
        if self.jump_t > 0.52:
            self.jump_t = -1.0

    def update(self, dt: float, width: int, height: int) -> None:
        if not self.alive or width < 40 or height < 40:
            return
        dt = min(dt, 0.05)
        self.score += dt * 8.0
        self.speed = 340.0 + min(280.0, self.score * 1.6)
        self.scroll += self.speed * dt
        if self.jump_t >= 0:
            self.jump_t += dt
            if self.jump_t > 0.52:
                self.jump_t = -1.0

        self.spawn_wait -= dt
        if self.spawn_wait <= 0:
            self.spawn_wait = max(0.72, 1.25 - self.score * 0.008)
            self._spawn(height)

        for obstacle in self.obstacles:
            obstacle.y += self.speed * dt
        self._collide(width, height)
        limit = height + 80
        self.obstacles = [item for item in self.obstacles if item.y < limit]

    def _spawn(self, height: float) -> None:
        busy = {item.lane for item in self.obstacles if item.y < height * 0.45}
        open_lanes = [lane for lane in (-1, 0, 1) if lane not in busy]
        if not open_lanes:
            return
        kind = random.choice(("block", "bar"))
        self.obstacles.append(Obstacle(random.choice(open_lanes), kind, -50.0))

    def pose(self, height: int) -> dict[str, float]:
        foot = height * 0.84
        lift = 0.0
        if self.jump_t >= 0:
            lift = 110.0 * math.sin(math.pi * min(self.jump_t, 0.52) / 0.52)
        body_h = 28.0 if self.ducking else 62.0
        return {"foot": foot, "lift": lift, "top": foot - body_h - lift, "body_h": body_h}

    def _collide(self, width: int, height: int) -> None:
        pose = self.pose(height)
        player_top = pose["top"]
        player_bottom = pose["foot"] - pose["lift"]
        for obstacle in self.obstacles:
            if obstacle.lane != self.lane:
                continue
            box = obstacle_box(obstacle, width)
            if player_top < box[1] + box[3] and player_bottom > box[1]:
                self.alive = False
                return


def lane_centers(width: int) -> dict[int, float]:
    return {-1: width * 0.22, 0: width * 0.5, 1: width * 0.78}


def lane_width(width: int) -> float:
    return width * 0.2


def obstacle_box(obstacle: Obstacle, width: int) -> tuple[float, float, float, float]:
    center = lane_centers(width)[obstacle.lane]
    if obstacle.kind == "block":
        box_w = lane_width(width) * 0.62
        box_h = 30.0
        y = obstacle.y - box_h
    else:
        box_w = lane_width(width) * 0.8
        box_h = 22.0
        y = obstacle.y - 80.0
    return center - box_w / 2, y, box_w, box_h
