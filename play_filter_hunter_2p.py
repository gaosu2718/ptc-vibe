#!/usr/bin/env python3
"""
Filter Hunter 2P (PTC Racing) — Pygame

Goal:
- Two players move directional kernels on the sphere and collect points from
  high PTC responses.
- First to 1000 points wins.
- After game end, press R to restart.

Controls:
- Player 1:
  Left/Right steer, Up forward, Down scan clockwise
- Player 2:
  A/D steer, W forward, S scan clockwise
- Global:
  1/2/3 size (1280/2560/3840 wide)
  R restart, Esc/Q exit
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pygame

EPS = 1e-10
FONT_SCALE = 1.8
WIN_SCORE = 200
TRAIL_MAX_POINTS = 300
TANGENT_ARROW_LENGTH = 80
TANGENT_ARROW_WIDTH = 6
TANGENT_ARROW_HEAD = 20
WINDOW_SIZES = {
    1: (1280, 720),
    2: (2560, 1440),
    3: (3840, 2160),
}


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < EPS:
        return v.copy()
    return v / n


def rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = normalize(axis)
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    eye = np.eye(3, dtype=np.float64)
    return eye + math.sin(angle) * k + (1.0 - math.cos(angle)) * (k @ k)


def exp_map_s2(x: np.ndarray, u: np.ndarray, step: float) -> np.ndarray:
    u = u - float(np.dot(u, x)) * x
    n = float(np.linalg.norm(u))
    if n < EPS:
        return x.copy()
    h = u / n
    return normalize(math.cos(step) * x + math.sin(step) * h)


def transport_on_s2(x: np.ndarray, x2: np.ndarray, v: np.ndarray) -> np.ndarray:
    x = normalize(x)
    x2 = normalize(x2)
    axis = np.cross(x, x2)
    axn = float(np.linalg.norm(axis))
    if axn < EPS:
        v2 = v.copy()
    else:
        dot = float(np.clip(np.dot(x, x2), -1.0, 1.0))
        ang = math.atan2(axn, dot)
        v2 = rodrigues(axis, ang) @ v
    v2 = v2 - float(np.dot(v2, x2)) * x2
    return normalize(v2)


def tangent_basis(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = normalize(x)
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(x, z))) > 0.95:
        z = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    east = normalize(np.cross(z, x))
    north = normalize(np.cross(x, east))
    return east, north


def lonlat_from_unit(x: np.ndarray) -> Tuple[float, float]:
    x = normalize(x)
    lon = math.atan2(float(x[1]), float(x[0]))
    lat = math.asin(float(np.clip(x[2], -1.0, 1.0)))
    return lon, lat


def screen_from_lonlat(lon: float, lat: float, w: int, h: int) -> Tuple[int, int]:
    sx = int((lon + math.pi) / (2 * math.pi) * (w - 1))
    sy = int((math.pi / 2 - lat) / math.pi * (h - 1))
    return sx, sy


@dataclass
class RidgeTarget:
    c: np.ndarray
    tdir: np.ndarray
    width: float
    strength: float


def make_targets(num: int = 2) -> List[RidgeTarget]:
    targets: List[RidgeTarget] = []
    for _ in range(num):
        lon = random.uniform(-math.pi, math.pi)
        lat = random.uniform(-1.0, 1.0)
        c = normalize(
            np.array(
                [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)],
                dtype=np.float64,
            )
        )
        z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        p = np.cross(c, z)
        if float(np.linalg.norm(p)) < 0.2:
            z = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            p = np.cross(c, z)
        p = normalize(p)
        tdir = normalize(np.cross(c, p))
        targets.append(
            RidgeTarget(
                c=c,
                tdir=tdir,
                width=random.uniform(0.10, 0.18),
                strength=random.uniform(0.8, 1.2),
            )
        )
    return targets


def ridge_signal(x: np.ndarray, targets: List[RidgeTarget]) -> float:
    x = normalize(x)
    val = 0.0
    for t in targets:
        d = abs(float(np.dot(t.c, x)))
        val += t.strength * math.exp(-0.5 * (d / t.width) ** 2)
    return val


KERNEL = [
    (+0.28, 0.00, +1.00),
    (+0.14, +0.10, +0.60),
    (+0.14, -0.10, +0.60),
    (-0.22, 0.00, -0.85),
]


def ptc_score(x: np.ndarray, e1: np.ndarray, kernel_angle: float, targets: List[RidgeTarget]) -> float:
    x = normalize(x)
    e1 = normalize(e1 - float(np.dot(e1, x)) * x)
    e2 = normalize(np.cross(x, e1))
    k1 = normalize(math.cos(kernel_angle) * e1 + math.sin(kernel_angle) * e2)
    k2 = normalize(np.cross(x, k1))
    total = 0.0
    for u, v, w in KERNEL:
        y = exp_map_s2(x, u * k1 + v * k2, step=math.hypot(u, v))
        total += w * ridge_signal(y, targets)
    return total


def draw_text(surface, text, pos, color=(240, 240, 240), size=20, align_left=True):
    scaled = max(12, int(round(size * FONT_SCALE)))
    font = pygame.font.SysFont("DejaVu Sans", scaled)
    img = font.render(text, True, color)
    rect = img.get_rect()
    rect.topleft = pos if align_left else rect.center
    surface.blit(img, rect)


def line_step(size: int, pad: int = 6) -> int:
    return max(12, int(round(size * FONT_SCALE))) + pad


def draw_arrow(
    surface: pygame.Surface,
    start: Tuple[int, int],
    end: Tuple[int, int],
    color: Tuple[int, int, int],
    width: int = 3,
    head_len: int = 12,
    head_angle_deg: float = 26.0,
) -> None:
    pygame.draw.line(surface, color, start, end, width)
    vx = float(end[0] - start[0])
    vy = float(end[1] - start[1])
    n = math.hypot(vx, vy)
    if n < EPS:
        return
    theta = math.atan2(vy, vx)
    ang = math.radians(head_angle_deg)
    left = (
        int(end[0] + head_len * math.cos(theta + math.pi - ang)),
        int(end[1] + head_len * math.sin(theta + math.pi - ang)),
    )
    right = (
        int(end[0] + head_len * math.cos(theta + math.pi + ang)),
        int(end[1] + head_len * math.sin(theta + math.pi + ang)),
    )
    pygame.draw.polygon(surface, color, [end, left, right])


def draw_polyline_wrapped(screen, pts: List[Tuple[int, int]], color, width=2, w=1):
    if len(pts) < 2:
        return
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        dx = x1 - x0
        if abs(dx) > w // 2:
            if dx > 0:
                pygame.draw.line(screen, color, (x0, y0), (0, y1), width)
                pygame.draw.line(screen, color, (w - 1, y0), (x1, y1), width)
            else:
                pygame.draw.line(screen, color, (x0, y0), (w - 1, y1), width)
                pygame.draw.line(screen, color, (0, y0), (x1, y1), width)
        else:
            pygame.draw.line(screen, color, (x0, y0), (x1, y1), width)


def build_grid_surface(w: int, h: int) -> pygame.Surface:
    grid = pygame.Surface((w, h))
    grid.fill((5, 10, 18))
    for i in range(13):
        lon = -math.pi + i * (2 * math.pi / 12)
        pts = []
        for j in range(181):
            lat = -math.pi / 2 + j * (math.pi / 180)
            pts.append(screen_from_lonlat(lon, lat, w, h))
        pygame.draw.lines(grid, (18, 32, 54), False, pts, 1)
    for j in range(7):
        lat = -math.pi / 2 + j * (math.pi / 6)
        pts = []
        for i in range(361):
            lon = -math.pi + i * (2 * math.pi / 360)
            pts.append(screen_from_lonlat(lon, lat, w, h))
        pygame.draw.lines(grid, (18, 32, 54), False, pts, 1)
    return grid


def target_points_for_display(t: RidgeTarget, samples: int = 140) -> List[np.ndarray]:
    c = normalize(t.c)
    v = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    a = np.cross(c, v)
    if float(np.linalg.norm(a)) < 0.2:
        v = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        a = np.cross(c, v)
    a = normalize(a)
    b = normalize(np.cross(c, a))
    pts = []
    for i in range(samples):
        phi = -math.pi + i * (2 * math.pi / (samples - 1))
        pts.append(normalize(math.cos(phi) * a + math.sin(phi) * b))
    return pts


@dataclass
class ControlScheme:
    left: int
    right: int
    up: int
    scan_clockwise: int


@dataclass
class PlayerState:
    x: np.ndarray
    e1: np.ndarray
    heading_angle: float = 0.0
    kernel_angle: float = 0.0
    speed: float = 0.0
    score: int = 0
    combo: int = 0
    trail: List[Tuple[int, int]] = field(default_factory=list)


def make_player(lon: float, lat: float) -> PlayerState:
    x = normalize(
        np.array(
            [math.cos(lat) * math.cos(lon), math.cos(lat) * math.sin(lon), math.sin(lat)],
            dtype=np.float64,
        )
    )
    east, north = tangent_basis(x)
    e1 = normalize(0.7 * east + 0.3 * north)
    return PlayerState(x=x, e1=e1)


def reset_players() -> Tuple[PlayerState, PlayerState]:
    lon = random.uniform(-math.pi, math.pi)
    lat = random.uniform(-0.85, 0.85)
    p1 = make_player(lon, lat)
    p2 = make_player(((lon + math.pi) % (2 * math.pi)) - math.pi, -0.45 * lat)
    return p1, p2


def update_player(player: PlayerState, controls: ControlScheme, keys, dt: float) -> None:
    steer = 0.0
    if keys[controls.left]:
        steer -= 1.0
    if keys[controls.right]:
        steer += 1.0
    player.heading_angle += steer * 1.9 * dt

    if keys[controls.scan_clockwise]:
        player.kernel_angle += 2.2 * dt

    if keys[controls.up]:
        player.speed = min(1.0, player.speed + 1.3 * dt)
    else:
        player.speed = max(0.0, player.speed - 0.7 * dt)

    if player.speed > 1e-4:
        x = player.x
        e1 = player.e1
        e2 = normalize(np.cross(x, e1))
        h = normalize(math.cos(player.heading_angle) * e1 + math.sin(player.heading_angle) * e2)
        step = player.speed * 0.55 * dt
        x2 = exp_map_s2(x, h, step)
        player.e1 = transport_on_s2(x, x2, e1)
        player.x = x2


def main():
    pygame.init()
    pygame.display.set_caption("Filter Hunter 2P (PTC Racing) — Pygame")
    w, h = WINDOW_SIZES[3]
    screen = pygame.display.set_mode((w, h))
    clock = pygame.time.Clock()

    grid = build_grid_surface(w, h)

    controls_1 = ControlScheme(
        left=pygame.K_LEFT,
        right=pygame.K_RIGHT,
        up=pygame.K_UP,
        scan_clockwise=pygame.K_DOWN,
    )
    controls_2 = ControlScheme(
        left=pygame.K_a,
        right=pygame.K_d,
        up=pygame.K_w,
        scan_clockwise=pygame.K_s,
    )

    p1_colors = {
        "dot": (250, 204, 21),
        "trail": (52, 211, 153),
        "axis": (249, 115, 22),
        "ctrl": (34, 197, 94),
        "kernel": (147, 197, 253),
    }
    p2_colors = {
        "dot": (244, 114, 182),
        "trail": (192, 132, 252),
        "axis": (244, 63, 94),
        "ctrl": (45, 212, 191),
        "kernel": (196, 181, 253),
    }

    def reset_game():
        targets_local = make_targets(2)
        curves_local = [target_points_for_display(t) for t in targets_local]
        p1_local, p2_local = reset_players()
        return targets_local, curves_local, p1_local, p2_local, None

    targets, target_curves, p1, p2, winner = reset_game()
    capture_threshold = 1.5
    running = True

    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    targets, target_curves, p1, p2, winner = reset_game()
                elif event.key in (pygame.K_1, pygame.K_KP1):
                    w, h = WINDOW_SIZES[1]
                    screen = pygame.display.set_mode((w, h))
                    grid = build_grid_surface(w, h)
                    p1.trail.clear()
                    p2.trail.clear()
                elif event.key in (pygame.K_2, pygame.K_KP2):
                    w, h = WINDOW_SIZES[2]
                    screen = pygame.display.set_mode((w, h))
                    grid = build_grid_surface(w, h)
                    p1.trail.clear()
                    p2.trail.clear()
                elif event.key in (pygame.K_3, pygame.K_KP3):
                    w, h = WINDOW_SIZES[3]
                    screen = pygame.display.set_mode((w, h))
                    grid = build_grid_surface(w, h)
                    p1.trail.clear()
                    p2.trail.clear()

        keys = pygame.key.get_pressed()

        if winner is None:
            update_player(p1, controls_1, keys, dt)
            update_player(p2, controls_2, keys, dt)

            resp1 = ptc_score(p1.x, p1.e1, p1.kernel_angle, targets)
            resp2 = ptc_score(p2.x, p2.e1, p2.kernel_angle, targets)

            for player, resp in ((p1, resp1), (p2, resp2)):
                if resp > capture_threshold:
                    player.score += 10 + 3 * player.combo
                    player.combo = min(player.combo + 1, 20)
                    idx = random.randrange(len(targets))
                    targets[idx] = make_targets(1)[0]
                    target_curves[idx] = target_points_for_display(targets[idx])
                else:
                    player.combo = max(0, player.combo - 1)

            if p1.score >= WIN_SCORE and p2.score >= WIN_SCORE:
                winner = 0 if p1.score == p2.score else (1 if p1.score > p2.score else 2)
            elif p1.score >= WIN_SCORE:
                winner = 1
            elif p2.score >= WIN_SCORE:
                winner = 2
        else:
            resp1 = ptc_score(p1.x, p1.e1, p1.kernel_angle, targets)
            resp2 = ptc_score(p2.x, p2.e1, p2.kernel_angle, targets)

        screen.blit(grid, (0, 0))

        for curve_pts in target_curves:
            pts2d = []
            for p in curve_pts:
                lonp, latp = lonlat_from_unit(p)
                pts2d.append(screen_from_lonlat(lonp, latp, w, h))
            draw_polyline_wrapped(screen, pts2d, (34, 211, 238), width=2, w=w)

        for player, colors in ((p1, p1_colors), (p2, p2_colors)):
            lon, lat = lonlat_from_unit(player.x)
            px, py = screen_from_lonlat(lon, lat, w, h)
            player.trail.append((px, py))
            if len(player.trail) > TRAIL_MAX_POINTS:
                player.trail = player.trail[-TRAIL_MAX_POINTS:]

            draw_polyline_wrapped(screen, player.trail, colors["trail"], width=2, w=w)
            pygame.draw.circle(screen, colors["dot"], (px, py), 7)

            east, north = tangent_basis(player.x)
            a1 = float(np.dot(player.e1, east))
            b1 = float(np.dot(player.e1, north))
            draw_arrow(
                screen,
                (px, py),
                (int(px + TANGENT_ARROW_LENGTH * a1), int(py - TANGENT_ARROW_LENGTH * b1)),
                colors["axis"],
                width=TANGENT_ARROW_WIDTH,
                head_len=TANGENT_ARROW_HEAD,
            )

            e2 = normalize(np.cross(player.x, player.e1))
            h_ctrl = normalize(math.cos(player.heading_angle) * player.e1 + math.sin(player.heading_angle) * e2)
            ah = float(np.dot(h_ctrl, east))
            bh = float(np.dot(h_ctrl, north))
            draw_arrow(
                screen,
                (px, py),
                (int(px + 54 * ah), int(py - 54 * bh)),
                colors["ctrl"],
                width=3,
                head_len=11,
            )

            k1 = normalize(math.cos(player.kernel_angle) * player.e1 + math.sin(player.kernel_angle) * e2)
            ak = float(np.dot(k1, east))
            bk = float(np.dot(k1, north))
            draw_arrow(
                screen,
                (px, py),
                (int(px + 46 * ak), int(py - 46 * bk)),
                colors["kernel"],
                width=3,
                head_len=10,
            )

            k2 = normalize(np.cross(player.x, k1))
            for u, v, kw in KERNEL:
                y = exp_map_s2(player.x, u * k1 + v * k2, step=math.hypot(u, v))
                lony, laty = lonlat_from_unit(y)
                sx, sy = screen_from_lonlat(lony, laty, w, h)
                col = (248, 250, 252) if kw > 0 else (253, 164, 175)
                pygame.draw.circle(screen, col, (sx, sy), 3)

        hud_x = 14
        hud_y = 12
        draw_text(screen, "Filter Hunter 2P (PTC Racing)", (hud_x, hud_y), (230, 235, 245), 26)
        hud_y += line_step(26, pad=8)

        draw_text(
            screen,
            f"P1 Score: {p1.score:4d}  Combo: {p1.combo:2d}  Resp: {resp1:+0.3f}",
            (hud_x, hud_y),
            p1_colors["dot"],
            20,
        )
        hud_y += line_step(20, pad=6)
        draw_text(
            screen,
            f"P2 Score: {p2.score:4d}  Combo: {p2.combo:2d}  Resp: {resp2:+0.3f}",
            (hud_x, hud_y),
            p2_colors["dot"],
            20,
        )
        hud_y += line_step(20, pad=6)
        draw_text(
            screen,
            f"Race target: {WIN_SCORE} points   Capture if PTC response > {capture_threshold:.2f}",
            (hud_x, hud_y),
            (190, 210, 240),
            18,
        )
        hud_y += line_step(18, pad=6)
        draw_text(
            screen,
            "Orange/Red: transported axis  Green/Teal: control heading  Blue/Violet: kernel scan axis",
            (hud_x, hud_y),
            (148, 163, 184),
            18,
        )

        controls_y = h - line_step(18, pad=12)
        draw_text(
            screen,
            "P1 Arrows + Down scan | P2 WASD + S scan | 1/2/3 size | R restart | Esc quit",
            (hud_x, controls_y),
            (148, 163, 184),
            18,
        )

        if winner is not None:
            if winner == 0:
                msg = "DRAW - BOTH REACHED 1000"
                color = (250, 204, 21)
            elif winner == 1:
                msg = "PLAYER 1 WINS!  (Press R to restart)"
                color = p1_colors["dot"]
            else:
                msg = "PLAYER 2 WINS!  (Press R to restart)"
                color = p2_colors["dot"]
            draw_text(screen, msg, (w // 2, h // 2), color=color, size=34, align_left=False)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
