#!/usr/bin/env python3
"""
Holonomy Lock (PTC Alignment Puzzle) — Pygame

Goal:
- Drive a point on the unit sphere, forming a closed loop.
- Your local tangent axis e1 is parallel-transported along the path.
- When you close the loop, the net rotation (holonomy) is compared to a target angle.

Controls:
- Left/Right: steer (change heading angle in tangent plane)
- Up: move forward
- Down: brake (slow)
- Space: attempt to close/unlock (only works if near start)
- R: reset level (new target)
- Esc/Quit: exit

Install:
  pip install pygame numpy

Notes:
- Intrinsic motion uses the spherical exponential map.
- Parallel transport along each step uses the ambient rotation that maps x -> x'
  (exact for great-circle steps), then projection onto the new tangent plane.
- Rendering uses a simple equirectangular map (lon-lat) for lightweight visuals.
"""

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pygame


# -------------------------
# Math helpers (sphere PTC)
# -------------------------

EPS = 1e-10
FONT_SCALE = 1.8
TANGENT_ARROW_LENGTH = 84
TANGENT_ARROW_WIDTH = 6
TANGENT_ARROW_HEAD = 22


def normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < EPS:
        return v.copy()
    return v / n


def rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = normalize(axis)
    x, y, z = axis
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    I = np.eye(3, dtype=np.float64)
    return I + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


def exp_map_s2(x: np.ndarray, u: np.ndarray, step: float) -> np.ndarray:
    """
    Exponential map on the unit sphere:
      x' = cos(step)*x + sin(step)*h  where h = u/||u|| is unit tangent direction at x.
    step is the geodesic distance.
    """
    u = u - float(np.dot(u, x)) * x  # ensure tangent
    n = float(np.linalg.norm(u))
    if n < EPS:
        return x.copy()
    h = u / n
    return normalize(math.cos(step) * x + math.sin(step) * h)


def transport_on_s2(x: np.ndarray, x2: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Parallel transport of tangent vector v from T_x S^2 to T_{x2} S^2 along the
    great-circle geodesic step x -> x2.
    Implemented by rotating v in R^3 by the rotation that maps x to x2, then
    projecting to the new tangent plane.
    """
    x = normalize(x)
    x2 = normalize(x2)
    axis = np.cross(x, x2)
    axn = float(np.linalg.norm(axis))
    if axn < EPS:
        # tiny step, no rotation needed
        v2 = v.copy()
    else:
        dot = float(np.clip(np.dot(x, x2), -1.0, 1.0))
        ang = math.atan2(axn, dot)
        R = rodrigues(axis, ang)
        v2 = R @ v

    # project to tangent at x2
    v2 = v2 - float(np.dot(v2, x2)) * x2
    return normalize(v2)


def tangent_basis(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return a stable local east/north basis for display given x on S^2.
    Not used for transport; only for 2D arrow rendering.
    """
    x = normalize(x)
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(x, z))) > 0.95:
        z = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    e = np.cross(z, x)
    e = normalize(e)
    n = normalize(np.cross(x, e))
    return e, n


def lonlat_from_unit(x: np.ndarray) -> Tuple[float, float]:
    """Return lon in [-pi, pi], lat in [-pi/2, pi/2]."""
    x = normalize(x)
    lon = math.atan2(float(x[1]), float(x[0]))
    lat = math.asin(float(np.clip(x[2], -1.0, 1.0)))
    return lon, lat


def screen_from_lonlat(lon: float, lat: float, w: int, h: int) -> Tuple[int, int]:
    """Equirectangular projection."""
    sx = int((lon + math.pi) / (2 * math.pi) * (w - 1))
    sy = int((math.pi / 2 - lat) / math.pi * (h - 1))
    return sx, sy


def signed_angle_in_tangent(x: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """
    Signed angle from a to b in T_x S^2, using x as orientation (right-hand rule).
    Both a,b assumed tangent and normalized.
    """
    a = a - float(np.dot(a, x)) * x
    b = b - float(np.dot(b, x)) * x
    a = normalize(a)
    b = normalize(b)
    # atan2( <x, a×b>, <a,b> )
    return math.atan2(float(np.dot(x, np.cross(a, b))), float(np.clip(np.dot(a, b), -1.0, 1.0)))


# -------------------------
# Game state
# -------------------------

@dataclass
class Level:
    target_deg: float
    tol_deg: float = 8.0
    close_dist: float = 0.22  # geodesic distance threshold (radians)

@dataclass
class Player:
    x: np.ndarray
    e1: np.ndarray
    heading_angle: float = 0.0
    speed: float = 0.0


# -------------------------
# Rendering helpers
# -------------------------

def draw_text(surface, text, pos, color=(240, 240, 240), size=20, align_left=True):
    scaled_size = max(12, int(round(size * FONT_SCALE)))
    font = pygame.font.SysFont("DejaVu Sans", scaled_size)
    img = font.render(text, True, color)
    rect = img.get_rect()
    if align_left:
        rect.topleft = pos
    else:
        rect.center = pos
    surface.blit(img, rect)


def line_step(size: int, pad: int = 6) -> int:
    scaled_size = max(12, int(round(size * FONT_SCALE)))
    return scaled_size + pad


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
    """
    Draw trail with basic wrap handling: if a segment jumps across the left/right edge,
    split it.
    """
    if len(pts) < 2:
        return
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        dx = x1 - x0
        if abs(dx) > w // 2:
            # wrap split
            if dx > 0:
                # went right across boundary; draw to left edge and from right edge
                pygame.draw.line(screen, color, (x0, y0), (0, y1), width)
                pygame.draw.line(screen, color, (w - 1, y0), (x1, y1), width)
            else:
                pygame.draw.line(screen, color, (x0, y0), (w - 1, y1), width)
                pygame.draw.line(screen, color, (0, y0), (x1, y1), width)
        else:
            pygame.draw.line(screen, color, (x0, y0), (x1, y1), width)


# -------------------------
# Main
# -------------------------

def new_level() -> Level:
    target = random.choice([20, 35, 50, 65, 80, 95, 110])
    target += random.uniform(-4.0, 4.0)
    return Level(target_deg=float(target))


def reset_player() -> Player:
    # pick a start point not too close to poles for nice visuals
    lon = random.uniform(-math.pi, math.pi)
    lat = random.uniform(-0.9, 0.9)
    x = np.array([math.cos(lat) * math.cos(lon),
                  math.cos(lat) * math.sin(lon),
                  math.sin(lat)], dtype=np.float64)
    x = normalize(x)

    # seed e1 from a stable display basis
    east, north = tangent_basis(x)
    e1 = normalize(0.9 * east + 0.1 * north)
    return Player(x=x, e1=e1, heading_angle=0.0, speed=0.0)


def main():
    pygame.init()
    pygame.display.set_caption("Holonomy Lock (PTC) — Pygame")
    W, H = 2560, 1440
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()

    level = new_level()
    player = reset_player()
    x0 = player.x.copy()
    e1_0 = player.e1.copy()

    trail: List[Tuple[int, int]] = []
    running = True
    unlocked = False
    last_hol_deg = None

    # background grid pre-render
    grid = pygame.Surface((W, H))
    grid.fill((7, 11, 20))
    # meridians/parallels
    for i in range(13):
        lon = -math.pi + i * (2 * math.pi / 12)
        pts = []
        for j in range(181):
            lat = -math.pi / 2 + j * (math.pi / 180)
            x = np.array([math.cos(lat) * math.cos(lon),
                          math.cos(lat) * math.sin(lon),
                          math.sin(lat)], dtype=np.float64)
            sx, sy = screen_from_lonlat(lon, lat, W, H)
            pts.append((sx, sy))
        pygame.draw.lines(grid, (22, 35, 60), False, pts, 1)

    for j in range(7):
        lat = -math.pi / 2 + j * (math.pi / 6)
        pts = []
        for i in range(361):
            lon = -math.pi + i * (2 * math.pi / 360)
            sx, sy = screen_from_lonlat(lon, lat, W, H)
            pts.append((sx, sy))
        pygame.draw.lines(grid, (22, 35, 60), False, pts, 1)

    while running:
        dt = clock.tick(60) / 1000.0

        # events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    level = new_level()
                    player = reset_player()
                    x0 = player.x.copy()
                    e1_0 = player.e1.copy()
                    trail.clear()
                    unlocked = False
                    last_hol_deg = None
                elif event.key == pygame.K_SPACE:
                    # always attempt measurement if near start
                    ang = math.acos(float(np.clip(np.dot(player.x, x0), -1.0, 1.0)))
                    # if ang < level.close_dist:
                    if True:
                        e1_back = transport_on_s2(player.x, x0, player.e1)
                        theta = signed_angle_in_tangent(x0, e1_0, e1_back)
                        hol_deg = abs(theta) * 180.0 / math.pi
                        last_hol_deg = hol_deg

                        if abs(hol_deg - level.target_deg) <= level.tol_deg:
                            unlocked = True
                        else:
                            unlocked = False

        keys = pygame.key.get_pressed()
        steer = 0.0
        if keys[pygame.K_LEFT]:
            steer -= 1.0
        if keys[pygame.K_RIGHT]:
            steer += 1.0

        # steering changes heading_angle in local tangent plane
        player.heading_angle += steer * 1.8 * dt

        # throttle
        if keys[pygame.K_UP]:
            player.speed = min(1.0, player.speed + 1.2 * dt)
        elif keys[pygame.K_DOWN]:
            player.speed = max(0.0, player.speed - 1.8 * dt)
        else:
            player.speed = max(0.0, player.speed - 0.6 * dt)

        # step
        if player.speed > 1e-4:
            # build e2 from x and e1
            x = player.x
            e1 = player.e1
            e2 = normalize(np.cross(x, e1))

            # heading in tangent plane
            h = normalize(math.cos(player.heading_angle) * e1 + math.sin(player.heading_angle) * e2)

            step = player.speed * 0.55 * dt  # radians
            x2 = exp_map_s2(x, h, step)

            # transport e1 along step
            e1_2 = transport_on_s2(x, x2, e1)

            player.x = x2
            player.e1 = e1_2

        # render
        screen.blit(grid, (0, 0))

        # update trail
        lon, lat = lonlat_from_unit(player.x)
        px, py = screen_from_lonlat(lon, lat, W, H)
        trail.append((px, py))
        if len(trail) > 2200:
            trail = trail[-2200:]

        # draw start point
        lon0, lat0 = lonlat_from_unit(x0)
        sx0, sy0 = screen_from_lonlat(lon0, lat0, W, H)
        pygame.draw.circle(screen, (200, 200, 200), (sx0, sy0), 6)

        # draw trail + player
        draw_polyline_wrapped(screen, trail, (52, 211, 153), width=2, w=W)
        pygame.draw.circle(screen, (250, 204, 21), (px, py), 7)

        # draw tangent axis arrow projected into lon/lat basis (for intuitive feedback)
        east, north = tangent_basis(player.x)
        # express e1 in (east,north)
        a = float(np.dot(player.e1, east))
        b = float(np.dot(player.e1, north))
        # map to screen direction: +east => +x, +north => -y (since screen y down)
        ax = int(px + TANGENT_ARROW_LENGTH * a)
        ay = int(py - TANGENT_ARROW_LENGTH * b)
        draw_arrow(
            screen,
            (px, py),
            (ax, ay),
            (249, 115, 22),
            width=TANGENT_ARROW_WIDTH,
            head_len=TANGENT_ARROW_HEAD,
        )

        # draw control heading direction arrow (what steering currently commands)
        e2 = normalize(np.cross(player.x, player.e1))
        h_ctrl = normalize(math.cos(player.heading_angle) * player.e1 + math.sin(player.heading_angle) * e2)
        ah = float(np.dot(h_ctrl, east))
        bh = float(np.dot(h_ctrl, north))
        hx = int(px + 56 * ah)
        hy = int(py - 56 * bh)
        draw_arrow(screen, (px, py), (hx, hy), (34, 197, 94), width=3, head_len=12)

        # HUD
        hud_x = 14
        hud_y = 12
        draw_text(screen, "Holonomy Lock (PTC)", (hud_x, hud_y), (230, 235, 245), 26)
        hud_y += line_step(26, pad=8)
        draw_text(
            screen,
            f"Target holonomy: {level.target_deg:.1f}°  (tol ±{level.tol_deg:.0f}°)",
            (hud_x, hud_y),
            (253, 230, 138),
            20,
        )

        # closeness to start
        dist = math.acos(float(np.clip(np.dot(player.x, x0), -1.0, 1.0))) * 180.0 / math.pi
        hud_y += line_step(20, pad=6)
        draw_text(screen, f"Distance to start: {dist:.1f}°   (Space to try if close)", (hud_x, hud_y), (190, 210, 240), 18)
        hud_y += line_step(18, pad=6)
        draw_text(screen, "Orange: transported axis e1   Green: control heading", (hud_x, hud_y), (148, 163, 184), 18)

        if last_hol_deg is not None:
            color = (34, 197, 94) if unlocked else (248, 113, 113)
            hud_y += line_step(18, pad=6)
            draw_text(screen, f"Measured holonomy: {last_hol_deg:.1f}°", (hud_x, hud_y), color, 20)
            hud_y += line_step(20, pad=6)
            draw_text(screen, "UNLOCKED" if unlocked else "NOT MATCHED", (hud_x, hud_y), color, 22)

        controls_y = H - line_step(18, pad=10)
        draw_text(
            screen,
            "Controls: Up move, Left/Right steer, Space check, R reset",
            (hud_x, controls_y),
            (148, 163, 184),
            18,
        )

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
