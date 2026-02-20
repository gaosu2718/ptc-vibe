#!/usr/bin/env python3
"""
Filter Hunter (PTC Orientation-Sensitive Targeting) — Pygame

Goal:
- Move a directional kernel over a sphere.
- Capture targets that are detectable only when the kernel is correctly aligned
  (PTC response spikes).
- Your kernel orientation evolves via parallel transport; curvature affects alignment.

Controls:
- Left/Right: steer (change heading angle in tangent plane)
- Up: move forward
- Down: rotate kernel clockwise relative to the transported frame (scan)
- 1/2/3: switch window size (1280/2560/3840 wide)
- R: respawn targets/signal
- Esc/Quit: exit

Install:
  pip install pygame numpy

Notes:
- Uses the same intrinsic exp-map motion and parallel-transport update as the holonomy game.
- Rendering uses equirectangular map; targets are intrinsic “ridges” on the sphere.
"""

import math
import random
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pygame

EPS = 1e-10
FONT_SCALE = 1.8
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
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)
    I = np.eye(3, dtype=np.float64)
    return I + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


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
        R = rodrigues(axis, ang)
        v2 = R @ v
    v2 = v2 - float(np.dot(v2, x2)) * x2
    return normalize(v2)


def tangent_basis(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    x = normalize(x)
    z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(x, z))) > 0.95:
        z = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    e = normalize(np.cross(z, x))
    n = normalize(np.cross(x, e))
    return e, n


def lonlat_from_unit(x: np.ndarray) -> Tuple[float, float]:
    x = normalize(x)
    lon = math.atan2(float(x[1]), float(x[0]))
    lat = math.asin(float(np.clip(x[2], -1.0, 1.0)))
    return lon, lat


def screen_from_lonlat(lon: float, lat: float, w: int, h: int) -> Tuple[int, int]:
    sx = int((lon + math.pi) / (2 * math.pi) * (w - 1))
    sy = int((math.pi / 2 - lat) / math.pi * (h - 1))
    return sx, sy


def geodesic_angle(a: np.ndarray, b: np.ndarray) -> float:
    return math.acos(float(np.clip(np.dot(normalize(a), normalize(b)), -1.0, 1.0)))


# -------------------------
# Signal + kernel (PTC score)
# -------------------------

@dataclass
class RidgeTarget:
    c: np.ndarray          # ridge normal direction (defines a great-circle ridge by c·x ~ 0)
    tdir: np.ndarray       # preferred tangent direction along the ridge (unit, tangent at points near ridge)
    width: float           # angular width
    strength: float        # amplitude


def make_targets(num: int = 4) -> List[RidgeTarget]:
    targets: List[RidgeTarget] = []
    for _ in range(num):
        # pick random unit vector c (ridge is where c·x ~ 0)
        lon = random.uniform(-math.pi, math.pi)
        lat = random.uniform(-1.0, 1.0)
        c = normalize(np.array([math.cos(lat) * math.cos(lon),
                                math.cos(lat) * math.sin(lon),
                                math.sin(lat)], dtype=np.float64))
        # pick a reference point on ridge: choose any vector orthogonal to c
        z = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        p = np.cross(c, z)
        if float(np.linalg.norm(p)) < 0.2:
            z = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            p = np.cross(c, z)
        p = normalize(p)  # p lies on ridge (c·p=0)
        # preferred tangent direction tdir at p: along the ridge direction (perp to both c and p)
        tdir = normalize(np.cross(c, p))  # tangent at p
        width = random.uniform(0.10, 0.18)
        strength = random.uniform(0.8, 1.2)
        targets.append(RidgeTarget(c=c, tdir=tdir, width=width, strength=strength))
    return targets


def ridge_signal(x: np.ndarray, targets: List[RidgeTarget]) -> float:
    """
    Anisotropic ridge-like signal:
    - High when x is near the great circle c·x=0.
    - Additional orientation selectivity: prefers kernel axis aligned with ridge direction.
    Orientation selectivity is handled in the convolution score, not here (keep f scalar).
    """
    x = normalize(x)
    val = 0.0
    for t in targets:
        # distance to ridge (great circle) via |c·x|
        d = abs(float(np.dot(t.c, x)))
        # convert to angular proximity: near 0 => close to ridge
        # use gaussian in d (d is sin of angle to ridge plane)
        val += t.strength * math.exp(-0.5 * (d / t.width) ** 2)
    return val


# Directional kernel samples in tangent coordinates (u,v) with weights
# Choose an elongated "edge" detector: positive ahead, negative behind.
KERNEL = [
    (+0.28, 0.00, +1.00),
    (+0.14, +0.10, +0.60),
    (+0.14, -0.10, +0.60),
    (-0.22, 0.00, -0.85),
]


def ptc_score(x: np.ndarray, e1: np.ndarray, kernel_angle: float, targets: List[RidgeTarget]) -> float:
    """
    Intrinsic convolution sample at x using local transported frame (e1,e2),
    with an extra controllable in-frame rotation kernel_angle for scanning.
    """
    x = normalize(x)
    e1 = e1 - float(np.dot(e1, x)) * x
    e1 = normalize(e1)
    e2 = normalize(np.cross(x, e1))

    # rotate kernel axes within the tangent plane by kernel_angle
    k1 = normalize(math.cos(kernel_angle) * e1 + math.sin(kernel_angle) * e2)
    k2 = normalize(np.cross(x, k1))

    total = 0.0
    for u, v, w in KERNEL:
        y = exp_map_s2(x, u * k1 + v * k2, step=math.hypot(u, v))
        total += w * ridge_signal(y, targets)
    return total


# -------------------------
# Rendering / HUD
# -------------------------

def draw_text(surface, text, pos, color=(240, 240, 240), size=20, align_left=True):
    scaled_size = max(12, int(round(size * FONT_SCALE)))
    font = pygame.font.SysFont("DejaVu Sans", scaled_size)
    img = font.render(text, True, color)
    rect = img.get_rect()
    rect.topleft = pos if align_left else rect.center
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
            sx, sy = screen_from_lonlat(lon, lat, w, h)
            pts.append((sx, sy))
        pygame.draw.lines(grid, (18, 32, 54), False, pts, 1)

    for j in range(7):
        lat = -math.pi / 2 + j * (math.pi / 6)
        pts = []
        for i in range(361):
            lon = -math.pi + i * (2 * math.pi / 360)
            sx, sy = screen_from_lonlat(lon, lat, w, h)
            pts.append((sx, sy))
        pygame.draw.lines(grid, (18, 32, 54), False, pts, 1)
    return grid


def target_points_for_display(t: RidgeTarget, samples: int = 140) -> List[np.ndarray]:
    """
    Build points along the ridge great circle (c·x=0) for display.
    Choose an orthonormal basis (a,b) spanning the plane orthogonal to c:
      x(φ)=cosφ a + sinφ b
    """
    c = normalize(t.c)
    # pick a not-parallel vector
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
        x = math.cos(phi) * a + math.sin(phi) * b
        pts.append(normalize(x))
    return pts


# -------------------------
# Main
# -------------------------

def main():
    pygame.init()
    pygame.display.set_caption("Filter Hunter (PTC) — Pygame")
    W, H = WINDOW_SIZES[3]
    screen = pygame.display.set_mode((W, H))
    clock = pygame.time.Clock()

    # pre-render map grid
    grid = build_grid_surface(W, H)

    # game objects
    targets = make_targets(2)
    target_curves = [target_points_for_display(t) for t in targets]

    # player init
    lon = random.uniform(-math.pi, math.pi)
    lat = random.uniform(-0.9, 0.9)
    x = normalize(np.array([math.cos(lat) * math.cos(lon),
                            math.cos(lat) * math.sin(lon),
                            math.sin(lat)], dtype=np.float64))
    east, north = tangent_basis(x)
    e1 = normalize(0.7 * east + 0.3 * north)

    heading_angle = 0.0
    kernel_angle = 0.0
    speed = 0.0

    score = 0
    combo = 0
    trail: List[Tuple[int, int]] = []

    running = True
    capture_threshold = 1.5  # tune for the chosen kernel/signal

    while running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_r:
                    targets = make_targets(2)
                    target_curves = [target_points_for_display(t) for t in targets]
                    score = 0
                    combo = 0
                    trail.clear()
                elif event.key in (pygame.K_1, pygame.K_KP1):
                    W, H = WINDOW_SIZES[1]
                    screen = pygame.display.set_mode((W, H))
                    grid = build_grid_surface(W, H)
                    trail.clear()
                elif event.key in (pygame.K_2, pygame.K_KP2):
                    W, H = WINDOW_SIZES[2]
                    screen = pygame.display.set_mode((W, H))
                    grid = build_grid_surface(W, H)
                    trail.clear()
                elif event.key in (pygame.K_3, pygame.K_KP3):
                    W, H = WINDOW_SIZES[3]
                    screen = pygame.display.set_mode((W, H))
                    grid = build_grid_surface(W, H)
                    trail.clear()

        keys = pygame.key.get_pressed()
        steer = 0.0
        if keys[pygame.K_LEFT]:
            steer -= 1.0
        if keys[pygame.K_RIGHT]:
            steer += 1.0
        heading_angle += steer * 1.9 * dt

        # one-button clockwise scan
        if keys[pygame.K_DOWN]:
            kernel_angle += 2.2 * dt

        # throttle
        if keys[pygame.K_UP]:
            speed = min(1.0, speed + 1.3 * dt)
        else:
            speed = max(0.0, speed - 0.7 * dt)

        # move step
        if speed > 1e-4:
            e2 = normalize(np.cross(x, e1))
            h = normalize(math.cos(heading_angle) * e1 + math.sin(heading_angle) * e2)
            step = speed * 0.55 * dt
            x2 = exp_map_s2(x, h, step)
            e1 = transport_on_s2(x, x2, e1)  # PTC transport of frame axis
            x = x2

        # compute ptc response
        resp = ptc_score(x, e1, kernel_angle, targets)

        # capture rule: if response high, award points and refresh one target
        if resp > capture_threshold:
            score += 10 + 3 * combo
            combo = min(combo + 1, 20)
            # respawn a random target to keep the game going
            idx = random.randrange(len(targets))
            targets[idx] = make_targets(1)[0]
            target_curves[idx] = target_points_for_display(targets[idx])
        else:
            combo = max(0, combo - 1)

        # render
        screen.blit(grid, (0, 0))

        # draw ridge targets (great circles)
        for curve_pts in target_curves:
            pts2d = []
            for p in curve_pts:
                lonp, latp = lonlat_from_unit(p)
                pts2d.append(screen_from_lonlat(lonp, latp, W, H))
            draw_polyline_wrapped(screen, pts2d, (34, 211, 238), width=2, w=W)

        # trail
        lonp, latp = lonlat_from_unit(x)
        px, py = screen_from_lonlat(lonp, latp, W, H)
        trail.append((px, py))
        if len(trail) > TRAIL_MAX_POINTS:
            trail = trail[-TRAIL_MAX_POINTS:]
        draw_polyline_wrapped(screen, trail, (52, 211, 153), width=2, w=W)

        # player dot
        pygame.draw.circle(screen, (250, 204, 21), (px, py), 7)

        # show transported axis e1 and kernel axis (rotated within tangent)
        east, north = tangent_basis(x)
        a1 = float(np.dot(e1, east))
        b1 = float(np.dot(e1, north))
        draw_arrow(
            screen,
            (px, py),
            (int(px + TANGENT_ARROW_LENGTH * a1), int(py - TANGENT_ARROW_LENGTH * b1)),
            (249, 115, 22),
            width=TANGENT_ARROW_WIDTH,
            head_len=TANGENT_ARROW_HEAD,
        )

        e2 = normalize(np.cross(x, e1))
        # control heading arrow (current steer direction in tangent plane)
        h_ctrl = normalize(math.cos(heading_angle) * e1 + math.sin(heading_angle) * e2)
        ah = float(np.dot(h_ctrl, east))
        bh = float(np.dot(h_ctrl, north))
        draw_arrow(screen, (px, py), (int(px + 54 * ah), int(py - 54 * bh)), (34, 197, 94), width=3, head_len=11)

        k1 = normalize(math.cos(kernel_angle) * e1 + math.sin(kernel_angle) * e2)
        ak = float(np.dot(k1, east))
        bk = float(np.dot(k1, north))
        draw_arrow(screen, (px, py), (int(px + 46 * ak), int(py - 46 * bk)), (147, 197, 253), width=3, head_len=10)

        # kernel footprint dots
        # build k2
        k2 = normalize(np.cross(x, k1))
        for u, v, w in KERNEL:
            y = exp_map_s2(x, u * k1 + v * k2, step=math.hypot(u, v))
            lony, laty = lonlat_from_unit(y)
            sx, sy = screen_from_lonlat(lony, laty, W, H)
            col = (248, 250, 252) if w > 0 else (253, 164, 175)
            pygame.draw.circle(screen, col, (sx, sy), 4)

        # HUD
        hud_x = 14
        hud_y = 12
        draw_text(screen, "Filter Hunter (PTC)", (hud_x, hud_y), (230, 235, 245), 26)
        hud_y += line_step(26, pad=8)
        draw_text(screen, f"Score: {score}   Combo: {combo}", (hud_x, hud_y), (253, 230, 138), 20)
        hud_y += line_step(20, pad=6)
        draw_text(
            screen,
            f"PTC response: {resp:+0.3f}   Capture if > {capture_threshold:.2f}",
            (hud_x, hud_y),
            (190, 210, 240),
            18,
        )
        hud_y += line_step(18, pad=6)
        draw_text(
            screen,
            "Orange: transported axis e1   Green: control heading   Blue: kernel scan axis",
            (hud_x, hud_y),
            (148, 163, 184),
            18,
        )
        controls_y = H - line_step(18, pad=10)
        draw_text(
            screen,
            "Controls: Up move, Left/Right steer, Down scan, 1/2/3 size, R respawn",
            (hud_x, controls_y),
            (148, 163, 184),
            18,
        )

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
