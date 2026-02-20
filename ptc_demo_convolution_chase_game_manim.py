"""
PTC Demo 2: Convolution Chase Game

What this demo does:
- Defines a scalar signal on a sphere with attractive and repulsive hotspots.
- Moves a kernel over the sphere in discrete steps.
- At each step, picks the heading with the highest convolution score.
- Draws the visited trajectory trail of the greedy walk.

PTC interpretation:
- The kernel lives in a transported tangent frame at the current center point.
- Each score is an intrinsic convolution sample of the signal with that local
  transported kernel, giving a game-like "follow the response peak" behavior.

Render:
- conda run -n ptc manim -ql ptc_demo_convolution_chase_game_manim.py PTCConvolutionChaseGame
"""

from __future__ import annotations

import numpy as np
from manim import (
    Create,
    DEGREES,
    Dot3D,
    FadeIn,
    Line,
    PI,
    Surface,
    TAU,
    Text,
    ThreeDScene,
    UR,
    ValueTracker,
    VMobject,
    config,
    linear,
)

config.pixel_width = 1440
config.pixel_height = 1080
config.frame_rate = 30
config.background_color = "#050A12"


def normalize(vec: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(vec)
    if nrm < 1e-10:
        return vec
    return vec / nrm


def rodrigues_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = normalize(axis)
    x, y, z = axis
    k = np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )
    eye = np.eye(3, dtype=np.float64)
    return eye + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def rotate_with_normal_change(vec: np.ndarray, n_prev: np.ndarray, n_cur: np.ndarray) -> np.ndarray:
    cross_val = np.cross(n_prev, n_cur)
    c_norm = np.linalg.norm(cross_val)
    if c_norm < 1e-10:
        return vec.copy()
    dot_val = float(np.clip(np.dot(n_prev, n_cur), -1.0, 1.0))
    angle = float(np.arctan2(c_norm, dot_val))
    return rodrigues_rotation(cross_val, angle) @ vec


def exp_map_sphere(point: np.ndarray, tangent_vec: np.ndarray, radius: float) -> np.ndarray:
    tangent_norm = np.linalg.norm(tangent_vec)
    if tangent_norm < 1e-10:
        return point
    angle = tangent_norm / radius
    return np.cos(angle) * point + (radius * np.sin(angle) / tangent_norm) * tangent_vec


HOTSPOTS: list[tuple[np.ndarray, float, float, str]] = [
    (normalize(np.array([0.72, -0.48, 0.50], dtype=np.float64)), 1.15, 0.34, "#22D3EE"),
    (normalize(np.array([-0.46, 0.67, 0.58], dtype=np.float64)), 0.92, 0.30, "#60A5FA"),
    (normalize(np.array([0.08, 0.94, -0.33], dtype=np.float64)), -0.62, 0.27, "#FB7185"),
]

KERNEL_SAMPLES: list[tuple[float, float, float, str]] = [
    (0.30, 0.00, 1.00, "#F8FAFC"),
    (0.16, 0.14, 0.60, "#A5F3FC"),
    (0.16, -0.14, 0.60, "#A5F3FC"),
    (-0.22, 0.00, -0.72, "#FDA4AF"),
]


def signal_value(normal: np.ndarray) -> float:
    value = 0.0
    for center, weight, sigma, _ in HOTSPOTS:
        angle = float(np.arccos(np.clip(np.dot(normal, center), -1.0, 1.0)))
        value += weight * np.exp(-0.5 * (angle / sigma) ** 2)
    return value


def convolution_score(center: np.ndarray, e1: np.ndarray, e2: np.ndarray, radius: float) -> float:
    total = 0.0
    for u_local, v_local, k_weight, _ in KERNEL_SAMPLES:
        sample_point = exp_map_sphere(center, u_local * e1 + v_local * e2, radius=radius)
        total += k_weight * signal_value(normalize(sample_point))
    return total


def build_chase_states(radius: float, step_size: float, steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points: list[np.ndarray] = []
    e1_values: list[np.ndarray] = []
    scores: list[float] = []

    current_point = radius * normalize(np.array([-0.78, -0.36, 0.51], dtype=np.float64))
    current_normal = normalize(current_point)
    seed = np.array([0.95, 0.18, 0.07], dtype=np.float64)
    current_e1 = seed - np.dot(seed, current_normal) * current_normal
    current_e1 = normalize(current_e1)

    for _ in range(steps + 1):
        current_e2 = normalize(np.cross(current_normal, current_e1))
        points.append(current_point.copy())
        e1_values.append(current_e1.copy())
        scores.append(convolution_score(current_point, current_e1, current_e2, radius=radius))

        candidate_angles = np.array([-0.95, -0.60, -0.25, 0.0, 0.25, 0.60, 0.95], dtype=np.float64)
        best_value = -1e9
        best_state: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

        for angle in candidate_angles:
            heading = normalize(np.cos(angle) * current_e1 + np.sin(angle) * current_e2)
            next_point = exp_map_sphere(current_point, step_size * heading, radius=radius)
            next_normal = normalize(next_point)
            next_e1 = rotate_with_normal_change(current_e1, current_normal, next_normal)
            next_e1 -= np.dot(next_e1, next_normal) * next_normal
            next_e1 = normalize(next_e1)
            next_e2 = normalize(np.cross(next_normal, next_e1))

            score = convolution_score(next_point, next_e1, next_e2, radius=radius) + 0.045 * np.cos(angle)
            if score > best_value:
                best_value = score
                best_state = (next_point, next_normal, next_e1)

        if best_state is None:
            break
        current_point, current_normal, current_e1 = best_state

    return np.asarray(points), np.asarray(e1_values), np.asarray(scores)


def interpolate_state(
    points: np.ndarray,
    e1_path: np.ndarray,
    scores: np.ndarray,
    progress: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    u = float(np.clip(progress, 0.0, len(points) - 1))
    i0 = min(int(np.floor(u)), len(points) - 1)
    i1 = min(i0 + 1, len(points) - 1)
    t = u - i0

    center = (1.0 - t) * points[i0] + t * points[i1]
    normal = normalize(center)
    e1 = (1.0 - t) * e1_path[i0] + t * e1_path[i1]
    e1 -= np.dot(e1, normal) * normal
    e1 = normalize(e1)
    e2 = normalize(np.cross(normal, e1))
    score = float((1.0 - t) * scores[i0] + t * scores[i1])
    return center, e1, e2, score


class PTCConvolutionChaseGame(ThreeDScene):
    """Greedy kernel walk driven by local parallel transport convolution score."""

    def construct(self) -> None:
        radius = 2.47
        points, e1_path, scores = build_chase_states(radius=radius, step_size=0.286, steps=28)

        self.set_camera_orientation(phi=66 * DEGREES, theta=-44 * DEGREES)

        sphere = Surface(
            lambda u, v: radius
            * np.array(
                [
                    np.cos(u) * np.cos(v),
                    np.cos(u) * np.sin(v),
                    np.sin(u),
                ],
                dtype=np.float64,
            ),
            u_range=[-PI / 2.0, PI / 2.0],
            v_range=[0.0, TAU],
            resolution=(22, 44),
            fill_color="#101C31",
            fill_opacity=0.40,
            stroke_color="#1D4ED8",
            stroke_opacity=0.22,
            stroke_width=1.6,
        )

        hotspot_dots = []
        for center, weight, _, color_val in HOTSPOTS:
            dot_radius = 0.143 if weight > 0.0 else 0.111
            hotspot = Dot3D(point=radius * center, radius=dot_radius, color=color_val)
            hotspot.set_opacity(0.95 if weight > 0.0 else 0.82)
            hotspot_dots.append(hotspot)

        progress = ValueTracker(0.0)

        score_label = Text("Score", font_size=26, color="#F8FAFC")
        score_label.to_corner(UR).shift(
            -0.28 * np.array([1.0, 0.0, 0.0], dtype=np.float64)
            + 0.14 * np.array([0.0, -1.0, 0.0], dtype=np.float64)
        )
        self.add_fixed_in_frame_mobjects(score_label)

        score_value = Text(f"{scores[0]:+0.3f}", font_size=28, color="#FDE047")
        score_value.next_to(
            score_label,
            np.array([0.0, -1.0, 0.0], dtype=np.float64),
            buff=0.12,
            aligned_edge=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        )

        def update_score_value(mob: Text) -> None:
            score = interpolate_state(points, e1_path, scores, progress.get_value())[3]
            new_text = Text(f"{score:+0.3f}", font_size=28, color="#FDE047")
            new_text.move_to(mob.get_center())
            mob.become(new_text)

        score_value.add_updater(update_score_value)
        self.add_fixed_in_frame_mobjects(score_value)

        step_label = Text("Step", font_size=24, color="#E2E8F0")
        step_label.next_to(
            score_value,
            np.array([0.0, -1.0, 0.0], dtype=np.float64),
            buff=0.16,
            aligned_edge=np.array([1.0, 0.0, 0.0], dtype=np.float64),
        )
        self.add_fixed_in_frame_mobjects(step_label)

        step_value = Text("0", font_size=27, color="#93C5FD")
        step_value.next_to(step_label, np.array([1.0, 0.0, 0.0], dtype=np.float64), buff=0.18)

        def update_step_value(mob: Text) -> None:
            step = int(np.floor(progress.get_value()))
            new_text = Text(f"{step:d}", font_size=27, color="#93C5FD")
            new_text.move_to(mob.get_center())
            mob.become(new_text)

        step_value.add_updater(update_step_value)
        self.add_fixed_in_frame_mobjects(step_value)

        center0, e1_0, e2_0, _ = interpolate_state(points, e1_path, scores, 0.0)
        center_dot = Dot3D(point=center0, radius=0.081, color="#FACC15")
        center_dot.add_updater(
            lambda mob: mob.move_to(interpolate_state(points, e1_path, scores, progress.get_value())[0])
        )

        heading = Line(center0, center0 + 0.73 * e1_0, color="#F97316", stroke_width=7.0)

        def update_heading(mob: Line) -> None:
            center, e1, _, _ = interpolate_state(points, e1_path, scores, progress.get_value())
            mob.put_start_and_end_on(center, center + 0.73 * e1)

        heading.add_updater(update_heading)

        kernel_dots = []
        for u_local, v_local, _, color_val in KERNEL_SAMPLES:
            sample = exp_map_sphere(center0, u_local * e1_0 + v_local * e2_0, radius=radius)
            dot = Dot3D(point=sample, radius=0.049, color=color_val)

            def updater(mob: Dot3D, au=u_local, av=v_local) -> None:
                center, e1, e2, _ = interpolate_state(points, e1_path, scores, progress.get_value())
                moved = exp_map_sphere(center, au * e1 + av * e2, radius=radius)
                mob.move_to(moved)

            dot.add_updater(updater)
            kernel_dots.append(dot)

        trail = VMobject(color="#34D399", stroke_width=4.4, stroke_opacity=0.96)

        def update_trail(mob: VMobject) -> None:
            u = float(np.clip(progress.get_value(), 0.0, len(points) - 1))
            i0 = min(int(np.floor(u)), len(points) - 1)
            i1 = min(i0 + 1, len(points) - 1)
            t = u - i0

            trail_points = [pt.copy() for pt in points[: i0 + 1]]
            interp = (1.0 - t) * points[i0] + t * points[i1]
            trail_points.append(interp)
            if len(trail_points) < 2:
                trail_points.append(trail_points[0] + np.array([1e-4, 0.0, 0.0], dtype=np.float64))
            mob.set_points_as_corners(np.asarray(trail_points))

        trail.add_updater(update_trail)
        update_trail(trail)

        self.add(sphere)
        self.play(*[FadeIn(dot) for dot in hotspot_dots], run_time=0.8)
        self.play(
            FadeIn(center_dot),
            FadeIn(heading),
            FadeIn(trail),
            *[FadeIn(dot) for dot in kernel_dots],
            run_time=0.8,
        )
        self.play(progress.animate.set_value(len(points) - 1), run_time=11.0, rate_func=linear)
        self.wait(0.6)
