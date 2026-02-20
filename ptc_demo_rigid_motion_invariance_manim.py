"""
PTC Demo 3: Rigid Motion Invariance

What this demo does:
- Builds one intrinsic loop and signal on a left sphere.
- Creates a right sphere that is a rigidly rotated copy in 3D space.
- Transports the same kernel along corresponding loops on both spheres.
- Plots both convolution response traces on a shared inset graph.

PTC interpretation:
- If geometry and signal are matched by an isometry (here, rigid rotation),
  intrinsic parallel transport convolution responses should match.
- The side-by-side traces illustrate this invariance/equivariance behavior.

Render:
- conda run -n ptc manim -ql ptc_demo_rigid_motion_invariance_manim.py PTCRigidMotionInvarianceDemo
"""

from __future__ import annotations

import numpy as np
from manim import (
    Axes,
    Create,
    DEGREES,
    DOWN,
    Dot3D,
    FadeIn,
    Line,
    PI,
    RIGHT,
    Surface,
    TAU,
    Text,
    ThreeDScene,
    ValueTracker,
    VMobject,
    config,
    linear,
)

config.pixel_width = 1440
config.pixel_height = 1080
config.frame_rate = 30
config.background_color = "#070B14"


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


def build_intrinsic_loop(sample_count: int) -> np.ndarray:
    t = np.linspace(0.0, TAU, sample_count, dtype=np.float64)
    raw = np.stack(
        [
            np.cos(t) + 0.18 * np.cos(2.0 * t),
            0.78 * np.sin(t),
            0.38 * np.sin(2.0 * t) + 0.34,
        ],
        axis=1,
    )
    return raw / np.linalg.norm(raw, axis=1, keepdims=True)


def transport_tangent_frame(normals: np.ndarray) -> np.ndarray:
    n = len(normals)
    e1 = np.zeros((n, 3), dtype=np.float64)

    seed = normals[1] - normals[0]
    seed -= np.dot(seed, normals[0]) * normals[0]
    e1[0] = normalize(seed)

    for i in range(1, n):
        transported = rotate_with_normal_change(e1[i - 1], normals[i - 1], normals[i])
        transported -= np.dot(transported, normals[i]) * normals[i]
        e1[i] = normalize(transported)
    return e1


SIGNAL_COMPONENTS: list[tuple[np.ndarray, float, float, str]] = [
    (normalize(np.array([0.78, -0.44, 0.44], dtype=np.float64)), 1.00, 0.34, "#22D3EE"),
    (normalize(np.array([-0.56, 0.70, 0.44], dtype=np.float64)), 0.84, 0.30, "#60A5FA"),
    (normalize(np.array([0.06, 0.96, -0.29], dtype=np.float64)), -0.52, 0.27, "#FB7185"),
]

KERNEL_WEIGHTS: list[tuple[float, float, float]] = [
    (0.26, 0.00, 1.00),
    (0.12, 0.13, 0.62),
    (0.12, -0.13, 0.62),
    (-0.20, 0.00, -0.70),
]

KERNEL_DISPLAY: list[tuple[float, float, str]] = [
    (0.26, 0.00, "#F8FAFC"),
    (0.12, 0.13, "#A5F3FC"),
    (0.12, -0.13, "#A5F3FC"),
    (-0.20, 0.00, "#FDA4AF"),
]


def intrinsic_signal(normal: np.ndarray) -> float:
    value = 0.0
    for center, weight, sigma, _ in SIGNAL_COMPONENTS:
        angle = float(np.arccos(np.clip(np.dot(normal, center), -1.0, 1.0)))
        value += weight * np.exp(-0.5 * (angle / sigma) ** 2)
    return value


def convolution_response(normal: np.ndarray, e1: np.ndarray, signal_fn) -> float:
    e2 = normalize(np.cross(normal, e1))
    total = 0.0
    for u_local, v_local, k_weight in KERNEL_WEIGHTS:
        sample_normal = exp_map_sphere(normal, u_local * e1 + v_local * e2, radius=1.0)
        total += k_weight * signal_fn(normalize(sample_normal))
    return total


def interpolate_state(normals: np.ndarray, e1_path: np.ndarray, progress: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = float(np.clip(progress, 0.0, len(normals) - 1))
    i0 = min(int(np.floor(u)), len(normals) - 1)
    i1 = min(i0 + 1, len(normals) - 1)
    t = u - i0

    normal = normalize((1.0 - t) * normals[i0] + t * normals[i1])
    e1 = (1.0 - t) * e1_path[i0] + t * e1_path[i1]
    e1 -= np.dot(e1, normal) * normal
    e1 = normalize(e1)
    e2 = normalize(np.cross(normal, e1))
    return normal, e1, e2


class PTCRigidMotionInvarianceDemo(ThreeDScene):
    """Compare synchronized PTC responses on rigidly related surfaces."""

    def construct(self) -> None:
        radius = 1.50
        left_center = np.array([-3.45, 0.80, 0.55], dtype=np.float64)
        right_center = np.array([3.45, 0.80, 0.55], dtype=np.float64)

        rotation_q = rodrigues_rotation(np.array([0.33, 0.86, 0.39], dtype=np.float64), 0.90)

        left_normals = build_intrinsic_loop(sample_count=220)
        left_e1 = transport_tangent_frame(left_normals)
        right_normals = (rotation_q @ left_normals.T).T
        right_e1 = (rotation_q @ left_e1.T).T

        left_points = left_center + radius * left_normals
        right_points = right_center + radius * right_normals

        score_left = np.asarray(
            [convolution_response(left_normals[i], left_e1[i], intrinsic_signal) for i in range(len(left_normals))]
        )
        score_right = np.asarray(
            [
                convolution_response(
                    right_normals[i],
                    right_e1[i],
                    lambda n, q=rotation_q: intrinsic_signal(q.T @ n),
                )
                for i in range(len(right_normals))
            ]
        )

        self.set_camera_orientation(phi=66 * DEGREES, theta=-90 * DEGREES)

        def sphere_surface(center: np.ndarray, fill_color: str, stroke_color: str) -> Surface:
            return Surface(
                lambda u, v: center
                + radius
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
                resolution=(20, 40),
                fill_color=fill_color,
                fill_opacity=0.40,
                stroke_color=stroke_color,
                stroke_opacity=0.24,
                stroke_width=1.6,
            )

        left_surface = sphere_surface(left_center, "#0F172A", "#3B82F6")
        right_surface = sphere_surface(right_center, "#111827", "#60A5FA")

        left_curve = VMobject(color="#22D3EE", stroke_width=4.4, stroke_opacity=0.90)
        left_curve.set_points_as_corners(left_points)
        right_curve = VMobject(color="#F97316", stroke_width=4.4, stroke_opacity=0.90)
        right_curve.set_points_as_corners(right_points)

        left_signal_dots = []
        right_signal_dots = []
        for center_dir, weight, _, color_val in SIGNAL_COMPONENTS:
            left_dot = Dot3D(point=left_center + radius * center_dir, radius=0.111, color=color_val)
            right_dot = Dot3D(point=right_center + radius * (rotation_q @ center_dir), radius=0.111, color=color_val)
            left_dot.set_opacity(0.95 if weight > 0.0 else 0.80)
            right_dot.set_opacity(0.95 if weight > 0.0 else 0.80)
            left_signal_dots.append(left_dot)
            right_signal_dots.append(right_dot)

        progress = ValueTracker(0.0)

        left_normal0, left_e10, left_e20 = interpolate_state(left_normals, left_e1, 0.0)
        right_normal0, right_e10, right_e20 = interpolate_state(right_normals, right_e1, 0.0)

        left_center_dot = Dot3D(point=left_center + radius * left_normal0, radius=0.072, color="#FACC15")
        right_center_dot = Dot3D(point=right_center + radius * right_normal0, radius=0.072, color="#FACC15")

        left_center_dot.add_updater(
            lambda mob: mob.move_to(
                left_center + radius * interpolate_state(left_normals, left_e1, progress.get_value())[0]
            )
        )
        right_center_dot.add_updater(
            lambda mob: mob.move_to(
                right_center + radius * interpolate_state(right_normals, right_e1, progress.get_value())[0]
            )
        )

        left_axis = Line(
            left_center + radius * left_normal0,
            left_center + radius * left_normal0 + 0.62 * left_e10,
            color="#22D3EE",
            stroke_width=6.5,
        )
        right_axis = Line(
            right_center + radius * right_normal0,
            right_center + radius * right_normal0 + 0.62 * right_e10,
            color="#F97316",
            stroke_width=6.5,
        )

        def update_left_axis(mob: Line) -> None:
            normal, e1, _ = interpolate_state(left_normals, left_e1, progress.get_value())
            start = left_center + radius * normal
            mob.put_start_and_end_on(start, start + 0.62 * e1)

        def update_right_axis(mob: Line) -> None:
            normal, e1, _ = interpolate_state(right_normals, right_e1, progress.get_value())
            start = right_center + radius * normal
            mob.put_start_and_end_on(start, start + 0.62 * e1)

        left_axis.add_updater(update_left_axis)
        right_axis.add_updater(update_right_axis)

        left_kernel = []
        for u_local, v_local, color_val in KERNEL_DISPLAY:
            sample_n = exp_map_sphere(left_normal0, u_local * left_e10 + v_local * left_e20, radius=1.0)
            dot = Dot3D(point=left_center + radius * sample_n, radius=0.044, color=color_val)

            def updater(mob: Dot3D, au=u_local, av=v_local) -> None:
                normal, e1, e2 = interpolate_state(left_normals, left_e1, progress.get_value())
                moved = exp_map_sphere(normal, au * e1 + av * e2, radius=1.0)
                mob.move_to(left_center + radius * moved)

            dot.add_updater(updater)
            left_kernel.append(dot)

        right_kernel = []
        for u_local, v_local, color_val in KERNEL_DISPLAY:
            sample_n = exp_map_sphere(right_normal0, u_local * right_e10 + v_local * right_e20, radius=1.0)
            dot = Dot3D(point=right_center + radius * sample_n, radius=0.044, color=color_val)

            def updater(mob: Dot3D, au=u_local, av=v_local) -> None:
                normal, e1, e2 = interpolate_state(right_normals, right_e1, progress.get_value())
                moved = exp_map_sphere(normal, au * e1 + av * e2, radius=1.0)
                mob.move_to(right_center + radius * moved)

            dot.add_updater(updater)
            right_kernel.append(dot)

        x_vals = np.linspace(0.0, 1.0, len(score_left))
        y_min = float(min(score_left.min(), score_right.min()) - 0.12)
        y_max = float(max(score_left.max(), score_right.max()) + 0.12)

        axes = Axes(
            x_range=[0.0, 1.0, 0.25],
            y_range=[y_min, y_max, max((y_max - y_min) / 4.0, 0.1)],
            x_length=5.2,
            y_length=2.40,
            axis_config={"stroke_color": "#94A3B8", "stroke_width": 2.6},
            tips=False,
        )
        axes.to_edge(DOWN).shift(0.06 * np.array([0.0, 1.0, 0.0], dtype=np.float64))

        graph_label = Text("PTC Response", font_size=27, color="#E2E8F0")
        graph_label.next_to(axes, np.array([0.0, 1.0, 0.0], dtype=np.float64), buff=0.10)

        line_left = VMobject(color="#22D3EE", stroke_width=3.6, stroke_opacity=0.96)
        line_right = VMobject(color="#F97316", stroke_width=3.6, stroke_opacity=0.96)

        def update_trace(mob: VMobject, values: np.ndarray) -> None:
            u = float(np.clip(progress.get_value(), 0.0, len(values) - 1))
            i0 = min(int(np.floor(u)), len(values) - 1)
            i1 = min(i0 + 1, len(values) - 1)
            t = u - i0

            xs = list(x_vals[: i0 + 1])
            ys = list(values[: i0 + 1])
            xs.append((1.0 - t) * x_vals[i0] + t * x_vals[i1])
            ys.append((1.0 - t) * values[i0] + t * values[i1])

            pts = [axes.c2p(float(x), float(y)) for x, y in zip(xs, ys)]
            if len(pts) < 2:
                pts = [pts[0], pts[0] + 1e-4 * RIGHT]
            mob.set_points_as_corners(np.asarray(pts))

        line_left.add_updater(lambda mob: update_trace(mob, score_left))
        line_right.add_updater(lambda mob: update_trace(mob, score_right))
        update_trace(line_left, score_left)
        update_trace(line_right, score_right)

        self.add_fixed_in_frame_mobjects(
            axes,
            graph_label,
            line_left,
            line_right,
        )

        self.add(left_surface, right_surface)
        self.play(Create(left_curve), Create(right_curve), run_time=1.2)
        self.play(
            *[FadeIn(dot) for dot in left_signal_dots + right_signal_dots],
            FadeIn(left_center_dot),
            FadeIn(right_center_dot),
            FadeIn(left_axis),
            FadeIn(right_axis),
            *[FadeIn(dot) for dot in left_kernel + right_kernel],
            run_time=0.9,
        )
        self.play(progress.animate.set_value(len(left_normals) - 1), run_time=11.0, rate_func=linear)
        self.wait(0.6)
