"""
PTC Demo 1: Spherical Holonomy Loop

What this demo does:
- Draws a sphere and a closed geodesic-like triangle loop on the sphere.
- Moves a small kernel pattern along the loop using parallel transport.
- Shows how the transported tangent axis changes after one full loop.

PTC interpretation:
- Parallel transport convolution aligns the kernel at each point by transporting
  the local tangent frame along the path.
- On curved surfaces, returning to the start can rotate the frame (holonomy),
  which this scene visualizes as a non-zero final angle.

Render:
- conda run -n ptc manim -ql ptc_demo_spherical_holonomy_manim.py PTCSphericalHolonomyDemo
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


def slerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    dot_ab = float(np.clip(np.dot(a, b), -1.0, 1.0))
    omega = float(np.arccos(dot_ab))
    if omega < 1e-8:
        return normalize((1.0 - t) * a + t * b)
    so = np.sin(omega)
    return (np.sin((1.0 - t) * omega) / so) * a + (np.sin(t * omega) / so) * b


def build_spherical_triangle_loop(radius: float, samples_per_edge: int) -> np.ndarray:
    unit_vertices = [
        normalize(np.array([0.16, -0.82, 0.55], dtype=np.float64)),
        normalize(np.array([0.95, 0.23, 0.21], dtype=np.float64)),
        normalize(np.array([-0.38, 0.81, 0.44], dtype=np.float64)),
    ]

    path: list[np.ndarray] = []
    edge_pairs = [(0, 1), (1, 2), (2, 0)]
    for edge_idx, (i, j) in enumerate(edge_pairs):
        t_values = np.linspace(0.0, 1.0, samples_per_edge, endpoint=(edge_idx == len(edge_pairs) - 1))
        for t in t_values:
            path.append(radius * slerp(unit_vertices[i], unit_vertices[j], float(t)))
    return np.asarray(path, dtype=np.float64)


def transport_frame_along_path(path_points: np.ndarray, radius: float) -> np.ndarray:
    normals = path_points / radius
    n = len(path_points)
    e1 = np.zeros((n, 3), dtype=np.float64)

    seed = path_points[1] - path_points[0]
    seed -= np.dot(seed, normals[0]) * normals[0]
    e1[0] = normalize(seed)

    for i in range(1, n):
        transported = rotate_with_normal_change(e1[i - 1], normals[i - 1], normals[i])
        transported -= np.dot(transported, normals[i]) * normals[i]
        e1[i] = normalize(transported)
    return e1


def interpolate_state(path_points: np.ndarray, e1_path: np.ndarray, progress: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = float(np.clip(progress, 0.0, 1.0)) * (len(path_points) - 1)
    i0 = min(int(np.floor(u)), len(path_points) - 1)
    i1 = min(i0 + 1, len(path_points) - 1)
    t = u - i0

    center = (1.0 - t) * path_points[i0] + t * path_points[i1]
    normal = normalize(center)
    e1 = (1.0 - t) * e1_path[i0] + t * e1_path[i1]
    e1 -= np.dot(e1, normal) * normal
    e1 = normalize(e1)
    e2 = normalize(np.cross(normal, e1))
    return center, e1, e2


class PTCSphericalHolonomyDemo(ThreeDScene):
    """Visualize transported kernel orientation and holonomy on a sphere."""

    def construct(self) -> None:
        radius = 2.54
        path_points = build_spherical_triangle_loop(radius=radius, samples_per_edge=96)
        e1_path = transport_frame_along_path(path_points, radius=radius)

        self.set_camera_orientation(phi=67 * DEGREES, theta=-35 * DEGREES)

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
            fill_color="#0F1A2E",
            fill_opacity=0.42,
            stroke_color="#3B82F6",
            stroke_opacity=0.24,
            stroke_width=1.7,
        )

        loop_curve = VMobject(color="#34D399", stroke_width=4.9, stroke_opacity=0.95)
        loop_curve.set_points_as_corners(path_points)

        progress = ValueTracker(0.0)

        kernel_offsets = [
            (0.0, 0.0, "#F8FAFC"),
            (0.286, 0.0, "#22D3EE"),
            (-0.286, 0.0, "#22D3EE"),
            (0.0, 0.286, "#38BDF8"),
            (0.0, -0.286, "#38BDF8"),
            (0.208, 0.208, "#60A5FA"),
            (-0.208, -0.208, "#60A5FA"),
        ]

        center_start, e1_start, e2_start = interpolate_state(path_points, e1_path, 0.0)

        center_dot = Dot3D(point=center_start, radius=0.081, color="#FACC15")
        center_dot.add_updater(
            lambda mob: mob.move_to(interpolate_state(path_points, e1_path, progress.get_value())[0])
        )

        live_axis = Line(center_start, center_start + 0.81 * e1_start, color="#F97316", stroke_width=7.0)

        def update_live_axis(mob: Line) -> None:
            center, e1, _ = interpolate_state(path_points, e1_path, progress.get_value())
            mob.put_start_and_end_on(center, center + 0.81 * e1)

        live_axis.add_updater(update_live_axis)

        ref_axis = Line(center_start, center_start + 0.81 * e1_start, color="#9CA3AF", stroke_width=6.0)
        ref_axis.set_stroke(opacity=0.78)

        kernel_dots = []
        for u_local, v_local, color_val in kernel_offsets:
            sample_point = exp_map_sphere(
                center_start,
                u_local * e1_start + v_local * e2_start,
                radius=radius,
            )
            dot = Dot3D(point=sample_point, radius=0.052, color=color_val)

            def updater(mob: Dot3D, au=u_local, av=v_local) -> None:
                center, e1, e2 = interpolate_state(path_points, e1_path, progress.get_value())
                moved = exp_map_sphere(center, au * e1 + av * e2, radius=radius)
                mob.move_to(moved)

            dot.add_updater(updater)
            kernel_dots.append(dot)

        self.add(sphere)
        self.play(Create(loop_curve), run_time=1.2)
        self.play(
            FadeIn(center_dot),
            FadeIn(ref_axis),
            FadeIn(live_axis),
            *[FadeIn(dot) for dot in kernel_dots],
            run_time=0.9,
        )
        self.play(progress.animate.set_value(1.0), run_time=10.0, rate_func=linear)

        n0 = normalize(path_points[0])
        e1_final = e1_path[-1] - np.dot(e1_path[-1], n0) * n0
        e1_final = normalize(e1_final)
        signed = np.degrees(
            np.arctan2(
                np.dot(n0, np.cross(e1_start, e1_final)),
                float(np.clip(np.dot(e1_start, e1_final), -1.0, 1.0)),
            )
        )
        holonomy_deg = abs(signed)

        holonomy_text = Text(f"Holonomy ≈ {holonomy_deg:.1f} deg", font_size=30, color="#FDE047")
        holonomy_text.to_corner(UR).shift(np.array([-0.34, -0.18, 0.0], dtype=np.float64))
        self.add_fixed_in_frame_mobjects(holonomy_text)
        self.play(FadeIn(holonomy_text), run_time=0.7)
        self.wait(0.8)
