from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from manim import (
    Circle,
    Create,
    DEGREES,
    Dot,
    FadeIn,
    ParametricFunction,
    Polygon,
    Scene,
    Surface,
    TAU,
    ThreeDScene,
    ValueTracker,
    VGroup,
    VMobject,
    config,
    linear,
)

config.pixel_width = 1920
config.pixel_height = 1080
config.frame_rate = 30
config.background_color = "#0A0F14"


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < 1e-10:
        return vec
    return vec / norm


def rodrigues_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = normalize(axis)
    x, y, z = axis
    cross_matrix = np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )
    identity = np.eye(3, dtype=np.float64)
    return identity + np.sin(angle) * cross_matrix + (1.0 - np.cos(angle)) * (cross_matrix @ cross_matrix)


def exp_map_sphere(point: np.ndarray, tangent_vec: np.ndarray, radius: float) -> np.ndarray:
    tangent_norm = np.linalg.norm(tangent_vec)
    if tangent_norm < 1e-10:
        return point
    angle = tangent_norm / radius
    return np.cos(angle) * point + (radius * np.sin(angle) / tangent_norm) * tangent_vec


def plasma_rgb(z: torch.Tensor) -> torch.Tensor:
    stops = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float32, device=z.device)
    colors = torch.tensor(
        [
            [13, 8, 135],
            [126, 3, 167],
            [204, 71, 120],
            [248, 149, 64],
            [240, 249, 33],
        ],
        dtype=torch.float32,
        device=z.device,
    ) / 255.0

    idx = torch.bucketize(z, stops[1:-1])
    s0 = stops[idx]
    s1 = stops[idx + 1]
    c0 = colors[idx]
    c1 = colors[idx + 1]
    t = ((z - s0) / (s1 - s0 + 1e-8)).unsqueeze(-1)
    return c0 + t * (c1 - c0)


@dataclass
class TransportModel:
    vertices: np.ndarray
    project_to_chart: Callable[[np.ndarray], np.ndarray]
    geodesic_point: Callable[[int, float], np.ndarray]
    lifted_kernel_point: Callable[[float, float, float], np.ndarray]


def build_transport_model(radius: float, chart_radius: float, chart_center: np.ndarray) -> TransportModel:
    def project_to_chart(p: np.ndarray) -> np.ndarray:
        return chart_center + np.array([chart_radius * p[0] / radius, chart_radius * p[1] / radius, 0.0])

    view_xy_dir = normalize(np.array([np.cos(-np.pi / 4.0), np.sin(-np.pi / 4.0)], dtype=np.float64))
    center_xy = 1.62 * view_xy_dir
    tri_r = 0.98
    alpha = np.deg2rad(22.0)

    vertex_xy = np.stack(
        [
            center_xy + tri_r * np.array([np.cos(alpha + 2.0 * np.pi * k / 3.0), np.sin(alpha + 2.0 * np.pi * k / 3.0)])
            for k in range(3)
        ],
        axis=0,
    )

    vertices = []
    for i in range(3):
        x, y = float(vertex_xy[i, 0]), float(vertex_xy[i, 1])
        z = np.sqrt(max(radius * radius - x * x - y * y, 0.0))
        vertices.append(np.array([x, y, z], dtype=np.float64))
    vertices = np.stack(vertices, axis=0)

    for i in range(3):
        recovered_xy = radius * (project_to_chart(vertices[i])[:2] - chart_center[:2]) / chart_radius
        if not np.allclose(recovered_xy, vertices[i][:2], atol=1e-9):
            raise ValueError("Projection inverse consistency check failed for a triangle vertex.")

    unit_vertices = vertices / radius
    seg_axes: list[np.ndarray] = []
    seg_angles: list[float] = []
    for i in range(3):
        j = (i + 1) % 3
        axis = normalize(np.cross(unit_vertices[i], unit_vertices[j]))
        angle = float(np.arccos(np.clip(np.dot(unit_vertices[i], unit_vertices[j]), -1.0, 1.0)))
        seg_axes.append(axis)
        seg_angles.append(angle)

    def geodesic_point(seg_idx: int, t: float) -> np.ndarray:
        rotation = rodrigues_rotation(seg_axes[seg_idx], t * seg_angles[seg_idx])
        return radius * (rotation @ unit_vertices[seg_idx])

    e1_starts: list[np.ndarray] = []
    e2_starts: list[np.ndarray] = []
    e1_0 = normalize(np.cross(seg_axes[0], unit_vertices[0]))
    e2_0 = normalize(np.cross(unit_vertices[0], e1_0))
    e1_starts.append(e1_0)
    e2_starts.append(e2_0)

    for i in range(2):
        rotation_end = rodrigues_rotation(seg_axes[i], seg_angles[i])
        end_dir = unit_vertices[(i + 1) % 3]
        e1_end = rotation_end @ e1_starts[i]
        e1_end -= np.dot(e1_end, end_dir) * end_dir
        e1_end = normalize(e1_end)
        e2_end = normalize(np.cross(end_dir, e1_end))
        e1_starts.append(e1_end)
        e2_starts.append(e2_end)

    def state_at(path_progress: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        clamped = min(max(path_progress, 0.0), 3.0)
        seg = min(int(np.floor(clamped)), 2)
        local_t = clamped - seg

        rotation = rodrigues_rotation(seg_axes[seg], local_t * seg_angles[seg])
        point_dir = rotation @ unit_vertices[seg]
        e1 = normalize(rotation @ e1_starts[seg])
        e2 = normalize(rotation @ e2_starts[seg])
        point = radius * point_dir
        return point, e1, e2

    def lifted_kernel_point(u_local: float, v_local: float, p_state: float) -> np.ndarray:
        center, e1, e2 = state_at(p_state)
        tangent = u_local * e1 + v_local * e2
        return exp_map_sphere(center, tangent, radius)

    return TransportModel(
        vertices=vertices,
        project_to_chart=project_to_chart,
        geodesic_point=geodesic_point,
        lifted_kernel_point=lifted_kernel_point,
    )


def iter_kernel_cells() -> list[tuple[float, float, float, float, str]]:
    kernel_extent = 0.48
    kernel_grid_size = 37
    us = torch.linspace(-kernel_extent, kernel_extent, kernel_grid_size)
    vs = torch.linspace(-kernel_extent, kernel_extent, kernel_grid_size)

    u_centers = 0.5 * (us[:-1] + us[1:])
    v_centers = 0.5 * (vs[:-1] + vs[1:])
    vv_c, uu_c = torch.meshgrid(v_centers, u_centers, indexing="ij")
    local_centers = torch.stack([uu_c, vv_c], dim=-1)

    offsets = torch.tensor(
        [
            [0.0, 0.18],
            [-0.1559, -0.09],
            [0.1559, -0.09],
        ],
        dtype=torch.float32,
    )
    sigmas = torch.tensor([0.062, 0.062, 0.062], dtype=torch.float32)
    weights = torch.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=torch.float32)

    diff = local_centers.unsqueeze(-2) - offsets
    dist_sq = (diff * diff).sum(dim=-1)
    mixture = (weights * torch.exp(-0.5 * dist_sq / (sigmas * sigmas))).sum(dim=-1)
    z = mixture / mixture.max().clamp_min(1e-8)
    rgb = torch.pow(plasma_rgb(0.15 + 0.85 * z), 0.55)
    alpha_v = torch.clamp((z - 0.08) / 0.92, 0.0, 1.0)
    cell_mask = ((uu_c * uu_c + vv_c * vv_c) <= (kernel_extent * kernel_extent)) & (alpha_v > 0.02)

    cells: list[tuple[float, float, float, float, str]] = []
    for i in range(kernel_grid_size - 1):
        for j in range(kernel_grid_size - 1):
            if not bool(cell_mask[i, j].item()):
                continue

            u0 = float(us[j].item())
            u1 = float(us[j + 1].item())
            v0 = float(vs[i].item())
            v1 = float(vs[i + 1].item())
            rgb_u8 = (255.0 * rgb[i, j].clamp(0.0, 1.0)).to(torch.uint8).tolist()
            color_val = f"#{int(rgb_u8[0]):02X}{int(rgb_u8[1]):02X}{int(rgb_u8[2]):02X}"
            cells.append((u0, u1, v0, v1, color_val))
    return cells


def make_sphere_kernel_group(
    world_point: Callable[[np.ndarray], np.ndarray],
    lifted_kernel_point: Callable[[float, float, float], np.ndarray],
    progress: ValueTracker,
) -> VGroup:
    kernel_sphere = VGroup()

    for u0, u1, v0, v1, color_val in iter_kernel_cells():
        corners = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
        sphere_pts = [world_point(lifted_kernel_point(u_local, v_local, 0.0)) for u_local, v_local in corners]
        sphere_cell = Polygon(
            *sphere_pts,
            fill_color=color_val,
            fill_opacity=1.0,
            stroke_width=0.0,
            stroke_opacity=0.0,
        )
        sphere_cell.set_shade_in_3d(False)

        def updater(mob, au0=u0, au1=u1, av0=v0, av1=v1, acol=color_val):
            local_corners = [(au0, av0), (au1, av0), (au1, av1), (au0, av1)]
            pts = [world_point(lifted_kernel_point(uu, vv, progress.get_value())) for uu, vv in local_corners]
            new_poly = Polygon(
                *pts,
                fill_color=acol,
                fill_opacity=1.0,
                stroke_width=0.0,
                stroke_opacity=0.0,
            )
            new_poly.set_shade_in_3d(False)
            mob.become(new_poly)

        sphere_cell.add_updater(updater)
        kernel_sphere.add(sphere_cell)
    return kernel_sphere


def make_chart_kernel_group(
    project_to_chart: Callable[[np.ndarray], np.ndarray],
    lifted_kernel_point: Callable[[float, float, float], np.ndarray],
    progress: ValueTracker,
) -> VGroup:
    kernel_chart = VGroup()

    for u0, u1, v0, v1, color_val in iter_kernel_cells():
        corners = [(u0, v0), (u1, v0), (u1, v1), (u0, v1)]
        chart_pts = [project_to_chart(lifted_kernel_point(u_local, v_local, 0.0)) for u_local, v_local in corners]
        chart_cell = Polygon(
            *chart_pts,
            fill_color=color_val,
            fill_opacity=1.0,
            stroke_width=0.0,
            stroke_opacity=0.0,
        )
        chart_cell.set_shade_in_3d(False)

        def updater(mob, au0=u0, au1=u1, av0=v0, av1=v1, acol=color_val):
            local_corners = [(au0, av0), (au1, av0), (au1, av1), (au0, av1)]
            pts = [project_to_chart(lifted_kernel_point(uu, vv, progress.get_value())) for uu, vv in local_corners]
            new_poly = Polygon(
                *pts,
                fill_color=acol,
                fill_opacity=1.0,
                stroke_width=0.0,
                stroke_opacity=0.0,
            )
            new_poly.set_shade_in_3d(False)
            mob.become(new_poly)

        chart_cell.add_updater(updater)
        kernel_chart.add(chart_cell)
    kernel_chart.set_z_index(5)
    return kernel_chart


class MovingParallelTransportKernelHemisphere(ThreeDScene):
    def construct(self) -> None:
        radius = 2.64
        sphere_shift = np.array([0.0, -2.0, 0.0], dtype=np.float64)

        self.set_camera_orientation(phi=66 * DEGREES, theta=-45 * DEGREES, zoom=1.1)

        chart_radius = 2.4
        chart_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        model = build_transport_model(radius, chart_radius, chart_center)

        def world_point(p: np.ndarray) -> np.ndarray:
            return p + sphere_shift

        hemisphere = Surface(
            lambda u, v: world_point(
                np.array(
                    [
                        radius * np.sin(v) * np.cos(u),
                        radius * np.sin(v) * np.sin(u),
                        radius * np.cos(v),
                    ],
                    dtype=np.float64,
                )
            ),
            u_range=[0.0, TAU],
            v_range=[0.0, np.pi / 2.0],
            resolution=(34, 18),
            fill_color="#1A3E5C",
            fill_opacity=1.0,
            checkerboard_colors=["#1A6C88", "#1A6C88"],
            stroke_opacity=0.0,
        )

        equator = ParametricFunction(
            lambda t: world_point(np.array([radius * np.cos(t), radius * np.sin(t), 0.0], dtype=np.float64)),
            t_range=[0.0, TAU],
            color="#86D549",
            stroke_width=2.4,
            stroke_opacity=0.9,
        )

        geodesic_sphere = VGroup()
        for i in range(3):
            geodesic_sphere.add(
                ParametricFunction(
                    lambda t, seg_idx=i: world_point(model.geodesic_point(seg_idx, t)),
                    t_range=[0.0, 1.0],
                    color="#22A884",
                    stroke_width=5.0,
                    stroke_opacity=0.92,
                )
            )

        vertex_sphere = VGroup(
            *[
                Dot(point=world_point(model.vertices[i]), radius=0.075, color="#FDE725")
                for i in range(3)
            ]
        )

        transport_progress = ValueTracker(0.0)
        kernel_sphere = make_sphere_kernel_group(world_point, model.lifted_kernel_point, transport_progress)

        self.play(FadeIn(hemisphere), FadeIn(equator), run_time=0.6)
        self.play(Create(geodesic_sphere), FadeIn(vertex_sphere), run_time=1.4)
        self.play(FadeIn(kernel_sphere), run_time=0.7)
        self.play(transport_progress.animate.set_value(3.0), run_time=11.0, rate_func=linear)
        self.wait(0.6)


class MovingParallelTransportKernelChart(Scene):
    def construct(self) -> None:
        radius = 2.64
        chart_radius = 2.4
        chart_center = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        model = build_transport_model(radius, chart_radius, chart_center)

        chart_disk = Circle(
            radius=chart_radius,
            stroke_color="#86D549",
            stroke_width=2.5,
            fill_color="#121B2A",
            fill_opacity=0.9,
        ).move_to(chart_center)
        chart_ring = Circle(
            radius=chart_radius,
            stroke_color="#28AE80",
            stroke_width=1.2,
            stroke_opacity=0.75,
            fill_opacity=0.0,
        ).move_to(chart_center)

        geodesic_chart = VGroup()
        for i in range(3):
            chart_curve_points = [model.project_to_chart(model.geodesic_point(i, float(t))) for t in np.linspace(0.0, 1.0, 65)]
            chart_curve = VMobject()
            chart_curve.set_points_as_corners(chart_curve_points)
            chart_curve.set_stroke(color="#22A884", width=4.0, opacity=0.92)
            geodesic_chart.add(chart_curve)
        geodesic_chart.set_z_index(2)

        vertex_chart = VGroup(
            *[
                Dot(point=model.project_to_chart(model.vertices[i]), radius=0.09, color="#FDE725")
                for i in range(3)
            ]
        )
        vertex_chart.set_z_index(3)

        transport_progress = ValueTracker(0.0)
        kernel_chart = make_chart_kernel_group(model.project_to_chart, model.lifted_kernel_point, transport_progress)

        self.play(FadeIn(chart_disk), FadeIn(chart_ring), run_time=0.6)
        self.play(Create(geodesic_chart), FadeIn(vertex_chart), run_time=1.4)
        self.play(FadeIn(kernel_chart), run_time=0.7)
        self.play(transport_progress.animate.set_value(3.0), run_time=11.0, rate_func=linear)
        self.wait(0.6)
