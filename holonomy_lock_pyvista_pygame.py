#!/usr/bin/env python3
"""
Holonomy Lock: interactive PTC alignment puzzle using PyVista + Pygame.

Gameplay:
- Pick a closed loop (keys 1/2/3).
- Press SPACE to parallel-transport the kernel frame around that loop.
- Unlock succeeds if final angle matches the hidden target holonomy within tolerance.

Controls:
- 1 / 2 / 3 : select loop
- SPACE     : run transport on selected loop
- R         : new random target lock angle
- ESC / Q   : quit

Install (example):
- pip install pyvista pygame numpy
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

try:
    import pyvista as pv
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit("Missing dependency: pyvista. Install with `pip install pyvista`.") from exc

try:
    import pygame
except ImportError as exc:  # pragma: no cover - runtime dependency guard
    raise SystemExit("Missing dependency: pygame. Install with `pip install pygame`.") from exc


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


def build_spherical_loop(unit_vertices: list[np.ndarray], radius: float, samples_per_edge: int = 90) -> np.ndarray:
    path: list[np.ndarray] = []
    for i in range(len(unit_vertices)):
        a = unit_vertices[i]
        b = unit_vertices[(i + 1) % len(unit_vertices)]
        t_values = np.linspace(0.0, 1.0, samples_per_edge, endpoint=False)
        for t in t_values:
            path.append(radius * slerp(a, b, float(t)))
    path.append(radius * unit_vertices[0])
    return np.asarray(path, dtype=np.float64)


def transport_frame_along_path(path_points: np.ndarray, radius: float, e1_init: np.ndarray) -> np.ndarray:
    normals = path_points / radius
    n = len(path_points)
    e1 = np.zeros((n, 3), dtype=np.float64)

    e1_seed = e1_init - np.dot(e1_init, normals[0]) * normals[0]
    e1[0] = normalize(e1_seed)

    for i in range(1, n):
        transported = rotate_with_normal_change(e1[i - 1], normals[i - 1], normals[i])
        transported -= np.dot(transported, normals[i]) * normals[i]
        e1[i] = normalize(transported)
    return e1


def signed_holonomy_deg(e1_start: np.ndarray, e1_final: np.ndarray, normal: np.ndarray) -> float:
    return float(
        np.degrees(
            np.arctan2(
                np.dot(normal, np.cross(e1_start, e1_final)),
                float(np.clip(np.dot(e1_start, e1_final), -1.0, 1.0)),
            )
        )
    )


@dataclass
class LoopSpec:
    name: str
    color: str
    vertices: list[np.ndarray]


@dataclass
class LoopData:
    name: str
    color: str
    path: np.ndarray
    e1_path: np.ndarray
    holonomy_deg: float


class HolonomyLockGame:
    def __init__(self) -> None:
        self.radius = 2.2
        self.axis_len = 0.72
        self.tolerance_deg = 1.0
        self.rng = np.random.default_rng(4)

        self.start_unit = normalize(np.array([0.40, -0.78, 0.48], dtype=np.float64))
        self.e1_init = normalize(np.array([0.06, 0.82, 0.57], dtype=np.float64))

        self.kernel_offsets = [
            (0.0, 0.0),
            (0.25, 0.0),
            (-0.25, 0.0),
            (0.0, 0.25),
            (0.0, -0.25),
        ]

        self.loop_specs = self._build_loop_specs()
        self.loops = self._precompute_loops()

        self.target_idx = 0
        self.target_deg = 0.0
        self.selected_idx = 0
        self.last_theta: float | None = None
        self.last_error: float | None = None
        self.unlocked = False

        self.animating = False
        self.anim_step = 0

        headless = os.getenv("PTC_HEADLESS", "0") == "1"
        self.plotter = pv.Plotter(
            window_size=(1280, 960),
            title="Holonomy Lock (PyVista)",
            off_screen=headless,
        )
        self.plotter.set_background("#070B14")

        self.path_actor = None
        self.mover_actor = None
        self.axis_actor = None
        self.final_axis_actor = None
        self.kernel_actor = None

        self.path_poly: pv.PolyData | None = None
        self.mover_poly: pv.PolyData | None = None
        self.axis_poly: pv.PolyData | None = None
        self.final_axis_poly: pv.PolyData | None = None
        self.kernel_poly: pv.PolyData | None = None

        self._setup_scene()
        self._new_target()
        self._set_selected_loop(0)

    def _build_loop_specs(self) -> list[LoopSpec]:
        return [
            LoopSpec(
                name="Loop A (triangle)",
                color="#22D3EE",
                vertices=[
                    self.start_unit,
                    normalize(np.array([0.62, -0.66, 0.42], dtype=np.float64)),
                    normalize(np.array([0.23, -0.56, 0.79], dtype=np.float64)),
                ],
            ),
            LoopSpec(
                name="Loop B (irregular)",
                color="#60A5FA",
                vertices=[
                    self.start_unit,
                    normalize(np.array([0.91, -0.07, 0.40], dtype=np.float64)),
                    normalize(np.array([0.08, 0.73, 0.68], dtype=np.float64)),
                    normalize(np.array([-0.26, -0.48, 0.84], dtype=np.float64)),
                ],
            ),
            LoopSpec(
                name="Loop C (wide loop)",
                color="#34D399",
                vertices=[
                    self.start_unit,
                    normalize(np.array([0.95, 0.23, 0.20], dtype=np.float64)),
                    normalize(np.array([-0.34, 0.84, 0.42], dtype=np.float64)),
                    normalize(np.array([-0.86, -0.19, 0.47], dtype=np.float64)),
                ],
            ),
        ]

    def _precompute_loops(self) -> list[LoopData]:
        out: list[LoopData] = []
        for spec in self.loop_specs:
            path = build_spherical_loop(spec.vertices, radius=self.radius, samples_per_edge=90)
            e1_path = transport_frame_along_path(path, radius=self.radius, e1_init=self.e1_init)

            n0 = normalize(path[0])
            e1_start = normalize(e1_path[0] - np.dot(e1_path[0], n0) * n0)
            e1_final = normalize(e1_path[-1] - np.dot(e1_path[-1], n0) * n0)
            theta = abs(signed_holonomy_deg(e1_start, e1_final, n0))

            out.append(LoopData(spec.name, spec.color, path, e1_path, theta))
        return out

    def _setup_scene(self) -> None:
        sphere = pv.Sphere(radius=self.radius, theta_resolution=56, phi_resolution=56)
        self.plotter.add_mesh(
            sphere,
            color="#1D2A46",
            opacity=0.45,
            show_edges=True,
            edge_color="#3B82F6",
            line_width=0.6,
            smooth_shading=True,
        )

        start_point = self.radius * self.start_unit
        start_e1 = normalize(self.e1_init - np.dot(self.e1_init, self.start_unit) * self.start_unit)

        start_poly = pv.PolyData(np.asarray([start_point], dtype=np.float64))
        self.plotter.add_mesh(
            start_poly,
            color="#E5E7EB",
            point_size=14,
            render_points_as_spheres=True,
        )

        init_axis_poly = pv.Line(start_point, start_point + self.axis_len * start_e1)
        self.plotter.add_mesh(init_axis_poly, color="#9CA3AF", line_width=4.5)

        self.plotter.camera_position = [
            (6.2, -5.3, 4.8),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
        ]

    def _new_target(self) -> None:
        self.target_idx = int(self.rng.integers(0, len(self.loops)))
        self.target_deg = self.loops[self.target_idx].holonomy_deg
        self.last_theta = None
        self.last_error = None
        self.unlocked = False
        self.animating = False
        self.anim_step = 0
        self._remove_attempt_actors()

    def _set_selected_loop(self, idx: int) -> None:
        self.selected_idx = int(np.clip(idx, 0, len(self.loops) - 1))
        self.animating = False
        self.anim_step = 0
        self._remove_attempt_actors()

        loop = self.loops[self.selected_idx]
        self.path_poly = pv.Spline(loop.path, len(loop.path))
        self.path_actor = self.plotter.add_mesh(self.path_poly, color=loop.color, line_width=4.5)

    def _remove_attempt_actors(self) -> None:
        for actor_name in [
            "path_actor",
            "mover_actor",
            "axis_actor",
            "final_axis_actor",
            "kernel_actor",
        ]:
            actor = getattr(self, actor_name)
            if actor is not None:
                self.plotter.remove_actor(actor)
                setattr(self, actor_name, None)

        self.path_poly = None
        self.mover_poly = None
        self.axis_poly = None
        self.final_axis_poly = None
        self.kernel_poly = None

    def _start_attempt(self) -> None:
        if self.animating:
            return

        loop = self.loops[self.selected_idx]
        self.last_theta = None
        self.last_error = None
        self.unlocked = False
        self.anim_step = 0
        self.animating = True

        if self.path_actor is not None:
            self.plotter.remove_actor(self.path_actor)
        self.path_poly = pv.Spline(loop.path, len(loop.path))
        self.path_actor = self.plotter.add_mesh(self.path_poly, color=loop.color, line_width=5.2)

        center = loop.path[0]
        e1 = loop.e1_path[0]
        e2 = normalize(np.cross(normalize(center), e1))

        self.mover_poly = pv.PolyData(np.asarray([center], dtype=np.float64))
        self.mover_actor = self.plotter.add_mesh(
            self.mover_poly,
            color="#FACC15",
            point_size=16,
            render_points_as_spheres=True,
        )

        self.axis_poly = pv.Line(center, center + self.axis_len * e1)
        self.axis_actor = self.plotter.add_mesh(self.axis_poly, color="#F97316", line_width=5.2)

        kernel_pts = np.asarray(
            [exp_map_sphere(center, u * e1 + v * e2, radius=self.radius) for u, v in self.kernel_offsets],
            dtype=np.float64,
        )
        self.kernel_poly = pv.PolyData(kernel_pts)
        self.kernel_actor = self.plotter.add_mesh(
            self.kernel_poly,
            color="#22D3EE",
            point_size=9,
            render_points_as_spheres=True,
        )

        if self.final_axis_actor is not None:
            self.plotter.remove_actor(self.final_axis_actor)
            self.final_axis_actor = None

    def _advance_attempt(self, steps: int = 2) -> None:
        if not self.animating:
            return

        loop = self.loops[self.selected_idx]
        max_idx = len(loop.path) - 1
        self.anim_step = min(max_idx, self.anim_step + steps)

        center = loop.path[self.anim_step]
        e1 = loop.e1_path[self.anim_step]
        normal = normalize(center)
        e2 = normalize(np.cross(normal, e1))

        if self.mover_poly is not None:
            self.mover_poly.points = np.asarray([center], dtype=np.float64)
        if self.axis_poly is not None:
            self.axis_poly.points = np.asarray([center, center + self.axis_len * e1], dtype=np.float64)
        if self.kernel_poly is not None:
            kernel_pts = np.asarray(
                [exp_map_sphere(center, u * e1 + v * e2, radius=self.radius) for u, v in self.kernel_offsets],
                dtype=np.float64,
            )
            self.kernel_poly.points = kernel_pts

        if self.anim_step >= max_idx:
            self.animating = False
            self._finish_attempt()

    def _finish_attempt(self) -> None:
        loop = self.loops[self.selected_idx]
        n0 = normalize(loop.path[0])
        e1_start = normalize(loop.e1_path[0] - np.dot(loop.e1_path[0], n0) * n0)
        e1_final = normalize(loop.e1_path[-1] - np.dot(loop.e1_path[-1], n0) * n0)

        theta = abs(signed_holonomy_deg(e1_start, e1_final, n0))
        error = abs(theta - self.target_deg)

        self.last_theta = theta
        self.last_error = error
        self.unlocked = error < self.tolerance_deg

        start_point = loop.path[0]
        self.final_axis_poly = pv.Line(start_point, start_point + self.axis_len * e1_final)
        self.final_axis_actor = self.plotter.add_mesh(self.final_axis_poly, color="#FB923C", line_width=5.8)

    def _hud_progress(self) -> float:
        if self.last_error is None:
            return 0.0
        return float(np.clip(1.0 - self.last_error / max(self.target_deg, 1.0), 0.0, 1.0))

    def _draw_hud(self, screen: pygame.Surface, font: pygame.font.Font, small_font: pygame.font.Font) -> None:
        bg = (7, 11, 20)
        white = (226, 232, 240)
        cyan = (34, 211, 238)
        red = (252, 165, 165)
        green = (134, 239, 172)
        yellow = (253, 224, 71)

        screen.fill(bg)

        title = font.render("Holonomy Lock", True, white)
        screen.blit(title, (18, 14))

        target_text = small_font.render(
            f"Target angle: {self.target_deg:0.1f} deg   (tol +/- {self.tolerance_deg:0.1f} deg)",
            True,
            yellow,
        )
        screen.blit(target_text, (18, 56))

        selected = self.loops[self.selected_idx]
        selected_text = small_font.render(f"Selected: {selected.name}", True, cyan)
        screen.blit(selected_text, (18, 84))

        if self.last_theta is None:
            result_line = "Last holonomy: --.- deg"
            error_line = "Error: --.- deg"
        else:
            result_line = f"Last holonomy: {self.last_theta:0.1f} deg"
            error_line = f"Error: {self.last_error:0.1f} deg"

        screen.blit(small_font.render(result_line, True, white), (18, 112))
        screen.blit(small_font.render(error_line, True, white), (18, 138))

        status = "UNLOCKED" if self.unlocked else "LOCKED"
        status_color = green if self.unlocked else red
        screen.blit(font.render(status, True, status_color), (410, 14))

        pygame.draw.rect(screen, (30, 41, 59), pygame.Rect(18, 172, 350, 20), border_radius=7)
        fill_w = int(350 * self._hud_progress())
        fill_col = (34, 197, 94) if self.unlocked else (34, 211, 238)
        pygame.draw.rect(screen, fill_col, pygame.Rect(18, 172, fill_w, 20), border_radius=7)

        controls = "Controls: 1/2/3 select loop | SPACE run | R new target | ESC/Q quit"
        screen.blit(small_font.render(controls, True, (148, 163, 184)), (18, 208))

        pygame.display.flip()

    def run(self) -> None:
        pygame.init()
        pygame.display.set_caption("Holonomy Lock HUD")
        hud = pygame.display.set_mode((760, 250))

        font = pygame.font.SysFont("consolas", 30, bold=True)
        small_font = pygame.font.SysFont("consolas", 22)
        clock = pygame.time.Clock()

        self.plotter.show(auto_close=False, interactive_update=True)

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE, pygame.K_q):
                        running = False
                    elif event.key == pygame.K_1:
                        self._set_selected_loop(0)
                    elif event.key == pygame.K_2:
                        self._set_selected_loop(1)
                    elif event.key == pygame.K_3:
                        self._set_selected_loop(2)
                    elif event.key == pygame.K_SPACE:
                        self._start_attempt()
                    elif event.key == pygame.K_r:
                        self._new_target()
                        self._set_selected_loop(self.selected_idx)

            if self.animating:
                self._advance_attempt(steps=2)

            self._draw_hud(hud, font, small_font)
            self.plotter.update()
            clock.tick(60)

        self.plotter.close()
        pygame.quit()


def main() -> int:
    game = HolonomyLockGame()
    game.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
