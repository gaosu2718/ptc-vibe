#!/usr/bin/env python3
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Avoid OpenMP shared-memory issues in restricted environments.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import trimesh
from manim import (
    Create,
    DEGREES,
    Dot3D,
    FadeIn,
    LEFT,
    OUT,
    Polyhedron,
    RIGHT,
    ThreeDScene,
    UP,
    ValueTracker,
    VGroup,
    VMobject,
    config,
    linear,
)
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

from stl_to_manifold_mesh import make_manifold, mesh_from_any, pick_input_path

config.pixel_width = 1920
config.pixel_height = 1080
config.frame_rate = 30
config.background_color = "#0A0F14"


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
    dot_val = float(np.clip(np.dot(n_prev, n_cur), -1.0, 1.0))
    if c_norm < 1e-10:
        return vec.copy()
    angle = float(np.arctan2(c_norm, dot_val))
    return rodrigues_rotation(cross_val, angle) @ vec


def plasma_rgb_scalar(z: float) -> tuple[int, int, int]:
    z = float(np.clip(z, 0.0, 1.0))
    stops = np.array([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float64)
    colors = np.array(
        [
            [13, 8, 135],
            [126, 3, 167],
            [204, 71, 120],
            [248, 149, 64],
            [240, 249, 33],
        ],
        dtype=np.float64,
    )
    idx = int(np.clip(np.searchsorted(stops, z, side="right") - 1, 0, len(stops) - 2))
    t = (z - stops[idx]) / max(stops[idx + 1] - stops[idx], 1e-8)
    rgb = (1.0 - t) * colors[idx] + t * colors[idx + 1]
    rgb = np.clip(np.power(rgb / 255.0, 0.6) * 255.0, 0.0, 255.0).astype(np.uint8)
    return int(rgb[0]), int(rgb[1]), int(rgb[2])


def hex_color(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def voxel_cluster_simplify(vertices: np.ndarray, faces: np.ndarray, voxel_size: float) -> tuple[np.ndarray, np.ndarray]:
    vmin = vertices.min(axis=0)
    grid = np.floor((vertices - vmin) / max(voxel_size, 1e-12)).astype(np.int64)
    _, inverse = np.unique(grid, axis=0, return_inverse=True)

    count = np.bincount(inverse).astype(np.float64)
    new_vertices = np.column_stack(
        [
            np.bincount(inverse, weights=vertices[:, 0]) / count,
            np.bincount(inverse, weights=vertices[:, 1]) / count,
            np.bincount(inverse, weights=vertices[:, 2]) / count,
        ]
    )

    new_faces = inverse[faces]
    distinct = (new_faces[:, 0] != new_faces[:, 1]) & (new_faces[:, 1] != new_faces[:, 2]) & (new_faces[:, 0] != new_faces[:, 2])
    new_faces = new_faces[distinct]
    if len(new_faces) == 0:
        return new_vertices, new_faces

    sorted_faces = np.sort(new_faces, axis=1)
    _, unique_idx = np.unique(sorted_faces, axis=0, return_index=True)
    new_faces = new_faces[np.sort(unique_idx)]

    used = np.unique(new_faces.ravel())
    remap = -np.ones(len(new_vertices), dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    new_faces = remap[new_faces]
    new_vertices = new_vertices[used]
    return new_vertices, new_faces


def simplify_mesh_to_target(mesh: trimesh.Trimesh, target_faces: int = 2200) -> trimesh.Trimesh:
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    extent = vertices.max(axis=0) - vertices.min(axis=0)
    diag = float(np.linalg.norm(extent))

    lo = diag / 800.0
    hi = diag / 12.0
    best_vertices = vertices
    best_faces = faces
    best_count = len(faces)

    for _ in range(20):
        mid = 0.5 * (lo + hi)
        v_mid, f_mid = voxel_cluster_simplify(vertices, faces, mid)
        f_count = len(f_mid)
        if f_count == 0:
            hi = mid
            continue

        if abs(f_count - target_faces) < abs(best_count - target_faces):
            best_vertices, best_faces, best_count = v_mid, f_mid, f_count

        if f_count > target_faces:
            lo = mid
        else:
            hi = mid

    out = trimesh.Trimesh(vertices=best_vertices, faces=best_faces, process=False)
    out.remove_unreferenced_vertices()
    out.merge_vertices(digits_vertex=6)
    out.remove_unreferenced_vertices()
    trimesh.repair.fix_normals(out)
    return out


def unique_edges_from_faces(faces: np.ndarray) -> np.ndarray:
    edges = np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    return edges


def shortest_vertex_path(vertices: np.ndarray, faces: np.ndarray, start_idx: int, end_idx: int) -> np.ndarray:
    edges = unique_edges_from_faces(faces)
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    n_vertices = len(vertices)

    rows = np.hstack([edges[:, 0], edges[:, 1]])
    cols = np.hstack([edges[:, 1], edges[:, 0]])
    data = np.hstack([lengths, lengths])
    graph = csr_matrix((data, (rows, cols)), shape=(n_vertices, n_vertices))

    _, predecessors = dijkstra(graph, directed=False, indices=start_idx, return_predecessors=True)
    sentinel = -9999
    if predecessors[end_idx] == sentinel:
        raise RuntimeError("No path found between start and end vertices.")

    path = [end_idx]
    while path[-1] != start_idx:
        path.append(int(predecessors[path[-1]]))
    path.reverse()
    return np.asarray(path, dtype=np.int64)


def resample_path(points: np.ndarray, normals: np.ndarray, sample_count: int) -> tuple[np.ndarray, np.ndarray]:
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    total = float(cumulative[-1])
    if total < 1e-10:
        out_points = np.repeat(points[:1], sample_count, axis=0)
        out_normals = np.repeat(normals[:1], sample_count, axis=0)
        return out_points, out_normals

    target = np.linspace(0.0, total, sample_count)
    out_points = np.zeros((sample_count, 3), dtype=np.float64)
    out_normals = np.zeros((sample_count, 3), dtype=np.float64)

    seg = 0
    for i, s in enumerate(target):
        while seg < len(cumulative) - 2 and cumulative[seg + 1] < s:
            seg += 1
        denom = max(cumulative[seg + 1] - cumulative[seg], 1e-12)
        t = (s - cumulative[seg]) / denom
        out_points[i] = (1.0 - t) * points[seg] + t * points[seg + 1]
        n = (1.0 - t) * normals[seg] + t * normals[seg + 1]
        out_normals[i] = normalize(n)
    return out_points, out_normals


def parallel_transport_frames(path_points: np.ndarray, path_normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(path_points)
    tangents = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        if i == 0:
            diff = path_points[1] - path_points[0]
        elif i == n - 1:
            diff = path_points[-1] - path_points[-2]
        else:
            diff = path_points[i + 1] - path_points[i - 1]
        tangents[i] = normalize(diff)

    e1 = np.zeros((n, 3), dtype=np.float64)
    e2 = np.zeros((n, 3), dtype=np.float64)

    initial = tangents[0] - np.dot(tangents[0], path_normals[0]) * path_normals[0]
    if np.linalg.norm(initial) < 1e-10:
        fallback = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(np.dot(fallback, path_normals[0])) > 0.9:
            fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        initial = fallback - np.dot(fallback, path_normals[0]) * path_normals[0]
    e1[0] = normalize(initial)
    e2[0] = normalize(np.cross(path_normals[0], e1[0]))

    for i in range(1, n):
        transported = rotate_with_normal_change(e1[i - 1], path_normals[i - 1], path_normals[i])
        transported -= np.dot(transported, path_normals[i]) * path_normals[i]
        if np.linalg.norm(transported) < 1e-10:
            transported = tangents[i] - np.dot(tangents[i], path_normals[i]) * path_normals[i]
        e1[i] = normalize(transported)
        e2[i] = normalize(np.cross(path_normals[i], e1[i]))
    return e1, e2


def frame_at(progress: float, points: np.ndarray, normals: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    clamped = min(max(progress, 0.0), 1.0)
    u = clamped * (len(points) - 1)
    i0 = min(int(np.floor(u)), len(points) - 1)
    i1 = min(i0 + 1, len(points) - 1)
    t = u - i0

    center = (1.0 - t) * points[i0] + t * points[i1]
    normal = normalize((1.0 - t) * normals[i0] + t * normals[i1])
    b1 = normalize((1.0 - t) * e1[i0] + t * e1[i1])
    b1 -= np.dot(b1, normal) * normal
    b1 = normalize(b1)
    b2 = normalize(np.cross(normal, b1))
    return center, b1, b2


def rotation_matrix_from_angles(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)

    rz = np.array(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(pitch), -np.sin(pitch)],
            [0.0, np.sin(pitch), np.cos(pitch)],
        ],
        dtype=np.float64,
    )
    return rz @ rx


def apply_view(points: np.ndarray, rotation: np.ndarray, shift: np.ndarray) -> np.ndarray:
    return points @ rotation.T + shift[None, :]


def build_two_gmm_kernel(local_scale: float) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    extent = 5.0 * local_scale
    grid = 19
    u = np.linspace(-extent, extent, grid)
    v = np.linspace(-extent, extent, grid)
    vv, uu = np.meshgrid(v, u, indexing="ij")

    centers = np.array(
        [
            [-2.0 * local_scale, 0.0],
            [2.0 * local_scale, 0.0],
        ],
        dtype=np.float64,
    )
    sigma = 1.4 * local_scale
    weights = np.array([0.5, 0.5], dtype=np.float64)

    local_uv = np.stack([uu, vv], axis=-1)
    vals = np.zeros_like(uu, dtype=np.float64)
    for w, c in zip(weights, centers):
        d = local_uv - c[None, None, :]
        vals += w * np.exp(-0.5 * np.sum(d * d, axis=-1) / (sigma * sigma))

    vals /= max(float(vals.max()), 1e-8)
    alpha = np.clip((vals - 0.06) / 0.94, 0.0, 1.0)
    mask = (uu * uu + vv * vv) <= (extent * extent)
    mask &= alpha > 0.04

    uv_points = np.stack([uu[mask], vv[mask]], axis=-1)
    strengths = vals[mask]
    alpha_vals = alpha[mask]
    colors = [hex_color(plasma_rgb_scalar(0.15 + 0.85 * float(s))) for s in strengths]
    return uv_points, strengths, colors, alpha_vals


@dataclass
class TransportData:
    vertices: np.ndarray
    faces: np.ndarray
    path_points: np.ndarray
    path_normals: np.ndarray
    frame_e1: np.ndarray
    frame_e2: np.ndarray
    local_uv: np.ndarray
    kernel_colors: list[str]
    kernel_alpha: np.ndarray
    mesh_scale: float


def build_transport_data(input_path: Path | None = None) -> TransportData:
    stl_path = pick_input_path(input_path)
    manifold_cache = Path("output") / f"{stl_path.stem}_manifold.ply"
    if manifold_cache.exists():
        manifold = mesh_from_any(manifold_cache)
    else:
        raw_mesh = mesh_from_any(stl_path)
        manifold = make_manifold(raw_mesh, merge_digits=6)
        manifold_cache.parent.mkdir(parents=True, exist_ok=True)
        manifold.export(manifold_cache)
    simple = simplify_mesh_to_target(manifold, target_faces=2200)

    vertices = np.asarray(simple.vertices, dtype=np.float64)
    faces = np.asarray(simple.faces, dtype=np.int64)

    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    center = 0.5 * (bounds_min + bounds_max)
    extent = bounds_max - bounds_min
    scale = 2.6 / max(float(np.max(extent)), 1e-8)
    vertices = (vertices - center) * scale

    working = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    normals = np.asarray(working.vertex_normals, dtype=np.float64)
    normals = np.array([normalize(n) for n in normals], dtype=np.float64)

    start_idx = int(np.argmin(vertices[:, 2]))
    end_idx = int(np.argmax(vertices[:, 2]))
    path_idx = shortest_vertex_path(vertices, faces, start_idx, end_idx)

    path_vertices = vertices[path_idx]
    path_normals_vertex = normals[path_idx]
    path_points, path_normals = resample_path(path_vertices, path_normals_vertex, sample_count=260)
    frame_e1, frame_e2 = parallel_transport_frames(path_points, path_normals)

    edges = unique_edges_from_faces(faces)
    edge_lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    local_scale = float(np.median(edge_lengths)) * 2.8
    local_uv, _, kernel_colors, kernel_alpha = build_two_gmm_kernel(local_scale)

    return TransportData(
        vertices=vertices,
        faces=faces,
        path_points=path_points,
        path_normals=path_normals,
        frame_e1=frame_e1,
        frame_e2=frame_e2,
        local_uv=local_uv,
        kernel_colors=kernel_colors,
        kernel_alpha=kernel_alpha,
        mesh_scale=scale,
    )


class PTCOnSTLThreeViews(ThreeDScene):
    def construct(self) -> None:
        data = build_transport_data()
        kd_tree = cKDTree(data.vertices)

        self.set_camera_orientation(phi=62 * DEGREES, theta=-90 * DEGREES, zoom=1.08)
        self.renderer.camera.light_source.move_to(6.0 * OUT + 4.0 * UP + 3.0 * LEFT)

        view_rotations = [
            rotation_matrix_from_angles(yaw_deg=-32.0, pitch_deg=24.0),
            rotation_matrix_from_angles(yaw_deg=22.0, pitch_deg=56.0),
            rotation_matrix_from_angles(yaw_deg=88.0, pitch_deg=28.0),
        ]
        view_shifts = [
            np.array([-4.6, -0.2, 0.0], dtype=np.float64),
            np.array([0.0, -0.2, 0.0], dtype=np.float64),
            np.array([4.6, -0.2, 0.0], dtype=np.float64),
        ]

        progress = ValueTracker(0.0)

        all_meshes = VGroup()
        all_paths = VGroup()
        all_endpoints = VGroup()
        all_kernels = VGroup()

        def kernel_surface_points(path_progress: float) -> np.ndarray:
            center, e1, e2 = frame_at(path_progress, data.path_points, data.path_normals, data.frame_e1, data.frame_e2)
            candidate = center[None, :] + data.local_uv[:, :1] * e1[None, :] + data.local_uv[:, 1:] * e2[None, :]
            _, idx = kd_tree.query(candidate, k=1)
            return data.vertices[idx]

        initial_kernel_points = kernel_surface_points(0.0)

        for panel_idx in range(3):
            rot = view_rotations[panel_idx]
            shift = view_shifts[panel_idx]

            panel_vertices = apply_view(data.vertices, rot, shift)
            mesh = Polyhedron(
                [v for v in panel_vertices],
                data.faces.tolist(),
                faces_config={
                    "fill_color": "#235C78",
                    "fill_opacity": 1.0,
                    "stroke_width": 0.0,
                    "stroke_opacity": 0.0,
                },
                graph_config={
                    "vertex_config": {"fill_opacity": 0.0, "stroke_opacity": 0.0, "stroke_width": 0.0},
                    "edge_config": {"stroke_opacity": 0.0, "stroke_width": 0.0},
                },
            )
            mesh.set_shade_in_3d(True)
            all_meshes.add(mesh)

            path_panel = apply_view(data.path_points, rot, shift)
            path_obj = VMobject(stroke_color="#2FB47C", stroke_width=5.0, stroke_opacity=0.92)
            path_obj.set_points_as_corners(path_panel)
            all_paths.add(path_obj)

            start_dot = Dot3D(point=path_panel[0], radius=0.06, color="#FDE725", resolution=(8, 8))
            end_dot = Dot3D(point=path_panel[-1], radius=0.06, color="#FDE725", resolution=(8, 8))
            all_endpoints.add(start_dot, end_dot)

            kernel_points_panel = apply_view(initial_kernel_points, rot, shift)
            dots = VGroup()
            for k in range(len(kernel_points_panel)):
                dot = Dot3D(
                    point=kernel_points_panel[k],
                    radius=0.026,
                    color=data.kernel_colors[k],
                    resolution=(6, 6),
                )
                dot.set_opacity(float(0.12 + 0.88 * data.kernel_alpha[k]))
                dots.add(dot)

            def update_kernel(group, panel_rotation=rot, panel_shift=shift):
                base_points = kernel_surface_points(progress.get_value())
                moved = apply_view(base_points, panel_rotation, panel_shift)
                for d, p in zip(group, moved):
                    d.move_to(p)

            dots.add_updater(update_kernel)
            all_kernels.add(dots)

        self.add(all_meshes)
        self.wait(0.4)
        self.play(Create(all_paths), FadeIn(all_endpoints), run_time=1.6)
        self.play(FadeIn(all_kernels), run_time=0.7)
        self.play(progress.animate.set_value(1.0), run_time=11.0, rate_func=linear)
        self.wait(0.6)
