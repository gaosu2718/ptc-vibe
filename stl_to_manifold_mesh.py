#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

# Keep BLAS/OpenMP single-threaded to avoid shared-memory issues in restricted environments.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load an STL from ./data and convert it to a manifold triangle mesh."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Path to input STL. Defaults to the first .stl found in ./data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output mesh path (.stl/.ply/.obj). Defaults to ./output/<name>_manifold.ply.",
    )
    parser.add_argument(
        "--merge-digits",
        type=int,
        default=6,
        help="Vertex merge precision used during welding (default: 6).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if the resulting mesh is not closed manifold.",
    )
    return parser.parse_args()


def pick_input_path(input_path: Path | None) -> Path:
    if input_path is not None:
        return input_path

    candidates = sorted(Path("data").glob("*.stl"))
    if not candidates:
        raise FileNotFoundError("No STL file found in ./data")
    return candidates[0]


def mesh_from_any(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=False)

    if isinstance(loaded, trimesh.Scene):
        meshes = [geom for geom in loaded.geometry.values() if isinstance(geom, trimesh.Trimesh)]
        if not meshes:
            raise ValueError(f"No mesh geometry found in scene: {path}")
        mesh = trimesh.util.concatenate(tuple(meshes))
    elif isinstance(loaded, trimesh.Trimesh):
        mesh = loaded
    else:
        raise TypeError(f"Unsupported loaded object type: {type(loaded)}")

    return mesh


def edge_face_stats(mesh: trimesh.Trimesh) -> tuple[int, int, int]:
    edge_count = len(mesh.edges_unique)
    if edge_count == 0:
        return 0, 0, 0

    counts = np.bincount(mesh.edges_unique_inverse, minlength=edge_count)
    gt2 = int((counts > 2).sum())
    eq1 = int((counts == 1).sum())
    eq2 = int((counts == 2).sum())
    return gt2, eq1, eq2


def make_manifold(mesh: trimesh.Trimesh, merge_digits: int) -> trimesh.Trimesh:
    mesh = mesh.copy()

    mesh.remove_infinite_values()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()

    mesh.merge_vertices(digits_vertex=merge_digits)
    mesh.remove_unreferenced_vertices()

    trimesh.repair.fix_normals(mesh)
    trimesh.repair.fix_winding(mesh)

    if not mesh.is_watertight:
        trimesh.repair.fill_holes(mesh)
        mesh.remove_unreferenced_vertices()

    return mesh


def main() -> int:
    args = parse_args()
    input_path = pick_input_path(args.input)
    if input_path.suffix.lower() != ".stl":
        raise ValueError(f"Input must be an STL file, got: {input_path}")

    if args.output is None:
        output_path = Path("output") / f"{input_path.stem}_manifold.ply"
    else:
        output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_mesh = mesh_from_any(input_path)
    manifold_mesh = make_manifold(raw_mesh, merge_digits=args.merge_digits)

    gt2, eq1, eq2 = edge_face_stats(manifold_mesh)
    is_closed_manifold = (
        manifold_mesh.is_watertight
        and manifold_mesh.is_winding_consistent
        and gt2 == 0
        and eq1 == 0
    )

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Raw mesh:      {len(raw_mesh.vertices)} vertices, {len(raw_mesh.faces)} faces")
    print(f"Processed mesh:{len(manifold_mesh.vertices)} vertices, {len(manifold_mesh.faces)} faces")
    print(
        "Edge-face incidence: "
        f">2: {gt2}, ==1: {eq1}, ==2: {eq2}"
    )
    print(
        "Manifold checks: "
        f"watertight={manifold_mesh.is_watertight}, "
        f"winding_consistent={manifold_mesh.is_winding_consistent}, "
        f"closed_manifold={is_closed_manifold}"
    )

    if args.strict and not is_closed_manifold:
        print("ERROR: Strict mode enabled and mesh is not closed manifold after repair.")
        return 2

    manifold_mesh.export(output_path)
    print("Saved manifold triangle mesh.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
