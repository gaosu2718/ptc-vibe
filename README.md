# PTC Vibe

Parallel Transport Convolution (PTC) visual demos, mini-games, and mesh utilities.

This repository contains:
- Manim animations for geometric PTC concepts on curved surfaces.
- Interactive Pygame (and PyVista + Pygame) games that make holonomy/PTC behavior playable.
- A utility to clean STL meshes into manifold triangle meshes.

## Project Layout

### Manim demos
- `ptc_demo_spherical_holonomy_manim.py`
  - Scene: `PTCSphericalHolonomyDemo`
  - Shows holonomy from parallel transport around a closed loop on a sphere.
- `ptc_demo_convolution_chase_game_manim.py`
  - Scene: `PTCConvolutionChaseGame`
  - Greedy kernel walk guided by local convolution score on a sphere.
- `ptc_demo_rigid_motion_invariance_manim.py`
  - Scene: `PTCRigidMotionInvarianceDemo`
  - Side-by-side rigid-motion invariance/equivariance visualization.
- `moving_ptc_hemisphere_manim.py`
  - Scenes:
    - `MovingParallelTransportKernelHemisphere`
    - `MovingParallelTransportKernelChart`
  - Parallel transport kernel motion on a hemisphere and its chart view.
- `moving_gmm_kernel_manim.py`
  - Scene: `MovingGaussianKernel2D`
  - 2D moving Gaussian kernel baseline visualization.

### Interactive games
- `holonomy_lock.py`
  - Pygame-only holonomy lock puzzle on sphere coordinates.
- `filter_hunter.py`
  - Single-player PTC target hunting game.
- `filter_hunter_2p.py`
  - Two-player competitive PTC racing/hunting game.
- `holonomy_lock_pyvista_pygame.py`
  - PyVista 3D view + Pygame HUD/controls for interactive holonomy lock.

### Utilities
- `stl_to_manifold_mesh.py`
  - Loads STL and repairs/exports manifold-friendly triangle mesh.

### Data/output
- `data/` for source meshes (e.g. STL files).
- `output/` for generated mesh and media artifacts.

## Environment Setup

Use the provided Conda environment file:

```bash
conda env create -f req.yaml
conda activate ptc
```

Core dependencies are pinned in `req.yaml`:
- Python 3.9
- NumPy, SciPy
- Manim
- PyTorch
- PyVista
- Pygame
- Trimesh

## Run Commands

### Manim scenes

```bash
conda run -n ptc manim -pql ptc_demo_spherical_holonomy_manim.py PTCSphericalHolonomyDemo
conda run -n ptc manim -pql ptc_demo_convolution_chase_game_manim.py PTCConvolutionChaseGame
conda run -n ptc manim -pql ptc_demo_rigid_motion_invariance_manim.py PTCRigidMotionInvarianceDemo

conda run -n ptc manim -pql moving_ptc_hemisphere_manim.py MovingParallelTransportKernelHemisphere
conda run -n ptc manim -pql moving_ptc_hemisphere_manim.py MovingParallelTransportKernelChart

conda run -n ptc manim -pql moving_gmm_kernel_manim.py MovingGaussianKernel2D
```

Notes:
- Scene scripts set their own render config (resolution/frame rate) in code.
- Replace `-pql` with higher quality flags if needed.

### Pygame games

```bash
conda run -n ptc python holonomy_lock.py
conda run -n ptc python filter_hunter.py
conda run -n ptc python filter_hunter_2p.py
```

### PyVista + Pygame holonomy game

```bash
conda run -n ptc python holonomy_lock_pyvista_pygame.py
```

If running in a headless environment:

```bash
PTC_HEADLESS=1 conda run -n ptc python holonomy_lock_pyvista_pygame.py
```

## Game Controls (Current)

### `holonomy_lock.py`
- `Left/Right`: steer
- `Up`: move
- `Down`: brake
- `Space`: evaluate lock
- `1/2/3`: window size (1280x720 / 2560x1440 / 3840x2160)
- `R`: reset level
- `Esc` or `Q`: quit

### `filter_hunter.py`
- `Left/Right`: steer
- `Up`: move
- `Down`: rotate kernel scan clockwise
- `1/2/3`: window size (1280x720 / 2560x1440 / 3840x2160)
- `R`: respawn targets/reset score
- `Esc` or `Q`: quit

### `filter_hunter_2p.py`
- Player 1: `Left/Right` steer, `Up` move, `Down` scan clockwise
- Player 2: `A/D` steer, `W` move, `S` scan clockwise
- Global:
  - `1/2/3`: window size (1280x720 / 2560x1440 / 3840x2160)
  - `R`: restart
  - `Esc` or `Q`: quit

## STL to Manifold Mesh Utility

Basic usage:

```bash
conda run -n ptc python stl_to_manifold_mesh.py
```

Optional flags:

```bash
conda run -n ptc python stl_to_manifold_mesh.py \
  --input data/YourMesh.stl \
  --output output/YourMesh_manifold.ply \
  --merge-digits 6 \
  --strict
```

By default:
- Input is the first `.stl` found in `data/`.
- Output goes to `output/<name>_manifold.ply`.

## Notes

- This repo is focused on concise, visual PTC intuition and interactive geometry.
- For reproducible results, run everything from the `ptc` Conda environment.
