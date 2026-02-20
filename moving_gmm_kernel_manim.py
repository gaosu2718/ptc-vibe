from __future__ import annotations

import numpy as np
import torch
from manim import (
    Create,
    FadeIn,
    ImageMobject,
    ParametricFunction,
    Rectangle,
    Scene,
    ValueTracker,
    config,
    linear,
)

config.pixel_width = 1920
config.pixel_height = 1080
config.frame_rate = 30
config.background_color = "#0A0F14"


class MovingGaussianKernel2D(Scene):
    def construct(self) -> None:
        domain_size = 6.5
        domain = Rectangle(
            width=domain_size,
            height=domain_size,
            stroke_color="#86D549",
            stroke_width=2.8,
            fill_color="#121B2A",
            fill_opacity=0.82,
        )

        width_px, height_px = 180, 180
        kernel_size_scene = 1.9
        xs = torch.linspace(-kernel_size_scene / 2, kernel_size_scene / 2, width_px)
        ys = torch.linspace(-kernel_size_scene / 2, kernel_size_scene / 2, height_px)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack([xx, yy], dim=-1)

        offsets = torch.tensor(
            [
                [0.0, 0.24],
                [-0.2078, -0.12],
                [0.2078, -0.12],
            ],
            dtype=torch.float32,
        )
        sigmas = torch.tensor([0.17, 0.17, 0.17], dtype=torch.float32)
        weights = torch.tensor([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=torch.float32)

        triangle_side = domain_size * 0.72
        triangle_radius = triangle_side / np.sqrt(3.0)
        trajectory_shift_y = 0.35
        triangle_center = domain.get_center() + np.array([0.0, trajectory_shift_y, 0.0], dtype=np.float32)
        v0 = triangle_center + np.array([0.0, -triangle_radius, 0.0], dtype=np.float32)
        v1 = triangle_center + np.array(
            [-triangle_side / 2.0, triangle_radius / 2.0, 0.0], dtype=np.float32
        )
        v2 = triangle_center + np.array(
            [triangle_side / 2.0, triangle_radius / 2.0, 0.0], dtype=np.float32
        )

        def viridis_rgb(z: torch.Tensor) -> torch.Tensor:
            stops = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float32, device=z.device)
            colors = torch.tensor(
                [
                    [68, 1, 84],
                    [59, 82, 139],
                    [33, 145, 140],
                    [94, 201, 98],
                    [253, 231, 37],
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

        def kernel_center(t: float) -> np.ndarray:
            u = (t % 1.0) * 3.0
            if u < 1.0:
                return v0 + u * (v1 - v0)
            if u < 2.0:
                return v1 + (u - 1.0) * (v2 - v1)
            return v2 + (u - 2.0) * (v0 - v2)

        def mixture_rgba() -> np.ndarray:
            centers = offsets
            diff = grid.unsqueeze(-2) - centers
            dist_sq = (diff * diff).sum(dim=-1)
            mixture = (weights * torch.exp(-0.5 * dist_sq / (sigmas * sigmas))).sum(dim=-1)
            z = mixture / mixture.max().clamp_min(1e-8)
            rgb = viridis_rgb(z)
            alpha = torch.clamp((z - 0.06) / 0.94, 0.0, 1.0).pow(0.62)

            rgba = torch.cat([rgb, alpha.unsqueeze(-1)], dim=-1)
            rgba_u8 = (255.0 * rgba).to(torch.uint8)
            return torch.flip(rgba_u8, dims=[0]).cpu().numpy()

        progress = ValueTracker(0.0)
        trajectory = ParametricFunction(
            lambda t: kernel_center(t),
            t_range=[0.0, 1.0],
            color="#28AE80",
            stroke_width=2.8,
            stroke_opacity=0.55,
            use_smoothing=False,
        )

        kernel = ImageMobject(mixture_rgba())
        kernel.set(width=kernel_size_scene, height=kernel_size_scene)
        kernel.move_to(kernel_center(0.0))
        kernel.add_updater(lambda m: m.move_to(kernel_center(progress.get_value())))

        self.add(domain)
        self.play(Create(trajectory), run_time=1.3)
        self.play(FadeIn(kernel), run_time=0.6)
        self.play(progress.animate.set_value(1.0), run_time=10.0, rate_func=linear)
        self.wait(0.6)
