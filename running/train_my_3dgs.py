#!/usr/bin/env python
"""
My 3D Gaussian Splatting implementation.
Usage:
    python train_my_3dgs.py --dataset chair --iterations 7000

Produces:
    output/<dataset>/point_cloud/iteration_<N>/point_cloud.ply
"""

import os
import sys
import time
import argparse
import numpy as np
import torch
from tqdm import tqdm
from plyfile import PlyData

# Add parent to path
sys.path.insert(0, os.path.dirname(__file__))

from my_3dgs.dataset import read_nerf_synthetic
from my_3dgs.gaussian import GaussianModel
from my_3dgs.renderer import render
from my_3dgs.utils import psnr, save_ply, RGB2SH

# ============================================================
# BasicPointCloud (compat)
# ============================================================
class BasicPointCloud:
    def __init__(self, points, colors, normals):
        self.points = points
        self.colors = colors
        self.normals = normals


# ============================================================
# Pipeline params (same as original 3DGS)
# ============================================================
class PipelineParams:
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = True  # use Python cov3D (more stable for gradients)
        self.debug = False


# ============================================================
# Training args
# ============================================================
class TrainingArgs:
    def __init__(self, iterations=7000):
        # Optimizer
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = iterations
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01

        # Densification
        self.densification_interval = 100
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = min(15000, iterations)
        self.densify_grad_threshold = 0.0002

        # Pruning
        self.prune_interval = 100
        self.min_opacity = 0.005


# ============================================================
# Loss
# ============================================================
def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def ssim(img1, img2, window_size=11, size_average=True):
    """Simplified SSIM using avg pooling."""
    from torch.nn.functional import avg_pool2d as avg_pool2d_fn

    # Create Gaussian window
    def gaussian(window_size, sigma):
        gauss = torch.tensor([
            np.exp(-(x - window_size // 2) ** 2 / (2 * sigma ** 2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    _window = gaussian(window_size, 1.5).unsqueeze(1)
    window = _window.mm(_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = window.expand(img1.size(1), 1, window_size, window_size).contiguous()
    window = window.to(img1.device)

    mu1 = avg_pool2d_fn(img1, window_size, 1, 0)
    mu2 = avg_pool2d_fn(img2, window_size, 1, 0)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = avg_pool2d_fn(img1 * img1, window_size, 1, 0) - mu1_sq
    sigma2_sq = avg_pool2d_fn(img2 * img2, window_size, 1, 0) - mu2_sq
    sigma12 = avg_pool2d_fn(img1 * img2, window_size, 1, 0) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean() if size_average else ssim_map.mean([1, 2, 3])


# ============================================================
# Main training function
# ============================================================
def train(dataset_path, output_path, iterations=7000, sh_degree=3, save_iterations=None):
    """
    Train 3DGS on a NeRF synthetic dataset.

    Args:
        dataset_path: path to nerf_synthetic/<object>/
        output_path: where to save checkpoint PLYs
        iterations: number of training iterations
        sh_degree: max SH degree (0-3)
        save_iterations: list of iteration numbers to save checkpoints
    """
    if save_iterations is None:
        save_iterations = [iterations]  # always save final

    os.makedirs(output_path, exist_ok=True)
    device = torch.device("cuda")

    # ---- Load dataset ----
    print(f"\n{'='*60}")
    print(f"Loading dataset: {dataset_path}")
    print(f"{'='*60}")
    train_cameras, nerf_norm, ply_path, num_pts = read_nerf_synthetic(
        dataset_path, white_background=True, eval=False,
    )
    bg_color = torch.tensor([1.0, 1.0, 1.0], device=device)
    pipe = PipelineParams()

    print(f"  Cameras: {len(train_cameras)}")
    print(f"  Normalization: translate={nerf_norm['translate']}, radius={nerf_norm['radius']:.3f}")

    # ---- Initialize Gaussians ----
    print(f"\nInitializing {num_pts} random Gaussians...")
    # Same as original: random points in [-1.3, 1.3]^3 box
    xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
    # Random colors via SH
    shs = np.random.random((num_pts, 3)) / 255.0
    from my_3dgs.utils import SH2RGB
    rgb = np.clip(SH2RGB(shs.reshape(-1, 1, 3)).reshape(-1, 3), 0, 1)
    pcd = BasicPointCloud(points=xyz, colors=rgb, normals=np.zeros((num_pts, 3)))

    gaussians = GaussianModel(sh_degree=sh_degree)
    gaussians.create_from_pcd(pcd.points, pcd.colors, nerf_norm["radius"])
    gaussians.training_setup(TrainingArgs(iterations))
    gaussians.to(device)

    print(f"  Initial points: {gaussians._xyz.shape[0]}")

    # ---- Training loop ----
    print(f"\n{'='*60}")
    print(f"Training for {iterations} iterations...")
    print(f"{'='*60}")

    viewpoint_stack = train_cameras.copy()
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(1, iterations + 1), desc="Training")
    best_psnr = 0.0

    for iteration in progress_bar:
        # Pick a random camera
        if not viewpoint_stack:
            viewpoint_stack = train_cameras.copy()
        viewpoint_cam = viewpoint_stack.pop(np.random.randint(len(viewpoint_stack)))

        # Render
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg_color)
        image = render_pkg["render"]
        viewspace_point_tensor = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        # Loss: 0.8 * L1 + 0.2 * (1 - SSIM)
        gt_image = viewpoint_cam.original_image
        Ll1 = l1_loss(image, gt_image)
        ssim_loss = 1.0 - ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        loss = 0.8 * Ll1 + 0.2 * ssim_loss

        # Backward
        loss.backward()

        # Logging
        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.5f}",
                    "Pts": f"{gaussians._xyz.shape[0]}",
                })

            # Track PSNR occasionally
            if iteration % 500 == 0:
                current_psnr = psnr(image, gt_image)
                if current_psnr > best_psnr:
                    best_psnr = current_psnr
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.5f}",
                    "PSNR": f"{current_psnr:.2f}",
                    "Pts": f"{gaussians._xyz.shape[0]}",
                })

        # Densification
        if iteration < TrainingArgs().densify_until_iter:
            # Update max_radii2D
            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
            )
            # Accumulate gradients
            gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

            if iteration > TrainingArgs().densify_from_iter and \
               iteration % TrainingArgs().densification_interval == 0:
                size_threshold = 20 if iteration > TrainingArgs().opacity_reset_interval else None
                gaussians.densify_and_prune(
                    TrainingArgs().densify_grad_threshold,
                    TrainingArgs().min_opacity,
                    gaussians.spatial_lr_scale,
                    size_threshold,
                )

            if iteration % TrainingArgs().opacity_reset_interval == 0:
                gaussians.reset_opacity()

        # Update learning rate
        gaussians.update_learning_rate(iteration)

        # Increase SH degree gradually
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Optimizer step
        gaussians.optimizer.step()
        gaussians.optimizer.zero_grad(set_to_none=True)

        # Save checkpoint
        if iteration in save_iterations:
            save_dir = os.path.join(output_path, "point_cloud", f"iteration_{iteration}")
            os.makedirs(save_dir, exist_ok=True)
            ply_path = os.path.join(save_dir, "point_cloud.ply")
            save_ply(ply_path, gaussians.get_xyz, gaussians._features_dc,
                     gaussians.get_opacity, gaussians.get_scaling, gaussians.get_rotation)
            print(f"\n  Checkpoint saved: {ply_path}")

    # ---- Save final PLY ----
    save_dir = os.path.join(output_path, "point_cloud", f"iteration_{iterations}")
    os.makedirs(save_dir, exist_ok=True)
    ply_path = os.path.join(save_dir, "point_cloud.ply")
    save_ply(ply_path, gaussians.get_xyz, gaussians._features_dc,
             gaussians.get_opacity, gaussians.get_scaling, gaussians.get_rotation)

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Best PSNR: {best_psnr:.2f} dB")
    print(f"  Final points: {gaussians._xyz.shape[0]}")
    print(f"  PLY saved to: {ply_path}")
    print(f"{'='*60}")

    return ply_path, best_psnr


def oneupSHdegree(self):
    """Increase active SH degree up to max."""
    if self.active_sh_degree < self.max_sh_degree:
        self.active_sh_degree += 1

# Monkey-patch
GaussianModel.oneupSHdegree = oneupSHdegree


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="My 3D Gaussian Splatting")
    parser.add_argument("--dataset", type=str, default="chair",
                        help="Dataset name under nerf_synthetic/")
    parser.add_argument("--iterations", type=int, default=7000,
                        help="Training iterations (default 7000, use 30000 for full)")
    parser.add_argument("--save_iters", type=int, nargs="*", default=None,
                        help="Additional iterations to save checkpoints")
    args = parser.parse_args()

    base_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(base_dir, "..", "nerf_synthetic", args.dataset)
    output_path = os.path.join(base_dir, "output", args.dataset + "_my3dgs")

    if not os.path.isdir(dataset_path):
        print(f"ERROR: Dataset not found: {dataset_path}")
        sys.exit(1)

    save_iters = args.save_iters if args.save_iters else []
    if args.iterations not in save_iters:
        save_iters.append(args.iterations)

    t0 = time.time()
    ply_path, best_psnr = train(
        dataset_path, output_path,
        iterations=args.iterations,
        save_iterations=save_iters,
    )
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
