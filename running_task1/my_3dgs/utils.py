"""Spherical harmonics and image utilities for 3DGS."""

import torch
import numpy as np

C0 = 0.28209479177387814


def RGB2SH(rgb):
    """Convert RGB [0,1] to zeroth-order SH coefficients."""
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    """Convert zeroth-order SH coefficients back to RGB [0,1]."""
    return sh * C0 + 0.5


def psnr(img1, img2):
    """Compute PSNR between two images [0,1]."""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()


def save_ply(path, means3D, sh_dc, opacity, scales, rotations):
    """Save Gaussians to PLY file compatible with PhysGaussian."""
    means3D = means3D.detach().cpu().numpy()
    sh_dc = sh_dc.detach().cpu().numpy()
    opacity = opacity.detach().cpu().numpy()
    scales = scales.detach().cpu().numpy()
    rotations = rotations.detach().cpu().numpy()

    # Convert opacity (sigmoid-activated) back to logit for PLY
    # PLY stores raw values; PhysGaussian applies sigmoid on load
    opacity_raw = np.log(np.clip(opacity, 1e-8, 1 - 1e-8) / (1 - np.clip(opacity, 1e-8, 1 - 1e-8)))

    # Convert SH DC to RGB colors (like original 3DGS storePly)
    # We store f_dc_0, f_dc_1, f_dc_2 directly (SH DC components)
    # But also compute colors for visualization
    f_dc = sh_dc.reshape(-1, 1, 3)
    colors = np.clip(SH2RGB(f_dc).reshape(-1, 3), 0, 1)

    from plyfile import PlyData, PlyElement
    N = means3D.shape[0]
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
    ]
    # Also store f_rest (higher order SH) if available, but we'll use zeros for simplicity
    # For sh_degree=3, we need (3+1)^2 - 1 = 15 - 1 = 14 higher-order coefficients (zero for degree 0)
    # Actually PhysGaussian expects: f_dc (3) + f_rest (45 for degree 3) = 48 total
    # f_rest: (degree 3 + 1)^2 * 3 - 3 = 48 - 3 = 45
    # But let's check what format they expect...

    # Simpler: use extra_f_names for remaining SH
    for i in range(45):
        dtype.append((f'f_rest_{i}', 'f4'))

    elements = np.empty(N, dtype=dtype)
    elements['x'] = means3D[:, 0]
    elements['y'] = means3D[:, 1]
    elements['z'] = means3D[:, 2]
    elements['nx'] = np.zeros(N)
    elements['ny'] = np.zeros(N)
    elements['nz'] = np.zeros(N)
    elements['f_dc_0'] = f_dc[:, 0, 0]
    elements['f_dc_1'] = f_dc[:, 0, 1]
    elements['f_dc_2'] = f_dc[:, 0, 2]
    elements['opacity'] = opacity_raw[:, 0]
    elements['scale_0'] = scales[:, 0]
    elements['scale_1'] = scales[:, 1]
    elements['scale_2'] = scales[:, 2]
    elements['rot_0'] = rotations[:, 0]
    elements['rot_1'] = rotations[:, 1]
    elements['rot_2'] = rotations[:, 2]
    elements['rot_3'] = rotations[:, 3]
    # Fill rest SH with zeros
    for i in range(45):
        elements[f'f_rest_{i}'] = np.zeros(N)

    vertex = PlyElement.describe(elements, 'vertex')
    PlyData([vertex]).write(path)
