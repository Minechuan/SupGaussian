"""Wrapper around diff-gaussian-rasterization CUDA kernel."""

import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer


def render(viewpoint_camera, pc, pipe, bg_color, scaling_modifier=1.0, override_color=None):
    """
    Render Gaussians from a given camera viewpoint.

    Args:
        viewpoint_camera: Camera object (my_3dgs.camera.Camera or MiniCam)
        pc: GaussianModel
        pipe: PipelineParams (render settings)
        bg_color: [3] background color tensor
        scaling_modifier: float
        override_color: [N, 3] optional color override
    Returns:
        dict with "render", "viewspace_points", "visibility_filter", "radii"
    """
    # Set up rasterization
    tanfovx = torch.tan(torch.tensor(viewpoint_camera.FoVx * 0.5, device="cuda"))
    tanfovy = torch.tan(torch.tensor(viewpoint_camera.FoVy * 0.5, device="cuda"))

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = torch.zeros_like(means3D, requires_grad=True, device="cuda")
    # means2D needs retain_grad for densification stats
    means2D.retain_grad()
    opacity = pc.get_opacity

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    shs = None
    colors_precomp = None
    if override_color is None:
        shs = pc.get_features
    else:
        colors_precomp = override_color

    # Render
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    return {
        "render": rendered_image,
        "viewspace_points": means2D,
        "visibility_filter": radii > 0,
        "radii": radii,
    }
