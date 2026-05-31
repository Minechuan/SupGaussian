'''python simulation.py --model_path ./model/ficus_whitebg-trained/ --prompt "a basketball falling down" --output_path ./output --physics_config ./config/ficus_config.json '''


import sys

sys.path.append("gaussian-splatting")

import argparse
import math
import cv2
import torch
from torch import nn
import torch.nn.functional as F
import os
import numpy as np
import json
from tqdm import tqdm
from omegaconf import OmegaConf

# Gaussian splatting dependencies
from utils.sh_utils import eval_sh
from scene.gaussian_model import GaussianModel
from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from scene.cameras import Camera as GSCamera
from gaussian_renderer import render, GaussianModel
from utils.system_utils import searchForMaxIteration
from utils.graphics_utils import focal2fov

# MPM dependencies
from mpm_solver_warp.engine_utils import *
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
from mpm_solver_warp.mpm_utils import sum_array, sum_mat33, sum_vec3, wp_clamp, update_param, update_param_linear, set_value
import warp as wp

# Particle filling dependencies
from particle_filling.filling import *

# Utils
from utils.decode_param import *
from utils.transformation_utils import *
from utils.camera_view_utils import *
from utils.render_utils import *
from utils.save_video import save_video
from utils.threestudio_utils import cleanup

from video_distillation.guidance import ModelscopeGuidance
from video_distillation.prompt_processors import ModelscopePromptProcessor

from termcolor import cprint
wp.init()
wp.config.verify_cuda = True

ti.init(arch=ti.cuda, device_memory_GB=8.0)


class PipelineParamsNoparse:
    """Same as PipelineParams but without argument parser."""

    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


def load_checkpoint(model_path, iteration=-1):
    # Find checkpoint
    checkpt_dir = os.path.join(model_path, "point_cloud")
    if iteration == -1:
        iteration = searchForMaxIteration(checkpt_dir)
    checkpt_path = os.path.join(
        checkpt_dir, f"iteration_{iteration}", "point_cloud.ply"
    )
    
    # sh_degree=0, if you use a 3D asset without spherical harmonics
    from plyfile import PlyData
    plydata = PlyData.read(checkpt_path)
    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    
    # Load guassians
    sh_degree = int(math.sqrt((len(extra_f_names)+3) // 3)) - 1
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(checkpt_path)
    return gaussians


def report_particle_bounds(mpm_solver, grid_lim, prefix=""):
    particle_x = wp.to_torch(mpm_solver.mpm_state.particle_x).detach()
    mins = particle_x.min(dim=0).values
    maxs = particle_x.max(dim=0).values
    out_of_domain = ((particle_x < 0.0) | (particle_x > grid_lim)).any(dim=1).sum()
    nan_count = torch.isnan(particle_x).any(dim=1).sum()
    inf_count = torch.isinf(particle_x).any(dim=1).sum()
    print(
        f"{prefix}particle bounds min={mins.tolist()} max={maxs.tolist()} "
        f"out_of_domain={int(out_of_domain)} / {particle_x.shape[0]} grid_lim={grid_lim} "
        f"nan={int(nan_count)} inf={int(inf_count)}"
    )


def report_tensor_finite(name, tensor, prefix=""):
    finite_mask = torch.isfinite(tensor)
    total = tensor.numel()
    finite = int(finite_mask.sum())
    print(f"{prefix}{name}: finite={finite}/{total}")


def render_inputs_are_safe(pos, cov3D, rot, batch, frame):
    checks = {
        "pos": pos,
        "cov3D": cov3D,
        "rot": rot,
    }
    for name, tensor in checks.items():
        if not torch.isfinite(tensor).all().item():
            cprint(
                f"WARNING: non-finite {name} before rasterize at batch={batch}, frame={frame}.",
                "red",
            )
            return False

    pos_abs_max = torch.max(torch.abs(pos)).item()
    cov_abs_max = torch.max(torch.abs(cov3D)).item()
    rot_abs_max = torch.max(torch.abs(rot)).item()
    if pos_abs_max > 1e4 or cov_abs_max > 1e3 or rot_abs_max > 1e3:
        cprint(
            f"WARNING: abnormal raster inputs at batch={batch}, frame={frame}: "
            f"pos_abs_max={pos_abs_max}, cov_abs_max={cov_abs_max}, rot_abs_max={rot_abs_max}. "
            "Skipping this epoch to avoid rasterizer OOM.",
            "red",
        )
        return False
    return True


def get_parameter_clip_bounds(material_params, name):
    bounds = material_params["parameter_clip"][name]
    return float(bounds["lower"]), float(bounds["upper"])


def get_nu_clip_bounds(material_params):
    lower, upper = get_parameter_clip_bounds(material_params, "nu")
    safe_lower = max(lower, 1e-3)
    safe_upper = min(upper, 0.49)
    if safe_lower != lower or safe_upper != upper:
        cprint(
            f"WARNING: nu clip [{lower}, {upper}] is outside the physical range. "
            f"Using [{safe_lower}, {safe_upper}] instead.",
            "yellow",
        )
    if safe_lower >= safe_upper:
        raise ValueError(f"Invalid nu clip range after safety clamp: [{safe_lower}, {safe_upper}]")
    return safe_lower, safe_upper


def get_param_lr(material_params, name):
    return float(material_params.get("param_lr", {}).get(name, 0.1))


def should_optimize_param(material_params, name):
    return name in material_params["param"]


def normalize_param_grad(grad):
    grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
    max_grad, min_grad = torch.max(grad), torch.min(grad)
    if max_grad - min_grad != 0:
        return (grad - min_grad) / (max_grad - min_grad) - 0.5
    return torch.zeros_like(grad)


def raw_grad_is_usable(name, grad):
    if not torch.isfinite(grad).all().item():
        cprint(
            f"WARNING: raw_grad_{name} contains NaN/Inf. "
            f"Skipping {name} update this epoch to avoid parameter blow-up.",
            "red",
        )
        return False
    return True


def get_mpm_param_arrays(mpm_solver):
    return {
        "E": mpm_solver.mpm_model.E,
        "nu": mpm_solver.mpm_model.nu,
        "mu_N": mpm_solver.mpm_model.mu_N,
        "lam_N": mpm_solver.mpm_model.lam_N,
        "viscosity": mpm_solver.mpm_model.viscosity,
    }


def snapshot_mpm_params(mpm_solver):
    return {
        name: wp.to_torch(param).detach().clone()
        for name, param in get_mpm_param_arrays(mpm_solver).items()
    }


def restore_mpm_params(mpm_solver, snapshot, device):
    for name, param in get_mpm_param_arrays(mpm_solver).items():
        value = snapshot[name].to(device=device)
        wp.launch(set_value, dim=value.shape[0], inputs=[param, wp.from_torch(value)], device=device)


def is_zero_tensor(tensor):
    return torch.max(torch.abs(tensor)).item() == 0.0


def warn_if_zero_grads(title, grads, missing_grads=None):
    if len(grads) == 0:
        cprint(f"WARNING: no {title} gradients were selected for checking.", "yellow")
        return
    missing_grads = missing_grads or {}
    missing_grad_names = [name for name, is_missing in missing_grads.items() if is_missing]
    zero_grad_names = [name for name, grad in grads.items() if is_zero_tensor(grad)]
    if len(missing_grad_names) > 0:
        cprint(
            f"WARNING: autograd returned None for {title} gradients: {missing_grad_names}. "
            "This usually means the loss is disconnected from these tensors.",
            "red",
        )
    if len(zero_grad_names) == len(grads):
        cprint(
            f"WARNING: all {title} gradients are zero: {zero_grad_names}. "
            "If autograd did not return None, the graph is connected but the local derivative is numerically zero.",
            "red",
        )
    elif len(zero_grad_names) > 0:
        cprint(
            f"WARNING: zero {title} gradients detected for: {zero_grad_names}",
            "yellow",
        )


def print_tensor_autograd_diagnostics(name, tensor):
    grad_fn_name = type(tensor.grad_fn).__name__ if tensor.grad_fn is not None else None
    finite_count = int(torch.isfinite(tensor).sum().item())
    total = tensor.numel()
    abs_max = torch.max(torch.abs(torch.nan_to_num(tensor.detach()))).item()
    print(
        f"{name}: shape={tuple(tensor.shape)}, requires_grad={tensor.requires_grad}, "
        f"is_leaf={tensor.is_leaf}, grad_fn={grad_fn_name}, finite={finite_count}/{total}, "
        f"abs_max={abs_max}"
    )


def print_render_graph_diagnostics(loss, img_list, particle_x, particle_cov, particle_R):
    cprint("Gradient graph diagnostics", "yellow")
    print_tensor_autograd_diagnostics("loss", loss.reshape(1))
    print_tensor_autograd_diagnostics("img_list", img_list)
    print_tensor_autograd_diagnostics("particle_x", particle_x)
    print_tensor_autograd_diagnostics("particle_cov", particle_cov)
    print_tensor_autograd_diagnostics("particle_R", particle_R)


def print_image_grad_diagnostics(grad_img):
    cprint("Guidance image-gradient diagnostics", "yellow")
    if grad_img is None:
        cprint("WARNING: loss is disconnected from img_list; image gradient is None.", "red")
        return
    flat = grad_img.detach().reshape(grad_img.shape[0], -1)
    frame_abs_mean = flat.abs().mean(dim=1)
    frame_abs_max = flat.abs().max(dim=1).values
    nonzero_frames = (frame_abs_max > 0.0).nonzero(as_tuple=False).flatten().tolist()
    print(f"img_grad nonzero_frames={nonzero_frames} / {grad_img.shape[0]}")
    print(f"img_grad frame_abs_mean={frame_abs_mean.tolist()}")
    print(f"img_grad frame_abs_max={frame_abs_max.tolist()}")


def print_param_clip_diagnostics(mpm_solver, material_params):
    param_arrays = {
        "E": mpm_solver.mpm_model.E,
        "nu": mpm_solver.mpm_model.nu,
        "mu_N": mpm_solver.mpm_model.mu_N,
        "lam_N": mpm_solver.mpm_model.lam_N,
        "viscosity": mpm_solver.mpm_model.viscosity,
    }
    eps = 1e-6
    any_at_clip = False
    cprint("MPM parameter clip diagnostics", "yellow")
    for name in material_params["param"]:
        param = torch.clamp(wp.to_torch(param_arrays[name]).detach(), min=1e-30)
        value_min = torch.min(param).item()
        value_max = torch.max(param).item()
        total = param.numel()
        if name == "nu":
            lower, upper = get_nu_clip_bounds(material_params)
            lower_count = int((param <= lower + eps).sum().item())
            upper_count = int((param >= upper - eps).sum().item())
            at_clip = lower_count > 0 or upper_count > 0
            any_at_clip = any_at_clip or at_clip
            status = "AT_CLIP" if at_clip else "not_at_clip"
            print(
                f"{name}: {status}, value_min={value_min}, value_max={value_max}, "
                f"clip=[{lower}, {upper}], lower_count={lower_count}/{total}, "
                f"upper_count={upper_count}/{total}"
            )
            if at_clip:
                cprint(
                    f"WARNING: {name} is at direct clip boundary "
                    f"(lower {lower_count}/{total}, upper {upper_count}/{total}, "
                    f"range=[{lower}, {upper}]).",
                    "yellow",
                )
            continue
        log_param = torch.log10(param)
        lower, upper = get_parameter_clip_bounds(material_params, name)
        lower_count = int((log_param <= lower + eps).sum().item())
        upper_count = int((log_param >= upper - eps).sum().item())
        log_min = torch.min(log_param).item()
        log_max = torch.max(log_param).item()
        at_clip = lower_count > 0 or upper_count > 0
        any_at_clip = any_at_clip or at_clip
        status = "AT_CLIP" if at_clip else "not_at_clip"
        print(
            f"{name}: {status}, value_min={value_min}, value_max={value_max}, "
            f"log10_min={log_min}, log10_max={log_max}, clip=[{lower}, {upper}], "
            f"lower_count={lower_count}/{total}, upper_count={upper_count}/{total}"
        )
        if lower_count > 0 or upper_count > 0:
            cprint(
                f"WARNING: {name} is at log10 clip boundary "
                f"(lower {lower_count}/{total}, upper {upper_count}/{total}, "
                f"range=[{lower}, {upper}]). Gradients may be zeroed by clipping.",
                "yellow",
            )
    if not any_at_clip:
        cprint(
            "No optimized MPM parameter is currently at a configured clip boundary. "
            "The zero gradient is more likely from graph disconnection or a zero local derivative.",
            "green",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--physics_config", type=str, required=True)
    parser.add_argument("--guidance_config", type=str, default="./config/guidance/guidance.yaml")
    parser.add_argument("--white_bg", type=bool, default=True)
    parser.add_argument("--output_ply", action="store_true")
    parser.add_argument("--output_h5", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        AssertionError("Model path does not exist!")
    if not os.path.exists(args.physics_config):
        AssertionError("Scene config does not exist!")
    if not os.path.exists(args.guidance_config):
        AssertionError("Scene config does not exist!")
    if args.output_path is not None and not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    # load scene config
    print("Loading scene config...")
    (
        material_params,
        bc_params,
        time_params,
        preprocessing_params,
        camera_params,
    ) = decode_param_json(args.physics_config)

    # load gaussians
    print("Loading gaussians...")
    model_path = args.model_path
    gaussians = load_checkpoint(model_path)
    pipeline = PipelineParamsNoparse()
    pipeline.compute_cov3D_python = True
    background = (
        torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda")
        if args.white_bg
        else torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    )

    # init the scene
    print("Initializing scene and pre-processing...")
    params = load_params_from_gs(gaussians, pipeline)

    init_pos = params["pos"]
    init_cov = params["cov3D_precomp"]
    init_screen_points = params["screen_points"]
    init_opacity = params["opacity"]
    init_shs = params["shs"]

    # throw away low opacity kernels
    mask = init_opacity[:, 0] > preprocessing_params["opacity_threshold"]
    init_pos = init_pos[mask, :]
    init_cov = init_cov[mask, :]
    init_opacity = init_opacity[mask, :]
    init_screen_points = init_screen_points[mask, :]
    init_shs = init_shs[mask, :]
    
    # optimize moving parts only
    unselected_pos, unselected_cov, unselected_opacity, unselected_shs = (
        None,
        None,
        None,
        None,
    )
    moving_pts_path = os.path.join(model_path, "moving_part_points.ply")
    if os.path.exists(moving_pts_path):
        import point_cloud_utils as pcu
        moving_pts = pcu.load_mesh_v(moving_pts_path)
        moving_pts = torch.from_numpy(moving_pts).float().to("cuda")
        # moving_pts = apply_rotations(moving_pts, rotation_matrices)
        freeze_mask = find_far_points(
            init_pos, moving_pts, thres=0.05
        ).bool()
        moving_pts.to("cpu")
        unselected_pos = init_pos[freeze_mask, :]
        unselected_cov = init_cov[freeze_mask, :]
        unselected_opacity = init_opacity[freeze_mask, :]
        unselected_shs = init_shs[freeze_mask, :]

        init_pos = init_pos[~freeze_mask, :]
        init_cov = init_cov[~freeze_mask, :]
        init_opacity = init_opacity[~freeze_mask, :]
        init_shs = init_shs[~freeze_mask, :]

    # rorate and translate object
    if args.debug:
        if not os.path.exists("./log"):
            os.makedirs("./log")
        particle_position_tensor_to_ply(
            init_pos,
            "./log/init_particles.ply",
        )
    rotation_matrices = generate_rotation_matrices(
        torch.tensor(preprocessing_params["rotation_degree"]),
        preprocessing_params["rotation_axis"],
    )
    rotated_pos = apply_rotations(init_pos, rotation_matrices)

    if args.debug:
        particle_position_tensor_to_ply(rotated_pos, "./log/rotated_particles.ply")

    # select a sim area and save params of unslected particles
    if preprocessing_params["sim_area"] is not None:
        boundary = preprocessing_params["sim_area"]
        assert len(boundary) == 6
        mask = torch.ones(rotated_pos.shape[0], dtype=torch.bool).to(device="cuda")
        for i in range(3):
            mask = torch.logical_and(mask, rotated_pos[:, i] > boundary[2 * i])
            mask = torch.logical_and(mask, rotated_pos[:, i] < boundary[2 * i + 1])

        unselected_pos = init_pos[~mask, :]
        unselected_cov = init_cov[~mask, :]
        unselected_opacity = init_opacity[~mask, :]
        unselected_shs = init_shs[~mask, :]

        rotated_pos = rotated_pos[mask, :]
        init_cov = init_cov[mask, :]
        init_opacity = init_opacity[mask, :]
        init_shs = init_shs[mask, :]

    transformed_pos, scale_origin, original_mean_pos = transform2origin(rotated_pos)
    transformed_pos = shift2center111(transformed_pos)

    # modify covariance matrix accordingly
    init_cov = apply_cov_rotations(init_cov, rotation_matrices)
    init_cov = scale_origin * scale_origin * init_cov

    if args.debug:
        particle_position_tensor_to_ply(
            transformed_pos,
            "./log/transformed_particles.ply",
        )

    # fill particles if needed
    gs_num = transformed_pos.shape[0]
    device = "cuda:0"
    filling_params = preprocessing_params["particle_filling"]

    if filling_params is not None:
        print("Filling internal particles...")
        mpm_init_pos = fill_particles(
            pos=transformed_pos,
            opacity=init_opacity,
            cov=init_cov,
            grid_n=filling_params["n_grid"],
            max_samples=filling_params["max_particles_num"],
            grid_dx=material_params["grid_lim"] / filling_params["n_grid"],
            density_thres=filling_params["density_threshold"],
            search_thres=filling_params["search_threshold"],
            max_particles_per_cell=filling_params["max_partciels_per_cell"],
            search_exclude_dir=filling_params["search_exclude_direction"],
            ray_cast_dir=filling_params["ray_cast_direction"],
            boundary=filling_params["boundary"],
            smooth=filling_params["smooth"],
        ).to(device=device)

        if args.debug:
            particle_position_tensor_to_ply(mpm_init_pos, "./log/filled_particles.ply")
    else:
        mpm_init_pos = transformed_pos.to(device=device)

    # densify for high-frequency elastic objects
    init_len = mpm_init_pos.shape[0]
    # new_pts = []
    # for pt in mpm_init_pos:
    #     if pt[2] < 1.4 and pt[2] > 0.6:
    #         new_pts.append([pt[0]+0.05, pt[1], pt[2]])
    #         new_pts.append([pt[0]-0.05, pt[1], pt[2]])
    #         new_pts.append([pt[0], pt[1]+0.05, pt[2]])
    #         new_pts.append([pt[0], pt[1]-0.05, pt[2]])
    #         new_pts.append([pt[0]+0.05, pt[1]+0.05, pt[2]])
    #         new_pts.append([pt[0]+0.1, pt[1]-0.1, pt[2]])
    #         new_pts.append([pt[0]-0.1, pt[1]+0.1, pt[2]])
    #         new_pts.append([pt[0]-0.05, pt[1]-0.05, pt[2]])
    # mpm_init_pos = torch.concat([mpm_init_pos, torch.tensor(new_pts).to(device)]).to(torch.float32)

    # init the mpm solver
    print("Initializing MPM solver and setting up boundary conditions...")
    mpm_init_vol = get_particle_volume(
        mpm_init_pos,
        material_params["n_grid"],
        material_params["grid_lim"] / material_params["n_grid"],
        unifrom=material_params["material"] == "sand",
    ).to(device=device)

    if filling_params is not None and filling_params["visualize"] == True:
        shs, opacity, mpm_init_cov = init_filled_particles(
            mpm_init_pos[:gs_num],
            init_shs,
            init_cov,
            init_opacity,
            mpm_init_pos[gs_num:],
        )
        _pos = apply_inverse_rotations(
                undotransform2origin(
                    undoshift2center111(mpm_init_pos[gs_num:]), scale_origin, original_mean_pos
                ),
                rotation_matrices,
            )
        print(gaussians._xyz.shape)
        gaussians._xyz = nn.Parameter(torch.tensor(torch.cat([gaussians._xyz, _pos], 0), dtype=torch.float, device="cuda").requires_grad_(True))
        _features_dc = torch.zeros((_pos.shape[0], 1, 3)).to("cuda:0")
        print(gaussians._features_dc.shape)
        gaussians._features_dc = nn.Parameter(torch.tensor(torch.cat([gaussians._features_dc, _features_dc], 0), dtype=torch.float, device="cuda").requires_grad_(True))
        _features_rest = torch.zeros((_pos.shape[0], 15, 3)).to("cuda:0")
        print(gaussians._features_rest.shape)
        gaussians._features_rest = nn.Parameter(torch.tensor(torch.cat([gaussians._features_rest, _features_rest], 0), dtype=torch.float, device="cuda").requires_grad_(True))
        _opacity = torch.zeros((_pos.shape[0], 1)).to("cuda:0")
        gaussians._opacity = nn.Parameter(torch.tensor(torch.cat([gaussians._opacity, _opacity], 0), dtype=torch.float, device="cuda").requires_grad_(True))
        _scaling = torch.zeros((_pos.shape[0], 3)).to("cuda:0")
        gaussians._scaling = nn.Parameter(torch.tensor(torch.cat([gaussians._scaling, _scaling], 0), dtype=torch.float, device="cuda").requires_grad_(True))
        _rotation = torch.zeros((_pos.shape[0], 4)).to("cuda:0")
        gaussians._rotation = nn.Parameter(torch.tensor(torch.cat([gaussians._rotation, _rotation], 0), dtype=torch.float, device="cuda").requires_grad_(True))

        gs_num = mpm_init_pos.shape[0]
    else:
        mpm_init_cov = torch.zeros((mpm_init_pos.shape[0], 6), device=device)
        mpm_init_cov[:gs_num] = init_cov
        shs = init_shs
        opacity = init_opacity

    if args.debug:
        print("check *.ply files to see if it's ready for simulation")

    # set up the mpm solver
    mpm_solver = MPM_Simulator_WARP(10)
    mpm_solver.load_initial_data_from_torch(
        mpm_init_pos,
        mpm_init_vol,
        mpm_init_cov,
        n_grid=material_params["n_grid"],
        grid_lim=material_params["grid_lim"],
    )
    mpm_solver.set_parameters_dict(material_params)
    last_stable_mpm_params = snapshot_mpm_params(mpm_solver)

    # Note: boundary conditions may depend on mass, so the order cannot be changed!
    set_boundary_conditions(mpm_solver, bc_params, time_params)

    if args.debug:
        report_particle_bounds(
            mpm_solver,
            material_params["grid_lim"],
            prefix="[init] ",
        )
    
    # moving_pts_path = os.path.join(model_path, "moving_part_points.ply")
    # if os.path.exists(moving_pts_path):
    #     import point_cloud_utils as pcu
    #     moving_pts = pcu.load_mesh_v(moving_pts_path)
    #     moving_pts = torch.from_numpy(moving_pts).float().to("cuda")
    #     moving_pts = apply_rotations(moving_pts, rotation_matrices)
    #     moving_pts, moving_scale_origin, moving_original_mean_pos = transform2origin(moving_pts)
    #     moving_pts = shift2center111(moving_pts)
    #     get_particle_volume(
    #         moving_pts,
    #         material_params["n_grid"],
    #         material_params["grid_lim"] / material_params["n_grid"],
    #         unifrom=False,
    #     )
    #     freeze_mask = find_far_points(
    #         mpm_init_pos, moving_pts, thres=0.5
    #     ).bool()
    #     freeze_pts = mpm_init_pos[freeze_mask, :]
    #     apply_grid_bc_w_freeze_pts(
    #         mpm_solver.mpm_model.n_grid, mpm_solver.mpm_model.grid_lim, freeze_pts, mpm_solver
    #     )
    
    tape = wp.Tape()

    # mpm_solver.finalize_mu_lam()

    # camera setting
    mpm_space_viewpoint_center = (
        torch.tensor(camera_params["mpm_space_viewpoint_center"]).reshape((1, 3)).cuda()
    )
    mpm_space_vertical_upward_axis = (
        torch.tensor(camera_params["mpm_space_vertical_upward_axis"])
        .reshape((1, 3))
        .cuda()
    )
    (
        viewpoint_center_worldspace,
        observant_coordinates,
    ) = get_center_view_worldspace_and_observant_coordinate(
        mpm_space_viewpoint_center,
        mpm_space_vertical_upward_axis,
        rotation_matrices,
        scale_origin,
        original_mean_pos,
    )

    # run the simulation
    if args.output_ply or args.output_h5:
        directory_to_save = os.path.join(args.output_path, "simulation_ply")
        if not os.path.exists(directory_to_save):
            os.makedirs(directory_to_save)
        save_data_at_frame(
            mpm_solver,
            directory_to_save,
            0,
            save_to_ply=args.output_ply,
            save_to_h5=args.output_h5,
        )

    substep_dt = time_params["substep_dt"]
    frame_dt = time_params["frame_dt"]
    frame_num = time_params["frame_num"]
    step_per_frame = int(frame_dt / substep_dt)
    opacity_render = opacity
    shs_render = shs
    height = None
    width = None

    stage_num = 4
    frame_per_stage = 4
    
    yaml_confs = OmegaConf.load(args.guidance_config)
    yaml_confs.prompt_processor.prompt = args.prompt
    if (
        hasattr(yaml_confs.guidance, "low_ram_vae")
        and yaml_confs.guidance.low_ram_vae is not None
        and yaml_confs.guidance.low_ram_vae > 0
        and yaml_confs.guidance.low_ram_vae < frame_per_stage
    ):
        cprint(
            f"WARNING: guidance.low_ram_vae={yaml_confs.guidance.low_ram_vae} "
            f"keeps gradients for only part of the {frame_per_stage} rendered frames. "
            "The low-RAM VAE path now keeps gradients for the last frames deterministically.",
            "yellow",
        )
    cprint("The prompt is: " + args.prompt, "yellow")
    guidance = ModelscopeGuidance(yaml_confs.guidance)
    vae_grad_frames = frame_per_stage
    if hasattr(yaml_confs.guidance, "low_ram_vae") and yaml_confs.guidance.low_ram_vae is not None:
        if yaml_confs.guidance.low_ram_vae > 0:
            vae_grad_frames = min(yaml_confs.guidance.low_ram_vae, frame_per_stage)
    vae_grad_start_frame = frame_per_stage - vae_grad_frames
    cprint(
        f"VAE gradients are kept for training frames [{vae_grad_start_frame}, {frame_per_stage - 1}]",
        "yellow",
    )
    prompt_processor = ModelscopePromptProcessor(yaml_confs.prompt_processor)
    prompt_utils = prompt_processor()
    


    '''
    Begin
    
    '''


    cprint("The initial value of the parameters", 'yellow')
    print("E: ", torch.mean(wp.to_torch(mpm_solver.mpm_model.E)).item())
    print("mu_N: ", torch.mean(wp.to_torch(mpm_solver.mpm_model.mu_N)).item())
    print("lam_N: ", torch.mean(wp.to_torch(mpm_solver.mpm_model.lam_N)).item())
    print("viscosity: ", torch.mean(wp.to_torch(mpm_solver.mpm_model.viscosity)).item())
    cprint("The optimized MPM parameters", "yellow")
    print(material_params["param"])
    cprint("The clip range of the learnable MPM parameters", "yellow")
    for param_name in ["E", "nu", "mu_N", "lam_N", "viscosity"]:
        if param_name == "nu":
            lower, upper = get_nu_clip_bounds(material_params)
            print(f"{param_name}: lower={lower}, upper={upper} (direct value range)")
        else:
            lower, upper = get_parameter_clip_bounds(material_params, param_name)
            print(f"{param_name}: lower={lower}, upper={upper} (log10 range)")
    cprint("The learning rate of the learnable MPM parameters", "yellow")
    for param_name in ["E", "nu", "mu_N", "lam_N", "viscosity"]:
        print(f"{param_name}: lr={get_param_lr(material_params, param_name)}")

    mpm_solver.finalize_mu_lam()
    with torch.no_grad():
        for frame in tqdm(range(stage_num * frame_per_stage)):
            
            current_camera = get_camera_view(
                model_path,
                default_camera_index=camera_params["default_camera_index"],
                center_view_world_space=viewpoint_center_worldspace,
                observant_coordinates=observant_coordinates,
                show_hint=camera_params["show_hint"],
                init_azimuthm=camera_params["init_azimuthm"],
                init_elevation=camera_params["init_elevation"],
                init_radius=camera_params["init_radius"],
                move_camera=camera_params["move_camera"],
                current_frame=frame,
                delta_a=camera_params["delta_a"],
                delta_e=camera_params["delta_e"],
                delta_r=camera_params["delta_r"],
            )
            rasterize = initialize_resterize(
                current_camera, gaussians, pipeline, background
            )
            
            for _ in range(step_per_frame):
                mpm_solver.p2g2p(frame, substep_dt, device=device)

            pos = mpm_solver.export_particle_x_to_torch()[:gs_num].to(device)
            cov3D = mpm_solver.export_particle_cov_to_torch()
            rot = mpm_solver.export_particle_R_to_torch()
            
            cov3D = cov3D.view(-1, 6)[:gs_num].to(device)
            rot = rot.view(-1, 3, 3)[:gs_num].to(device)

            pos = pos[:init_len,:]
            pos = apply_inverse_rotations(
                undotransform2origin(
                    undoshift2center111(pos), scale_origin, original_mean_pos
                ),
                rotation_matrices,
            )
            cov3D = cov3D / (scale_origin * scale_origin)
            cov3D = apply_inverse_cov_rotations(cov3D, rotation_matrices)
            opacity = opacity_render
            shs = shs_render
            if preprocessing_params["sim_area"] is not None:
                pos = torch.cat([pos, unselected_pos], dim=0)
                cov3D = torch.cat([cov3D, unselected_cov], dim=0)
                opacity = torch.cat([opacity_render, unselected_opacity], dim=0)
                shs = torch.cat([shs_render, unselected_shs], dim=0)
            if os.path.exists(moving_pts_path):
                pos = torch.cat([pos, unselected_pos], dim=0)
                cov3D = torch.cat([cov3D, unselected_cov], dim=0)
                opacity = torch.cat([opacity_render, unselected_opacity], dim=0)
                shs = torch.cat([shs_render, unselected_shs], dim=0)

            colors_precomp = convert_SH(shs, current_camera, gaussians, pos, rot)
            rendering, raddi = rasterize(
                means3D=pos,
                means2D=init_screen_points,
                shs=None,
                colors_precomp=colors_precomp,
                opacities=opacity,
                scales=None,
                rotations=None,
                cov3D_precomp=cov3D,
            )
            
            cv2_img = rendering.permute(1, 2, 0).detach().cpu().numpy()
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            if height is None or width is None:
                height = cv2_img.shape[0] // 2 * 2
                width = cv2_img.shape[1] // 2 * 2
            assert args.output_path is not None
            cv2.imwrite(
                os.path.join(args.output_path, f"{frame}.png".rjust(8, "0")),
                255 * cv2_img,
            )
    save_video(args.output_path, os.path.join(args.output_path, 'initial_render.mp4'))
    del pos, cov3D, rot, rendering, raddi, colors_precomp, rasterize, current_camera
    cleanup()


    for batch in range(20):
        loss_value = 0.
        img_list = []
        tape.reset()
        with tape:
            mpm_solver.finalize_mu_lam()

        if args.debug and batch % 5 == 0:
            report_particle_bounds(
                mpm_solver,
                material_params["grid_lim"],
                prefix=f"[batch {batch}] ",
            )
        
        cprint(f"Solving MPM", "green")
        # 并不是每次都从初始状态开始模拟，而是根据当前的batch数，先模拟一段时间，让粒子移动到不同的位置，增加训练的多样性
        start_frame = batch % (frame_per_stage * stage_num)
        for _ in range(step_per_frame * start_frame):
            mpm_solver.p2g2p(None, substep_dt, device=device)
        

        cprint(f"Rendering and computing guidance loss", "green")
        # avg = mpm_solver.export_latest_average_stress_to_torch()
        # print("The mean average stress of elastic is: ", torch.mean(avg['elastic']).item())
        # print("The mean average stress of viscoelastic is: ", torch.mean(avg['viscoelastic']).item())
        epoch_failed = False
        particle_x = None
        particle_cov = None
        particle_R = None
        for frame in tqdm(range(frame_per_stage)):
            current_camera = get_camera_view(
                model_path,
                default_camera_index=camera_params["default_camera_index"],
                center_view_world_space=viewpoint_center_worldspace,
                observant_coordinates=observant_coordinates,
                show_hint=camera_params["show_hint"],
                init_azimuthm=camera_params["init_azimuthm"],
                init_elevation=camera_params["init_elevation"],
                init_radius=camera_params["init_radius"],
                move_camera=camera_params["move_camera"],
                current_frame=frame,
                delta_a=camera_params["delta_a"],
                delta_e=camera_params["delta_e"],
                delta_r=camera_params["delta_r"],
            )
            rasterize = initialize_resterize(
                current_camera, gaussians, pipeline, background
            )
            
            # simulate to the current frame
            for _ in range(step_per_frame):
                mpm_solver.p2g2p(frame, substep_dt, device=device)
            
            # rendering and compute loss
            keep_frame_grad = frame >= vae_grad_start_frame
            if keep_frame_grad:
                with tape:
                    mpm_solver.p2g2p(frame, substep_dt, device=device)

                    particle_x = torch.nan_to_num(
                        mpm_solver.export_particle_x_to_torch().to(device)
                    )
                    particle_cov = torch.nan_to_num(
                        mpm_solver.export_particle_cov_to_torch().to(device)
                    )
                    particle_R = torch.nan_to_num(
                        mpm_solver.export_particle_R_to_torch().to(device)
                    )

                    pos = particle_x[:gs_num]
                    cov3D = particle_cov.view(-1, 6)[:gs_num]
                    rot = particle_R.view(-1, 3, 3)[:gs_num]
            else:
                mpm_solver.p2g2p(frame, substep_dt, device=device)
                with torch.no_grad():
                    pos = torch.nan_to_num(
                        mpm_solver.export_particle_x_to_torch()[:gs_num].to(device)
                    )
                    cov3D = torch.nan_to_num(
                        mpm_solver.export_particle_cov_to_torch().view(-1, 6)[:gs_num].to(device)
                    )
                    rot = torch.nan_to_num(
                        mpm_solver.export_particle_R_to_torch().view(-1, 3, 3)[:gs_num].to(device)
                    )

            if args.debug:
                # 统计其中有限的元素个数
                if keep_frame_grad:
                    report_tensor_finite("particle_x", particle_x, prefix=f"[batch {batch} frame {frame}] ")
                    report_tensor_finite("particle_cov", particle_cov, prefix=f"[batch {batch} frame {frame}] ")
                    report_tensor_finite("particle_R", particle_R, prefix=f"[batch {batch} frame {frame}] ")


            
            pos = pos[:init_len,:]
            pos = apply_inverse_rotations( # 将mpm空间的粒子位置转换回原始空间，方便后续与高斯点云对齐
                undotransform2origin(
                    undoshift2center111(pos), scale_origin, original_mean_pos
                ),
                rotation_matrices,
            )
            cov3D = cov3D / (scale_origin * scale_origin)
            cov3D = apply_inverse_cov_rotations(cov3D, rotation_matrices)
            pos = torch.nan_to_num(pos)
            cov3D = torch.nan_to_num(cov3D)
            rot = torch.nan_to_num(rot)
            opacity = opacity_render
            shs = shs_render
            # 
            if preprocessing_params["sim_area"] is not None:
                pos = torch.cat([pos, unselected_pos], dim=0)
                cov3D = torch.cat([cov3D, unselected_cov], dim=0)
                opacity = torch.cat([opacity_render, unselected_opacity], dim=0)
                shs = torch.cat([shs_render, unselected_shs], dim=0)
            
            
            if os.path.exists(moving_pts_path):
                pos = torch.cat([pos, unselected_pos], dim=0)
                cov3D = torch.cat([cov3D, unselected_cov], dim=0)
                opacity = torch.cat([opacity_render, unselected_opacity], dim=0)
                shs = torch.cat([shs_render, unselected_shs], dim=0)

            if not render_inputs_are_safe(pos, cov3D, rot, batch, frame):
                epoch_failed = True
                break

            if keep_frame_grad:
                colors_precomp = convert_SH(shs, current_camera, gaussians, pos, rot)
                colors_precomp = torch.nan_to_num(colors_precomp)
                try:
                    rendering, raddi = rasterize(
                        means3D=pos,
                        means2D=pos,
                        shs=None,
                        colors_precomp=colors_precomp,
                        opacities=opacity,
                        scales=None,
                        rotations=None,
                        cov3D_precomp=cov3D,
                    )
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower():
                        raise
                    cprint(
                        f"WARNING: rasterizer OOM at batch={batch}, frame={frame}. "
                        "Skipping this epoch and restoring the last stable parameters.",
                        "red",
                    )
                    epoch_failed = True
                    cleanup()
                    break
                img_list.append(rendering)
            else:
                with torch.no_grad():
                    colors_precomp = convert_SH(shs, current_camera, gaussians, pos, rot)
                    colors_precomp = torch.nan_to_num(colors_precomp)
                    try:
                        rendering, raddi = rasterize(
                            means3D=pos,
                            means2D=pos,
                            shs=None,
                            colors_precomp=colors_precomp,
                            opacities=opacity,
                            scales=None,
                            rotations=None,
                            cov3D_precomp=cov3D,
                        )
                    except RuntimeError as exc:
                        if "out of memory" not in str(exc).lower():
                            raise
                        cprint(
                            f"WARNING: rasterizer OOM at batch={batch}, frame={frame}. "
                            "Skipping this epoch and restoring the last stable parameters.",
                            "red",
                        )
                        epoch_failed = True
                        cleanup()
                        break
                    img_list.append(rendering.detach())
                del pos, cov3D, rot, rendering, raddi, colors_precomp, rasterize, current_camera
                cleanup()

        if epoch_failed or particle_x is None or particle_cov is None or particle_R is None:
            cprint(
                "WARNING: skipping this epoch because no safe differentiable frame was rendered. "
                "Restoring last stable parameters and resetting MPM state.",
                "red",
            )
            restore_mpm_params(mpm_solver, last_stable_mpm_params, device)
            mpm_solver.reset_pos_from_torch(mpm_init_pos, mpm_init_vol, mpm_init_cov)
            tape.reset()
            del img_list
            if "pos" in locals():
                del pos
            if "cov3D" in locals():
                del cov3D
            if "rot" in locals():
                del rot
            if "rasterize" in locals():
                del rasterize
            if "current_camera" in locals():
                del current_camera
            cleanup()
            continue
	        
        # save image
        path = "./tmp_imgs/"
        if not os.path.exists(path):
            os.makedirs(path)
        for i in range(len(img_list)):
            cv2_img = img_list[i].permute(1, 2, 0).detach().cpu().numpy()
            cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(
                os.path.join(path, f"{frame}_{i}.png".rjust(8, "0")),
                255 * cv2_img,
            )

        
        loss = 0.
        img_list = torch.stack(img_list) # （16,3,H, W）
        # 
        guidance_out = guidance(img_list, 
                                prompt_utils, torch.Tensor([camera_params['init_elevation']]), 
                                torch.Tensor([camera_params['init_azimuthm']]), torch.Tensor([camera_params['init_radius']]), 
                                rgb_as_latents=False, num_frames=frame_per_stage, train_dynamic_camera=False
                                )
        print("The guidance output loss is: ", guidance_out['loss_sds_video'])
        for name, value in guidance_out.items():
            if name.startswith('loss_'):
                loss += value * 3e-4

        # 避免 loss 随着仿真帧数增加过大，导致梯度爆炸
        # Normalize the loss by the number of frames in each stage
        loss = loss / stage_num
        print("The loss is: ", loss.item())
        loss_value += loss.item()
        if args.debug:
            grad_img = torch.autograd.grad(
                loss,
                img_list,
                retain_graph=True,
                allow_unused=True,
            )[0]
            print_image_grad_diagnostics(grad_img)
        grad_x, grad_cov, grad_r = torch.autograd.grad(
            loss,
            [particle_x, particle_cov, particle_R],
            retain_graph=False,
            allow_unused=True,
        )
        missing_state_grads = {
            "grad_x": grad_x is None,
            "grad_cov": grad_cov is None,
            "grad_r": grad_r is None,
        }
        if grad_x is None:
            grad_x = torch.zeros_like(particle_x)
        if grad_cov is None:
            grad_cov = torch.zeros_like(particle_cov)
        if grad_r is None:
            grad_r = torch.zeros_like(particle_R)
        # The gradient of mpm
        loss_wp = wp.zeros(1, dtype=float, device=device, requires_grad=True)
        
        # The gradients of x
        grad_x_wp = wp.from_torch(grad_x, dtype=wp.vec3)
        
        # The gradients of cov are in 6D, we need to convert them to mat33
        grad_cov_wp = wp.from_torch(grad_cov)
        grad_r_wp = wp.from_torch(grad_r.reshape(-1, 3, 3).contiguous(), dtype=wp.mat33)
        cprint("The mean/abs/max value of the gradients: ", "yellow")
        print("grad_x: ", torch.mean(grad_x).item(), torch.mean(torch.abs(grad_x)).item(), torch.max(torch.abs(grad_x)).item())
        print("grad_cov: ", torch.mean(grad_cov).item(), torch.mean(torch.abs(grad_cov)).item(), torch.max(torch.abs(grad_cov)).item())
        print("grad_r: ", torch.mean(grad_r).item(), torch.mean(torch.abs(grad_r)).item(), torch.max(torch.abs(grad_r)).item())
        warn_if_zero_grads(
            "render-to-MPM state",
            {"grad_x": grad_x, "grad_cov": grad_cov, "grad_r": grad_r},
            missing_state_grads,
        )
        if is_zero_tensor(grad_x) and is_zero_tensor(grad_cov) and is_zero_tensor(grad_r):
            print_render_graph_diagnostics(loss, img_list, particle_x, particle_cov, particle_R)
        with tape:
            wp.launch(sum_vec3, mpm_solver.n_particles, [mpm_solver.mpm_state.particle_x, grad_x_wp], [loss_wp], device=device)
            wp.launch(sum_array, mpm_solver.n_particles*6, [mpm_solver.mpm_state.particle_cov, grad_cov_wp], [loss_wp], device=device)
            wp.launch(sum_mat33, mpm_solver.n_particles, [mpm_solver.mpm_state.particle_R, grad_r_wp], [loss_wp], device=device)
        tape.backward(loss=loss_wp)
        raw_grad_E = wp.to_torch(mpm_solver.mpm_model.E.grad)
        raw_grad_nu = wp.to_torch(mpm_solver.mpm_model.nu.grad)
        raw_grad_mu_N = wp.to_torch(mpm_solver.mpm_model.mu_N.grad)
        raw_grad_lam_N = wp.to_torch(mpm_solver.mpm_model.lam_N.grad)
        raw_grad_viscosity = wp.to_torch(mpm_solver.mpm_model.viscosity.grad)
        cprint("The raw mean/abs/max value of the MPM parameter gradients: ", "yellow")
        print("raw_grad_E: ", torch.mean(raw_grad_E).item(), torch.mean(torch.abs(raw_grad_E)).item(), torch.max(torch.abs(raw_grad_E)).item())
        print("raw_grad_nu: ", torch.mean(raw_grad_nu).item(), torch.mean(torch.abs(raw_grad_nu)).item(), torch.max(torch.abs(raw_grad_nu)).item())
        print("raw_grad_mu_N: ", torch.mean(raw_grad_mu_N).item(), torch.mean(torch.abs(raw_grad_mu_N)).item(), torch.max(torch.abs(raw_grad_mu_N)).item())
        print("raw_grad_lam_N: ", torch.mean(raw_grad_lam_N).item(), torch.mean(torch.abs(raw_grad_lam_N)).item(), torch.max(torch.abs(raw_grad_lam_N)).item())
        print("raw_grad_viscosity: ", torch.mean(raw_grad_viscosity).item(), torch.mean(torch.abs(raw_grad_viscosity)).item(), torch.max(torch.abs(raw_grad_viscosity)).item())
        warn_if_zero_grads(
            "raw MPM parameter",
            {
                name: grad
                for name, grad in {
                    "E": raw_grad_E,
                    "nu": raw_grad_nu,
                    "mu_N": raw_grad_mu_N,
                    "lam_N": raw_grad_lam_N,
                    "viscosity": raw_grad_viscosity,
                }.items()
                if should_optimize_param(material_params, name)
            },
        )
        optimized_raw_grads = {
            name: grad
            for name, grad in {
                "E": raw_grad_E,
                "nu": raw_grad_nu,
                "mu_N": raw_grad_mu_N,
                "lam_N": raw_grad_lam_N,
                "viscosity": raw_grad_viscosity,
            }.items()
            if should_optimize_param(material_params, name)
        }
        if len(optimized_raw_grads) > 0 and all(is_zero_tensor(grad) for grad in optimized_raw_grads.values()):
            print_param_clip_diagnostics(mpm_solver, material_params)
        usable_raw_grads = {
            name: raw_grad_is_usable(name, grad)
            for name, grad in optimized_raw_grads.items()
        }
        update_allowed = len(usable_raw_grads) > 0 and all(usable_raw_grads.values())
        if len(usable_raw_grads) > 0 and not update_allowed:
            cprint(
                "WARNING: non-finite MPM parameter gradients detected. "
                "Restoring the last stable MPM parameters and skipping this epoch's parameter update.",
                "red",
            )
            restore_mpm_params(mpm_solver, last_stable_mpm_params, device)
        elif update_allowed:
            last_stable_mpm_params = snapshot_mpm_params(mpm_solver)

        grad_E = raw_grad_E
        grad_E = normalize_param_grad(grad_E)
        E_lower, E_upper = get_parameter_clip_bounds(material_params, "E")
        if update_allowed and should_optimize_param(material_params, "E") and usable_raw_grads.get("E", False):
            wp.launch(update_param, mpm_solver.n_particles, [mpm_solver.mpm_model.E, wp.from_torch(grad_E), get_param_lr(material_params, "E"), E_upper, E_lower])

        grad_nu = raw_grad_nu
        grad_nu = normalize_param_grad(grad_nu)
        nu_lower, nu_upper = get_nu_clip_bounds(material_params)
        if update_allowed and should_optimize_param(material_params, "nu") and usable_raw_grads.get("nu", False):
            wp.launch(update_param_linear, mpm_solver.n_particles, [mpm_solver.mpm_model.nu, wp.from_torch(grad_nu), get_param_lr(material_params, "nu"), nu_lower, nu_upper])
        
        # add
        grad_mu_N = raw_grad_mu_N
        grad_mu_N = normalize_param_grad(grad_mu_N)
        mu_N_lower, mu_N_upper = get_parameter_clip_bounds(material_params, "mu_N")
        if update_allowed and should_optimize_param(material_params, "mu_N") and usable_raw_grads.get("mu_N", False):
            wp.launch(update_param, mpm_solver.n_particles, [mpm_solver.mpm_model.mu_N, wp.from_torch(grad_mu_N), get_param_lr(material_params, "mu_N"), mu_N_upper, mu_N_lower])
        
        grad_lam_N = raw_grad_lam_N
        grad_lam_N = normalize_param_grad(grad_lam_N)
        lam_N_lower, lam_N_upper = get_parameter_clip_bounds(material_params, "lam_N")
        if update_allowed and should_optimize_param(material_params, "lam_N") and usable_raw_grads.get("lam_N", False):
            wp.launch(update_param, mpm_solver.n_particles, [mpm_solver.mpm_model.lam_N, wp.from_torch(grad_lam_N), get_param_lr(material_params, "lam_N"), lam_N_upper, lam_N_lower])
        
        grad_viscosity = raw_grad_viscosity
        grad_viscosity = normalize_param_grad(grad_viscosity)
        viscosity_lower, viscosity_upper = get_parameter_clip_bounds(material_params, "viscosity")
        if update_allowed and should_optimize_param(material_params, "viscosity") and usable_raw_grads.get("viscosity", False):
            wp.launch(update_param, mpm_solver.n_particles, [mpm_solver.mpm_model.viscosity, wp.from_torch(grad_viscosity), get_param_lr(material_params, "viscosity"), viscosity_upper, viscosity_lower])
        

        cprint("The mean value of the gradients: ", "yellow")
        print("grad_E: ", torch.mean(grad_E).item())
        print("grad_nu: ", torch.mean(grad_nu).item())
        print("grad_mu_N: ", torch.mean(grad_mu_N).item())
        print("grad_lam_N: ", torch.mean(grad_lam_N).item())
        print("grad_viscosity: ", torch.mean(grad_viscosity).item())
        
        cprint("The mean value of the parameters after update: ", "yellow")        
        print("E: ", torch.mean(wp.to_torch(mpm_solver.mpm_model.E)).item())
        print("nu: ", torch.mean(wp.to_torch(mpm_solver.mpm_model.nu)).item())
        print("mu_N: ", torch.mean(wp.to_torch(mpm_solver.mpm_model.mu_N)).item())
        print("lam_N: ", torch.mean(wp.to_torch(mpm_solver.mpm_model.lam_N)).item())
        print("viscosity: ", torch.mean(wp.to_torch(mpm_solver.mpm_model.viscosity)).item())


        cprint(f"Updating MPM parameters", "green")
        mpm_solver.reset_pos_from_torch(mpm_init_pos, mpm_init_vol, mpm_init_cov)
        tape.reset()
        if args.debug:
            del grad_img
        del guidance_out, loss, img_list, loss_wp, grad_x, grad_cov, grad_r
        del grad_x_wp, grad_cov_wp, grad_r_wp
        del raw_grad_E, raw_grad_nu, raw_grad_mu_N, raw_grad_lam_N, raw_grad_viscosity
        del missing_state_grads, optimized_raw_grads, usable_raw_grads, update_allowed
        del grad_E, grad_nu, grad_mu_N, grad_lam_N, grad_viscosity
        del particle_x, particle_cov, particle_R, pos, cov3D, rot, rendering, raddi, colors_precomp, rasterize, current_camera
        cleanup()
        if batch % 2 == 0:
            mpm_solver.finalize_mu_lam()
            with torch.no_grad():
                for frame in tqdm(range(stage_num * frame_per_stage)):
                    current_camera = get_camera_view(
                        model_path,
                        default_camera_index=camera_params["default_camera_index"],
                        center_view_world_space=viewpoint_center_worldspace,
                        observant_coordinates=observant_coordinates,
                        show_hint=camera_params["show_hint"],
                        init_azimuthm=camera_params["init_azimuthm"],
                        init_elevation=camera_params["init_elevation"],
                        init_radius=camera_params["init_radius"],
                        move_camera=camera_params["move_camera"],
                        current_frame=frame,
                        delta_a=camera_params["delta_a"],
                        delta_e=camera_params["delta_e"],
                        delta_r=camera_params["delta_r"],
                    )
                    rasterize = initialize_resterize(
                        current_camera, gaussians, pipeline, background
                    )
                    
                    for _ in range(step_per_frame):
                        mpm_solver.p2g2p(frame, substep_dt, device=device)

                    pos = mpm_solver.export_particle_x_to_torch()[:gs_num].to(device)
                    cov3D = mpm_solver.export_particle_cov_to_torch()
                    rot = mpm_solver.export_particle_R_to_torch()
                    
                    cov3D = cov3D.view(-1, 6)[:gs_num].to(device)
                    rot = rot.view(-1, 3, 3)[:gs_num].to(device)

                    pos = pos[:init_len,:]
                    pos = apply_inverse_rotations(
                        undotransform2origin(
                            undoshift2center111(pos), scale_origin, original_mean_pos
                        ),
                        rotation_matrices,
                    )
                    cov3D = cov3D / (scale_origin * scale_origin)
                    cov3D = apply_inverse_cov_rotations(cov3D, rotation_matrices)
                    opacity = opacity_render
                    shs = shs_render
                    if preprocessing_params["sim_area"] is not None:
                        pos = torch.cat([pos, unselected_pos], dim=0)
                        cov3D = torch.cat([cov3D, unselected_cov], dim=0)
                        opacity = torch.cat([opacity_render, unselected_opacity], dim=0)
                        shs = torch.cat([shs_render, unselected_shs], dim=0)
                    if os.path.exists(moving_pts_path):
                        pos = torch.cat([pos, unselected_pos], dim=0)
                        cov3D = torch.cat([cov3D, unselected_cov], dim=0)
                        opacity = torch.cat([opacity_render, unselected_opacity], dim=0)
                        shs = torch.cat([shs_render, unselected_shs], dim=0)

                    colors_precomp = convert_SH(shs, current_camera, gaussians, pos, rot)
                    rendering, raddi = rasterize(
                        means3D=pos,
                        means2D=init_screen_points,
                        shs=None,
                        colors_precomp=colors_precomp,
                        opacities=opacity,
                        scales=None,
                        rotations=None,
                        cov3D_precomp=cov3D,
                    )
                    
                    cv2_img = rendering.permute(1, 2, 0).detach().cpu().numpy()
                    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
                    if height is None or width is None:
                        height = cv2_img.shape[0] // 2 * 2
                        width = cv2_img.shape[1] // 2 * 2
                    assert args.output_path is not None
                    cv2.imwrite(
                        os.path.join(args.output_path, f"{frame}.png".rjust(8, "0")),
                        255 * cv2_img,
                    )
            save_video(args.output_path, os.path.join(args.output_path, 'video%02d.mp4' % batch))
            del pos, cov3D, rot, rendering, raddi, colors_precomp, rasterize, current_camera
            cleanup()
