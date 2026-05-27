import sys

sys.path.append("gaussian-splatting")

import argparse
import math
import os

import cv2
import numpy as np
import torch
import warp as wp
from omegaconf import OmegaConf
from tqdm import tqdm

import taichi as ti
from torch import nn

from diff_gaussian_rasterization import GaussianRasterizer, GaussianRasterizationSettings
from mpm_solver_warp.engine_utils import *
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
from mpm_solver_warp.mpm_utils import sum_array, sum_mat33, sum_vec3
from particle_filling.filling import fill_particles, get_particle_volume, init_filled_particles, particle_position_tensor_to_ply
from scene.gaussian_model import GaussianModel as SceneGaussianModel
from utils.camera_view_utils import get_camera_view
from utils.decode_param import decode_param_json, find_far_points, set_boundary_conditions
from utils.render_utils import convert_SH, initialize_resterize, load_params_from_gs
from utils.save_video import save_video
from utils.system_utils import searchForMaxIteration
from utils.transformation_utils import (
    apply_cov_rotations,
    apply_inverse_cov_rotations,
    apply_inverse_rotations,
    apply_rotations,
    generate_rotation_matrices,
    shift2center111,
    transform2origin,
    undoshift2center111,
    undotransform2origin,
    get_center_view_worldspace_and_observant_coordinate,
)
from video_distillation.guidance import ModelscopeGuidance
from video_distillation.prompt_processors import ModelscopePromptProcessor


wp.init()
wp.config.verify_cuda = True
ti.init(arch=ti.cuda, device_memory_GB=8.0)


class PipelineParamsNoparse:
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False


def load_checkpoint(model_path, iteration=-1):
    checkpt_dir = os.path.join(model_path, "point_cloud")
    if iteration == -1:
        iteration = searchForMaxIteration(checkpt_dir)
    checkpt_path = os.path.join(checkpt_dir, f"iteration_{iteration}", "point_cloud.ply")

    from plyfile import PlyData

    plydata = PlyData.read(checkpt_path)
    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))

    sh_degree = int(math.sqrt((len(extra_f_names) + 3) // 3)) - 1
    gaussians = SceneGaussianModel(sh_degree)
    gaussians.load_ply(checkpt_path)
    return gaussians


def extract_cluster_features(shs: torch.Tensor) -> torch.Tensor:
    if shs.ndim == 3 and shs.shape[1] >= 1 and shs.shape[2] == 3:
        colors = shs[:, 0, :]
    else:
        flat = shs.reshape(shs.shape[0], -1)
        colors = flat[:, :3]

    colors = colors.float()
    colors = colors - colors.min(dim=0, keepdim=True).values
    colors = colors / (colors.max(dim=0, keepdim=True).values - colors.min(dim=0, keepdim=True).values + 1e-6)
    return colors.clamp(0.0, 1.0)


def run_kmeans(features: torch.Tensor, num_clusters: int, max_iter: int = 20, seed: int = 0):
    features = features.detach()
    num_clusters = max(1, min(num_clusters, features.shape[0]))
    generator = torch.Generator(device=features.device)
    generator.manual_seed(seed)
    init_idx = torch.randperm(features.shape[0], generator=generator, device=features.device)[:num_clusters]
    centers = features[init_idx].clone()

    labels = torch.zeros(features.shape[0], dtype=torch.long, device=features.device)
    for _ in range(max_iter):
        dist = torch.cdist(features, centers)
        new_labels = torch.argmin(dist, dim=1)

        new_centers = []
        for cluster_id in range(num_clusters):
            mask = new_labels == cluster_id
            if mask.any():
                new_centers.append(features[mask].mean(dim=0))
            else:
                new_centers.append(centers[cluster_id])
        new_centers = torch.stack(new_centers, dim=0)

        if torch.equal(new_labels, labels):
            centers = new_centers
            break

        labels = new_labels
        centers = new_centers

    return labels, centers


def assign_clusters_to_filled_particles(reference_pos: torch.Tensor, reference_labels: torch.Tensor, query_pos: torch.Tensor):
    if query_pos.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=reference_labels.device)

    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(reference_pos.detach().cpu().numpy())
        _, nearest = tree.query(query_pos.detach().cpu().numpy(), k=1)
        nearest = torch.as_tensor(nearest, dtype=torch.long, device=reference_labels.device)
        return reference_labels[nearest]
    except Exception:
        batch_size = 4096
        out = []
        reference_pos = reference_pos.to(query_pos.device)
        for start in range(0, query_pos.shape[0], batch_size):
            end = min(start + batch_size, query_pos.shape[0])
            dist = torch.cdist(query_pos[start:end], reference_pos)
            nearest = torch.argmin(dist, dim=1)
            out.append(reference_labels[nearest].to(reference_labels.device))
        return torch.cat(out, dim=0)


def aggregate_cluster_grad(grad: torch.Tensor, cluster_ids: torch.Tensor, num_clusters: int):
    if grad is None:
        return torch.zeros((num_clusters,), device=cluster_ids.device)

    grad = grad.detach().reshape(-1)
    cluster_ids = cluster_ids.long()
    cluster_grad = torch.zeros((num_clusters,), device=grad.device, dtype=grad.dtype)
    cluster_count = torch.zeros((num_clusters,), device=grad.device, dtype=grad.dtype)
    cluster_grad.scatter_add_(0, cluster_ids, grad)
    cluster_count.scatter_add_(0, cluster_ids, torch.ones_like(grad))
    return cluster_grad / cluster_count.clamp_min(1.0)


def log10_update(param: torch.Tensor, grad: torch.Tensor, lr: float, upper: float):
    if grad is None:
        return param
    log_param = torch.log10(param.clamp_min(1e-12))
    log_param = torch.clamp(log_param - grad * lr, -1.0, upper)
    return torch.pow(torch.tensor(10.0, device=param.device, dtype=param.dtype), log_param)


def build_cluster_params(material_params: dict, num_clusters: int, device: str):
    return {
        "E": torch.full((num_clusters,), float(material_params["E"]), device=device),
        "mu_N": torch.full((num_clusters,), float(material_params["mu_N"]), device=device),
        "lam_N": torch.full((num_clusters,), float(material_params["lam_N"]), device=device),
        "viscosity": torch.full((num_clusters,), float(material_params["viscosity"]), device=device),
    }


def get_active_parameter_names(material_params: dict):
    active = material_params.get("activate_para", ["E", "mu_N", "lam_N", "viscosity"])
    return [name for name in ["E", "mu_N", "lam_N", "viscosity"] if name in active]


def broadcast_cluster_params(mpm_solver, cluster_params: dict, cluster_ids: torch.Tensor, device: str):
    particle_params = {
        "E": cluster_params["E"][cluster_ids],
        "mu_N": cluster_params["mu_N"][cluster_ids],
        "lam_N": cluster_params["lam_N"][cluster_ids],
        "viscosity": cluster_params["viscosity"][cluster_ids],
    }
    mpm_solver.set_parameters_dict(particle_params, device=device)


def cluster_summary(cluster_params: dict, material_params: dict):
    nu = float(material_params["nu"])
    for idx in range(cluster_params["E"].shape[0]):
        e_val = float(cluster_params["E"][idx].item())
        mu_val = 1e7 * e_val / (2.0 * (1.0 + nu))
        lam_val = 1e7 * e_val * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        print(
            f"cluster {idx}: E={e_val:.6f}, mu={mu_val:.6f}, lambda={lam_val:.6f}, viscosity={float(cluster_params['viscosity'][idx].item()):.6f}, mu_N={float(cluster_params['mu_N'][idx].item()):.6f}, lam_N={float(cluster_params['lam_N'][idx].item()):.6f}"
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
    parser.add_argument("--num_clusters", type=int, default=4)
    parser.add_argument("--cluster_seed", type=int, default=0)
    parser.add_argument("--cluster_iter", type=int, default=20)
    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        raise AssertionError("Model path does not exist!")
    if not os.path.exists(args.physics_config):
        raise AssertionError("Scene config does not exist!")
    if not os.path.exists(args.guidance_config):
        raise AssertionError("Scene config does not exist!")
    if args.output_path is not None and not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    print("Loading scene config...")
    material_params, bc_params, time_params, preprocessing_params, camera_params = decode_param_json(args.physics_config)

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

    print("Initializing scene and preprocessing...")
    params = load_params_from_gs(gaussians, pipeline)
    init_pos = params["pos"]
    init_cov = params["cov3D_precomp"]
    init_screen_points = params["screen_points"]
    init_opacity = params["opacity"]
    init_shs = params["shs"]

    mask = init_opacity[:, 0] > preprocessing_params["opacity_threshold"]
    init_pos = init_pos[mask, :]
    init_cov = init_cov[mask, :]
    init_opacity = init_opacity[mask, :]
    init_screen_points = init_screen_points[mask, :]
    init_shs = init_shs[mask, :]

    unselected_pos, unselected_cov, unselected_opacity, unselected_shs = (None, None, None, None)
    moving_pts_path = os.path.join(model_path, "moving_part_points.ply")
    if os.path.exists(moving_pts_path):
        import point_cloud_utils as pcu

        moving_pts = pcu.load_mesh_v(moving_pts_path)
        moving_pts = torch.from_numpy(moving_pts).float().to("cuda")
        freeze_mask = find_far_points(init_pos, moving_pts, thres=0.05).bool()
        unselected_pos = init_pos[freeze_mask, :]
        unselected_cov = init_cov[freeze_mask, :]
        unselected_opacity = init_opacity[freeze_mask, :]
        unselected_shs = init_shs[freeze_mask, :]

        init_pos = init_pos[~freeze_mask, :]
        init_cov = init_cov[~freeze_mask, :]
        init_opacity = init_opacity[~freeze_mask, :]
        init_shs = init_shs[~freeze_mask, :]

    if args.debug:
        os.makedirs("./log", exist_ok=True)
        particle_position_tensor_to_ply(init_pos, "./log/init_particles.ply")

    rotation_matrices = generate_rotation_matrices(
        torch.tensor(preprocessing_params["rotation_degree"]),
        preprocessing_params["rotation_axis"],
    )
    rotated_pos = apply_rotations(init_pos, rotation_matrices)

    if args.debug:
        particle_position_tensor_to_ply(rotated_pos, "./log/rotated_particles.ply")

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

    init_cov = apply_cov_rotations(init_cov, rotation_matrices)
    init_cov = scale_origin * scale_origin * init_cov

    if args.debug:
        particle_position_tensor_to_ply(transformed_pos, "./log/transformed_particles.ply")

    color_features = extract_cluster_features(init_shs)
    cluster_ids, cluster_centers = run_kmeans(color_features, args.num_clusters, max_iter=args.cluster_iter, seed=args.cluster_seed)

    if args.output_path is not None:
        np.save(os.path.join(args.output_path, "cluster_ids_original.npy"), cluster_ids.detach().cpu().numpy())
        np.save(os.path.join(args.output_path, "cluster_centers.npy"), cluster_centers.detach().cpu().numpy())

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

    if mpm_init_pos.shape[0] > gs_num:
        filled_cluster_ids = assign_clusters_to_filled_particles(transformed_pos, cluster_ids, mpm_init_pos[gs_num:])
        mpm_cluster_ids = torch.cat([cluster_ids, filled_cluster_ids], dim=0)
    else:
        mpm_cluster_ids = cluster_ids
    mpm_cluster_ids = mpm_cluster_ids.to(device=device, dtype=torch.long)

    init_len = mpm_init_pos.shape[0]

    print("Initializing MPM solver and boundary conditions...")
    mpm_init_vol = get_particle_volume(
        mpm_init_pos,
        material_params["n_grid"],
        material_params["grid_lim"] / material_params["n_grid"],
        unifrom=material_params["material"] == "sand",
    ).to(device=device)

    if filling_params is not None and filling_params["visualize"]:
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
        gaussians._xyz = nn.Parameter(torch.tensor(torch.cat([gaussians._xyz, _pos], 0), dtype=torch.float, device="cuda").requires_grad_(True))
        _features_dc = torch.zeros((_pos.shape[0], 1, 3)).to("cuda:0")
        gaussians._features_dc = nn.Parameter(torch.tensor(torch.cat([gaussians._features_dc, _features_dc], 0), dtype=torch.float, device="cuda").requires_grad_(True))
        _features_rest = torch.zeros((_pos.shape[0], 15, 3)).to("cuda:0")
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

    mpm_solver = MPM_Simulator_WARP(10)
    mpm_solver.load_initial_data_from_torch(
        mpm_init_pos,
        mpm_init_vol,
        mpm_init_cov,
        n_grid=material_params["n_grid"],
        grid_lim=material_params["grid_lim"],
        device=device,
    )
    mpm_solver.set_parameters_dict(material_params, device=device)
    set_boundary_conditions(mpm_solver, bc_params, time_params, device=device)

    cluster_params = build_cluster_params(material_params, args.num_clusters, device)
    active_parameter_names = get_active_parameter_names(material_params)
    broadcast_cluster_params(mpm_solver, cluster_params, mpm_cluster_ids, device)

    tape = wp.Tape()

    mpm_space_viewpoint_center = torch.tensor(camera_params["mpm_space_viewpoint_center"]).reshape((1, 3)).cuda()
    mpm_space_vertical_upward_axis = torch.tensor(camera_params["mpm_space_vertical_upward_axis"]).reshape((1, 3)).cuda()
    viewpoint_center_worldspace, observant_coordinates = get_center_view_worldspace_and_observant_coordinate(
        mpm_space_viewpoint_center,
        mpm_space_vertical_upward_axis,
        rotation_matrices,
        scale_origin,
        original_mean_pos,
    )

    if args.output_ply or args.output_h5:
        directory_to_save = os.path.join(args.output_path, "simulation_ply")
        os.makedirs(directory_to_save, exist_ok=True)
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

    yaml_confs = OmegaConf.load(args.guidance_config)
    yaml_confs.prompt_processor.prompt = args.prompt
    guidance = ModelscopeGuidance(yaml_confs.guidance)
    prompt_processor = ModelscopePromptProcessor(yaml_confs.prompt_processor)
    prompt_utils = prompt_processor()

    stage_num = 8
    frame_per_stage = 16

    cluster_summary(cluster_params, material_params)
    print("active parameters:", active_parameter_names)

    def maybe_aggregate_and_update(param_name: str, upper: float):
        grad_tensor = getattr(mpm_solver.mpm_model, param_name).grad
        cluster_grad = aggregate_cluster_grad(wp.to_torch(grad_tensor), mpm_cluster_ids, args.num_clusters)
        if param_name in active_parameter_names:
            cluster_params[param_name] = log10_update(cluster_params[param_name], cluster_grad, lr=1.0, upper=upper)
        return cluster_grad

    for batch in range(50):
        loss_value = 0.0
        img_list = []
        tape.reset()
        with tape:
            broadcast_cluster_params(mpm_solver, cluster_params, mpm_cluster_ids, device)
            mpm_solver.finalize_mu_lam(device=device)

        for _ in range(step_per_frame * (batch % stage_num)):
            mpm_solver.p2g2p(None, substep_dt, device=device)

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
            rasterize = initialize_resterize(current_camera, gaussians, pipeline, background)

            for _ in range(step_per_frame * (1 + stage_num) - 1):
                mpm_solver.p2g2p(frame, substep_dt, device=device)

            with tape:
                mpm_solver.p2g2p(frame, substep_dt, device=device)

                pos = mpm_solver.export_particle_x_to_torch()[:gs_num].to(device)
                cov3D = mpm_solver.export_particle_cov_to_torch(device=device)
                rot = mpm_solver.export_particle_R_to_torch()

            cov3D = cov3D.view(-1, 6)[:gs_num].to(device)
            rot = rot.view(-1, 3, 3)[:gs_num].to(device)

            pos = pos[:init_len, :]
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
            rendering, _ = rasterize(
                means3D=pos,
                means2D=pos,
                shs=None,
                colors_precomp=colors_precomp,
                opacities=opacity,
                scales=None,
                rotations=None,
                cov3D_precomp=cov3D,
            )
            img_list.append(rendering)

        loss = 0.0
        img_list = torch.stack(img_list)
        guidance_out = guidance(
            img_list,
            prompt_utils,
            torch.Tensor([camera_params["init_elevation"]]),
            torch.Tensor([camera_params["init_azimuthm"]]),
            torch.Tensor([camera_params["init_radius"]]),
            rgb_as_latents=False,
            num_frames=frame_per_stage,
            train_dynamic_camera=False,
        )
        print(guidance_out)
        for name, value in guidance_out.items():
            if name.startswith("loss_"):
                loss += value * 3e-4
        loss = loss / stage_num
        print(loss)
        loss.backward(retain_graph=True)
        loss_value += loss.item()

        grad_x = mpm_solver.mpm_state.particle_x.grad
        grad_cov = mpm_solver.mpm_state.particle_cov.grad
        grad_r = mpm_solver.mpm_state.particle_R.grad
        loss_wp = wp.zeros(1, dtype=float, device=device, requires_grad=True)
        wp.launch(sum_vec3, mpm_solver.n_particles, [mpm_solver.mpm_state.particle_x, grad_x], [loss_wp], device=device)
        wp.launch(sum_array, mpm_solver.n_particles * 6, [mpm_solver.mpm_state.particle_cov, grad_cov], [loss_wp], device=device)
        wp.launch(sum_mat33, mpm_solver.n_particles, [mpm_solver.mpm_state.particle_R, grad_r], [loss_wp], device=device)
        tape.backward(loss=loss_wp)

        with torch.no_grad():
            cluster_E_grad = maybe_aggregate_and_update("E", upper=-0.4)
            cluster_muN_grad = maybe_aggregate_and_update("mu_N", upper=1.0)
            cluster_lamN_grad = maybe_aggregate_and_update("lam_N", upper=1.0)
            cluster_viscosity_grad = maybe_aggregate_and_update("viscosity", upper=2.0)

        print("cluster grad E:", cluster_E_grad)
        print("cluster grad mu_N:", cluster_muN_grad)
        print("cluster grad lam_N:", cluster_lamN_grad)
        print("cluster grad viscosity:", cluster_viscosity_grad)
        cluster_summary(cluster_params, material_params)

        broadcast_cluster_params(mpm_solver, cluster_params, mpm_cluster_ids, device)
        mpm_solver.reset_pos_from_torch(mpm_init_pos, mpm_init_vol, mpm_init_cov, device=device)

        if batch % 2 == 0:
            broadcast_cluster_params(mpm_solver, cluster_params, mpm_cluster_ids, device)
            mpm_solver.finalize_mu_lam(device=device)
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
                rasterize = initialize_resterize(current_camera, gaussians, pipeline, background)

                for _ in range(step_per_frame):
                    mpm_solver.p2g2p(frame, substep_dt, device=device)

                pos = mpm_solver.export_particle_x_to_torch()[:gs_num].to(device)
                cov3D = mpm_solver.export_particle_cov_to_torch(device=device)
                rot = mpm_solver.export_particle_R_to_torch(device=device)

                cov3D = cov3D.view(-1, 6)[:gs_num].to(device)
                rot = rot.view(-1, 3, 3)[:gs_num].to(device)

                pos = pos[:init_len, :]
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
                rendering, _ = rasterize(
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
                cv2.imwrite(os.path.join(args.output_path, f"{frame}.png".rjust(8, "0")), 255 * cv2_img)

            save_video(args.output_path, os.path.join(args.output_path, f"video{batch:02d}.mp4"))
