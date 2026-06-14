#!/usr/bin/env python
"""
Multi-material segmentation and per-point material assignment for 3DGS.

Workflow:
  1. SAM (Segment Anything) auto-generates masks from training images
  2. Masks are clustered by color/position → material regions
  3. 2D masks are projected to 3D gaussian points via camera matrices
  4. Per-point material IDs are saved alongside the PLY

Usage:
  # Standalone (after 3DGS training):
  python segment_material.py --dataset chair

  # Via pipeline:
  python pipeline.py --dataset chair --multi-material
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import cv2
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))


# ============================================================
# Material type definitions
# ============================================================
MATERIAL_TYPES = {
    0: {"name": "wood",       "mpm": "plasticine", "E": 1e10, "nu": 0.35, "density": 700,  "yield_stress": 5e6},
    1: {"name": "metal",      "mpm": "metal",      "E": 2e11, "nu": 0.30, "density": 7800, "yield_stress": 2.5e8},
    2: {"name": "plastic",    "mpm": "plasticine", "E": 2e9,  "nu": 0.38, "density": 1200, "yield_stress": 2e7},
    3: {"name": "fabric",     "mpm": "jelly",      "E": 5e5,  "nu": 0.45, "density": 400,  "yield_stress": 1e4},
    4: {"name": "foam",       "mpm": "foam",       "E": 1e6,  "nu": 0.20, "density": 200,  "yield_stress": 5e4},
    5: {"name": "leather",    "mpm": "plasticine", "E": 5e8,  "nu": 0.40, "density": 900,  "yield_stress": 1e7},
    6: {"name": "rubber",     "mpm": "jelly",      "E": 1e7,  "nu": 0.49, "density": 1100, "yield_stress": 5e5},
}

NUM_MATERIALS = 4  # Default number of material clusters to produce


# ============================================================
# SAM Segmentation
# ============================================================
class SAMSegmenter:
    """Wrapper around Segment Anything Model for automatic mask generation."""

    def __init__(self, checkpoint_path, device="cuda"):
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

        print(f"  Loading SAM from {checkpoint_path}...")
        self.sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
        self.sam.to(device=device)
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=32,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.92,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=200,
        )
        self.device = device
        print(f"  SAM loaded successfully.")

    def generate_masks(self, image_bgr):
        """Generate automatic masks for an image.

        Args:
            image_bgr: numpy array [H, W, 3] in BGR format (OpenCV)
        Returns:
            list of mask dicts, each with:
                segmentation: [H, W] bool array
                bbox: [x, y, w, h]
                area: int
                predicted_iou: float
                stability_score: float
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        masks = self.mask_generator.generate(image_rgb)
        # Sort by area descending (largest masks first)
        masks.sort(key=lambda m: m["area"], reverse=True)
        return masks


# ============================================================
# Mask Feature Extraction & Clustering
# ============================================================
class MaskClassifier:
    """Classify SAM masks into material categories via color/position clustering."""

    def __init__(self, num_materials=NUM_MATERIALS):
        self.num_materials = num_materials

    def extract_features(self, image_bgr, mask):
        """Extract feature vector for a single mask.

        Returns:
            np.array of shape (6,): [avg_r, avg_g, avg_b, center_y_norm, aspect_ratio, area_norm]
        """
        h, w = image_bgr.shape[:2]
        seg = mask["segmentation"]
        bbox = mask["bbox"]  # [x, y, w, h]

        # Average color in masked region
        masked_pixels = image_bgr[seg]
        if len(masked_pixels) == 0:
            avg_color = np.array([128, 128, 128], dtype=np.float32)
        else:
            avg_color = masked_pixels.mean(axis=0).astype(np.float32)  # BGR

        # Normalized vertical center
        center_y = (bbox[1] + bbox[3] / 2) / h

        # Bounding box aspect ratio
        aspect = bbox[3] / (bbox[2] + 1e-8)  # height/width

        # Normalized area
        area_norm = mask["area"] / (h * w)

        return np.array([
            avg_color[2],  # R (from BGR)
            avg_color[1],  # G
            avg_color[0],  # B
            center_y,
            aspect,
            area_norm,
        ], dtype=np.float32)

    def cluster_masks(self, all_features):
        """Cluster mask features into material groups.

        Args:
            all_features: list of np.array [6] features
        Returns:
            cluster_labels: list of int (0..K-1)
            cluster_centers: np.array [K, 6]
        """
        from sklearn.cluster import KMeans

        if len(all_features) < self.num_materials:
            k = max(1, len(all_features))
        else:
            k = self.num_materials

        feats = np.stack(all_features, axis=0)
        # Normalize features
        feats_norm = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-8)

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(feats_norm)
        return labels, kmeans.cluster_centers_

    def assign_materials_to_clusters(self, cluster_centers, cluster_labels, all_features,
                                        material_hint=None):
        """Map each cluster to a material type based on average color.

        Uses HSV-like color reasoning for more robust classification.

        Args:
            material_hint: optional list of material names to restrict choices,
                          e.g. ['wood', 'metal', 'fabric']. Clusters are assigned
                          to the best-matching hint material.
        """
        k = len(cluster_centers)
        material_map = {}

        for c in range(k):
            # Get average color of this cluster (in original feature space)
            mask = cluster_labels == c
            cluster_feats = np.array([all_features[i] for i in range(len(all_features)) if mask[i]])
            if len(cluster_feats) == 0:
                material_map[c] = 2  # default plastic
                continue

            avg_r = float(cluster_feats[:, 0].mean())
            avg_g = float(cluster_feats[:, 1].mean())
            avg_b = float(cluster_feats[:, 2].mean())

            # HSV-like features
            max_c = max(avg_r, avg_g, avg_b)
            min_c = min(avg_r, avg_g, avg_b)
            brightness = (avg_r + avg_g + avg_b) / 3.0
            saturation = (max_c - min_c) / (max_c + 1e-8)
            rg_ratio = avg_r / (avg_g + 1e-8)
            rb_ratio = avg_r / (avg_b + 1e-8)

            if brightness < 50:
                # Very dark: wood or leather
                if rg_ratio > 1.3:
                    material_map[c] = 0  # wood (dark brown)
                else:
                    material_map[c] = 5  # leather (dark)
            elif saturation < 0.12:
                # Low saturation: gray/metallic or white
                if brightness > 170:
                    material_map[c] = 1  # metal (bright gray/silver)
                elif brightness < 80:
                    material_map[c] = 0  # wood (dark, low saturation)
                else:
                    material_map[c] = 2  # plastic (mid-gray)
            elif rg_ratio > 1.15 and rb_ratio > 1.2:
                # Warm / reddish / brown tones
                if brightness < 130:
                    material_map[c] = 0  # wood (brown)
                else:
                    material_map[c] = 5  # leather (lighter reddish)
            elif avg_g > avg_r * 1.1:
                # Greenish
                material_map[c] = 4  # foam
            elif brightness > 160:
                material_map[c] = 2  # plastic (bright)
            elif saturation < 0.25:
                material_map[c] = 3  # fabric (low saturation, medium brightness)
            else:
                material_map[c] = 2  # plastic default

        # If material_hint provided, remap clusters to hint materials
        if material_hint:
            # Build reverse mapping: material_name → material_id
            name_to_id = {v["name"]: k for k, v in MATERIAL_TYPES.items()}
            hint_ids = []
            for hint_name in material_hint:
                if hint_name in name_to_id:
                    hint_ids.append(name_to_id[hint_name])
            hint_ids = hint_ids or list(range(len(MATERIAL_TYPES)))

            # For each cluster, find best-matching hint material by color similarity
            new_material_map = {}
            for c in range(k):
                mask = cluster_labels == c
                cluster_feats = np.array([all_features[i] for i in range(len(all_features)) if mask[i]])
                if len(cluster_feats) == 0:
                    new_material_map[c] = hint_ids[0] if hint_ids else 2
                    continue

                avg_r = float(cluster_feats[:, 0].mean())
                avg_g = float(cluster_feats[:, 1].mean())
                avg_b = float(cluster_feats[:, 2].mean())

                # Score each hint material by color similarity
                best_score = float('-inf')
                best_mat = hint_ids[0] if hint_ids else 2
                for mid in hint_ids:
                    mat = MATERIAL_TYPES[mid]
                    # Use material name to infer expected color
                    expected = _expected_color(mat["name"])
                    # Simple Euclidean distance in RGB (negated → higher is better)
                    score = -((avg_r - expected[0])**2 + (avg_g - expected[1])**2 + (avg_b - expected[2])**2)
                    if score > best_score:
                        best_score = score
                        best_mat = mid
                new_material_map[c] = best_mat

            material_map = new_material_map

        return material_map


def _expected_color(material_name):
    """Return approximate expected RGB color for a material."""
    colors = {
        "wood": (140, 90, 50),      # Brown
        "metal": (180, 180, 185),   # Gray/silver
        "plastic": (160, 160, 160), # Mid-gray
        "fabric": (120, 100, 90),   # Muted brown/gray
        "foam": (200, 190, 140),    # Yellowish
        "leather": (100, 60, 40),   # Dark brown
        "rubber": (50, 50, 55),     # Dark gray/black
    }
    return colors.get(material_name, (150, 150, 150))


# ============================================================
# 3D Projection
# ============================================================
def project_gaussians_to_2d(xyz, camera):
    """Project 3D gaussian centers to 2D image coordinates.

    Args:
        xyz: [N, 3] tensor of gaussian positions (world coords)
        camera: Camera object with world_view_transform [4,4] and projection_matrix [4,4]
    Returns:
        [N, 2] tensor of (x, y) pixel coordinates
    """
    device = xyz.device
    N = xyz.shape[0]

    # Homogeneous world coords: [N, 4]
    ones = torch.ones(N, 1, device=device)
    p_world = torch.cat([xyz, ones], dim=-1)  # [N, 4]

    # world_view_transform is column-major [4,4]
    # p_view = p_world @ W (row vector @ column-major matrix = p * W)
    W = camera.world_view_transform  # [4,4] column-major
    p_view = p_world @ W  # [N, 4]

    # projection_matrix is column-major [4,4]
    P = camera.projection_matrix  # [4,4] column-major
    p_clip = p_view @ P  # [N, 4]

    # NDC
    w_coord = p_clip[:, 3:4]
    w_coord = torch.where(torch.abs(w_coord) < 1e-8, torch.ones_like(w_coord) * 1e-8, w_coord)
    ndc_x = p_clip[:, 0] / w_coord.squeeze()
    ndc_y = p_clip[:, 1] / w_coord.squeeze()

    # Screen coords
    screen_x = (ndc_x + 1.0) * 0.5 * camera.image_width
    screen_y = (1.0 - ndc_y) * 0.5 * camera.image_height  # flip Y for image coords

    return torch.stack([screen_x, screen_y], dim=-1)


def assign_materials_from_views(gaussians, train_cameras, segmenter, classifier,
                                 num_views=6, num_materials=NUM_MATERIALS,
                                 material_hint=None):
    """Main function: segment images, cluster masks, and assign per-point materials.

    Args:
        gaussians: GaussianModel (trained)
        train_cameras: list of Camera objects
        segmenter: SAMSegmenter
        classifier: MaskClassifier
        num_views: number of camera views to sample
        num_materials: number of material clusters

    Returns:
        point_material_ids: np.array [N] of material IDs (0..K-1)
        material_configs: dict mapping material_id → material params
        mask_visualizations: list of (image_path, mask_overlay) for debugging
    """
    N = gaussians._xyz.shape[0]
    device = gaussians._xyz.device

    # Sample diverse views
    indices = np.linspace(0, len(train_cameras) - 1, num_views, dtype=int)
    sampled_cameras = [train_cameras[i] for i in indices]

    print(f"\n  Segmenting {num_views} views with SAM...")

    # Collect all mask features across views
    all_masks_per_view = []  # list of (camera, [(mask_dict, material_label), ...])

    for i, cam in enumerate(sampled_cameras):
        # Get image from camera tensor data (CHW float32 [0,1] on GPU → HWC BGR uint8)
        img_tensor = cam.original_image  # [C, H, W] on GPU
        img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        # RGB → BGR for OpenCV
        image_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # Log image name for display
        img_label = cam.image_name if hasattr(cam, 'image_name') else f"view_{i}"
        print(f"    View {i+1}/{num_views}: {img_label}")

        # Generate masks
        masks = segmenter.generate_masks(image_bgr)

        # Filter: remove background mask (largest area, usually white)
        # Keep masks that are: not too large (>70% of image), not too small
        h_cam, w_cam = image_bgr.shape[:2]
        filtered_masks = []
        for m in masks:
            area_ratio = m["area"] / (h_cam * w_cam)
            if area_ratio < 0.70 and area_ratio > 0.002:
                filtered_masks.append(m)

        print(f"      Generated {len(filtered_masks)} valid masks (filtered from {len(masks)})")
        all_masks_per_view.append((cam, image_bgr, filtered_masks))

    if not all_masks_per_view:
        print("  ERROR: No valid masks generated from any view!")
        return None, None, []

    # ---- Cluster masks across all views ----
    print(f"\n  Clustering masks into {num_materials} material groups...")

    all_features = []
    mask_metadata = []  # (view_idx, mask_idx_in_view)

    for view_idx, (cam, image_bgr, masks) in enumerate(all_masks_per_view):
        for mask_idx, mask in enumerate(masks):
            feat = classifier.extract_features(image_bgr, mask)
            all_features.append(feat)
            mask_metadata.append((view_idx, mask_idx))

    if len(all_features) < 2:
        print("  WARNING: Too few masks for clustering. Using single material.")
        cluster_labels = np.zeros(len(all_features), dtype=int)
    else:
        cluster_labels, centers = classifier.cluster_masks(all_features)

    # Map clusters to materials
    material_map = classifier.assign_materials_to_clusters(
        centers if len(all_features) >= 2 else np.zeros((1, 6)),
        cluster_labels,
        all_features,
        material_hint=material_hint,
    )

    mat_names = {c: MATERIAL_TYPES[mat]["name"] for c, mat in material_map.items()}
    print(f"  Material mapping: {mat_names}")

    # Assign material labels to each mask
    for feat_idx, (view_idx, mask_idx) in enumerate(mask_metadata):
        cluster = cluster_labels[feat_idx]
        material_id = material_map[cluster]
        all_masks_per_view[view_idx][2][mask_idx]["material_id"] = material_id

    # ---- Project masks to 3D gaussians ----
    print(f"\n  Projecting masks to {N} gaussian points...")

    # Per-point material vote accumulator (size by all material types, not just clusters)
    num_mat_types = len(MATERIAL_TYPES)
    material_votes = torch.zeros(N, num_mat_types, device=device)
    total_views_per_point = torch.zeros(N, device=device)

    for view_idx, (cam, image_bgr, masks) in enumerate(all_masks_per_view):
        # Project all gaussians to this view
        screen_coords = project_gaussians_to_2d(gaussians._xyz, cam)  # [N, 2]

        # Check which gaussians are in view frustum
        in_view_x = (screen_coords[:, 0] >= 0) & (screen_coords[:, 0] < cam.image_width)
        in_view_y = (screen_coords[:, 1] >= 0) & (screen_coords[:, 1] < cam.image_height)
        in_view = in_view_x & in_view_y

        if in_view.sum() == 0:
            continue

        # For gaussians in view, check which mask they fall into
        for mask in masks:
            mat_id = mask["material_id"]
            seg = mask["segmentation"]  # [H, W] bool

            # Get pixel coords for in-view gaussians
            px = screen_coords[in_view, 0].long()
            py = screen_coords[in_view, 1].long()

            # Clamp to image bounds
            px = px.clamp(0, cam.image_width - 1)
            py = py.clamp(0, cam.image_height - 1)

            # Check which gaussians fall within this mask
            in_mask = seg[py.cpu().numpy(), px.cpu().numpy()]
            in_mask_tensor = torch.from_numpy(in_mask).to(device)

            # Vote
            indices = torch.where(in_view)[0][in_mask_tensor]
            material_votes[indices, mat_id] += 1
            total_views_per_point[indices] += 1

    # ---- Finalize per-point materials ----
    # Points that never got a vote: assign most common material
    never_labeled = total_views_per_point < 1

    if never_labeled.sum() > 0:
        # Find most common material across labeled points
        if (total_views_per_point >= 1).sum() > 0:
            labeled_votes = material_votes[~never_labeled].sum(dim=0)
            most_common = labeled_votes.argmax().item()
        else:
            most_common = 2  # default plastic
        material_votes[never_labeled, most_common] = 1

    # Majority vote
    point_material_ids = material_votes.argmax(dim=1).cpu().numpy()

    # Count distribution
    unique, counts = np.unique(point_material_ids, return_counts=True)
    print(f"\n  Per-point material distribution:")
    for mat_id, count in zip(unique, counts):
        mat_name = MATERIAL_TYPES[mat_id]["name"]
        pct = 100 * count / N
        print(f"    {mat_name}: {count} points ({pct:.1f}%)")

    # ---- Build material configs ----
    material_configs = {}
    for mat_id in unique:
        material_configs[int(mat_id)] = MATERIAL_TYPES[int(mat_id)]

    return point_material_ids, material_configs, all_masks_per_view


# ============================================================
# Save / Load utilities
# ============================================================
def save_per_point_materials(output_dir, point_material_ids, material_configs):
    """Save material assignments alongside the PLY file.

    Args:
        output_dir: path to e.g. output/chair/point_cloud/iteration_7000/
        point_material_ids: np.array [N] of int material IDs
        material_configs: dict material_id → material params
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save material IDs as numpy array
    ids_path = os.path.join(output_dir, "material_ids.npy")
    np.save(ids_path, point_material_ids)
    print(f"  Saved: {ids_path} ({len(point_material_ids)} points)")

    # Save material mapping config
    config_path = os.path.join(output_dir, "material_map.json")
    config_out = {}
    for mat_id, params in material_configs.items():
        config_out[str(mat_id)] = params
    with open(config_path, "w") as f:
        json.dump(config_out, f, indent=2)
    print(f"  Saved: {config_path}")

    # Also save as PLY attribute (append material_id to existing PLY)
    ply_path = os.path.join(output_dir, "point_cloud.ply")
    if os.path.exists(ply_path):
        try:
            append_material_to_ply(ply_path, point_material_ids)
        except Exception as e:
            print(f"  WARNING: Could not append material to PLY: {e}")


def append_material_to_ply(ply_path, material_ids):
    """Add material_id attribute to existing PLY file."""
    from plyfile import PlyData, PlyElement
    import numpy as np

    ply = PlyData.read(ply_path)
    verts = ply['vertex']

    # Reconstruct with extra attribute
    existing_dtype = verts.data.dtype
    new_dtype = np.dtype(existing_dtype.descr + [('material_id', 'i4')])

    new_data = np.zeros(verts.count, dtype=new_dtype)
    for name in existing_dtype.names:
        new_data[name] = verts[name]
    new_data['material_id'] = material_ids[:verts.count]

    new_vertex = PlyElement.describe(new_data, 'vertex')
    # Write to a new file with _multi suffix
    out_path = ply_path.replace('.ply', '_multi.ply')
    PlyData([new_vertex]).write(out_path)
    print(f"  Saved multi-material PLY: {out_path}")


def load_per_point_materials(output_dir):
    """Load previously saved material assignments."""
    ids_path = os.path.join(output_dir, "material_ids.npy")
    config_path = os.path.join(output_dir, "material_map.json")

    if not os.path.exists(ids_path):
        return None, None

    point_material_ids = np.load(ids_path)
    material_configs = None
    if os.path.exists(config_path):
        with open(config_path) as f:
            material_configs = json.load(f)

    return point_material_ids, material_configs


# ============================================================
# Multi-material config generator (for simulation)
# ============================================================
def generate_multi_material_config(material_configs, output_path, object_center=None):
    """Generate a PhysGaussian config with per-region material parameters.

    This produces a config.json that the simulation code can use,
    with a 'materials' section mapping region IDs to physics params.
    """
    if object_center is None:
        object_center = [1.0, 1.0, 1.0]

    # Default material (most common one)
    default_mat = list(material_configs.values())[0] if material_configs else MATERIAL_TYPES[2]

    config = {
        "opacity_threshold": 0.005,
        "substep_dt": 5e-5,
        "frame_dt": 2e-2,
        "frame_num": 100,
        "nu": default_mat["nu"],
        "E": default_mat["E"],
        "material": default_mat["mpm"],
        "density": default_mat["density"],
        "g": [0, 0, -9.8],
        "grid_v_damping_scale": 0.999,
        "rpic_damping": 0.01,
        "grid_lim": 2.5,
        "n_grid": 64,
        "scale": 0.8,
        "boundary_conditions": [
            {
                "type": "cuboid",
                "point": [object_center[0], object_center[1], 0.15],
                "size": [1.5, 1.5, 0.1],
                "velocity": [0, 0, 0],
                "start_time": 0,
                "end_time": 1e3,
                "reset": 1,
            },
            {"type": "bounding_box"},
        ],
        "mpm_space_vertical_upward_axis": [0, 0, 1],
        "mpm_space_viewpoint_center": object_center,
        "default_camera_index": -1,
        "show_hint": False,
        "init_azimuthm": -36.7,
        "init_elevation": 8.96,
        "init_radius": 4.11,
        "move_camera": True,
        "delta_a": 0.4,
        "delta_e": 0.0,
        "delta_r": 0.0,
        # Multi-material extension
        "multi_material": True,
        "regions": {},
    }

    for mat_id, params in material_configs.items():
        region = {
            "name": params["name"],
            "mpm_material": params["mpm"],
            "E": params["E"],
            "nu": params["nu"],
            "density": params["density"],
        }
        if "yield_stress" in params:
            region["yield_stress"] = params["yield_stress"]
        config["regions"][str(mat_id)] = region

    if "yield_stress" in default_mat:
        config["yield_stress"] = default_mat["yield_stress"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"  Multi-material config saved to {output_path}")
    return config


# ============================================================
# Visualization (debug)
# ============================================================
def visualize_masks(image_bgr, masks, output_path):
    """Save a visualization of SAM masks overlaid on the image."""
    import colorsys
    import random

    h, w = image_bgr.shape[:2]
    vis = image_bgr.copy()

    # Generate distinct colors
    random.seed(42)
    colors = []
    for _ in range(len(masks)):
        hue = random.random()
        rgb = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
        colors.append((int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255)))  # BGR

    for i, mask in enumerate(masks):
        seg = mask["segmentation"]
        color = colors[i % len(colors)]
        vis[seg] = (vis[seg] * 0.5 + np.array(color) * 0.5).astype(np.uint8)

        # Draw material label if present
        if "material_id" in mask:
            mat_name = MATERIAL_TYPES[mask["material_id"]]["name"]
            bx, by, bw, bh = [int(v) for v in mask["bbox"]]
            cv2.putText(vis, mat_name, (bx, max(by - 5, 10)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imwrite(output_path, vis)
    return output_path


# ============================================================
# Main entry point
# ============================================================
def run_multi_material_pipeline(dataset_path, output_dir, iterations=7000,
                                  sam_checkpoint=None, num_views=6,
                                  num_materials=NUM_MATERIALS, visualize=False,
                                  material_hint=None):
    """Complete multi-material pipeline: train 3DGS → segment → assign → save.

    Args:
        dataset_path: path to NeRF synthetic dataset
        output_dir: path to output/<dataset>/
        iterations: training iterations for 3DGS
        sam_checkpoint: path to SAM ViT-B checkpoint
        num_views: number of views to segment
        num_materials: number of material clusters
        visualize: if True, save mask visualization images
    Returns:
        point_material_ids, material_configs
    """
    device = torch.device("cuda")

    # ---- Check SAM checkpoint ----
    if sam_checkpoint is None:
        sam_checkpoint = os.path.join(os.path.dirname(__file__), "sam_vit_b.pth")
    if not os.path.exists(sam_checkpoint):
        print(f"ERROR: SAM checkpoint not found at {sam_checkpoint}")
        print("Download from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth")
        return None, None

    # ---- Load trained gaussians ----
    # Find PLY
    from pipeline import find_latest_ply
    ply_path = find_latest_ply(output_dir)
    if ply_path is None:
        print("ERROR: No trained PLY found. Run 3DGS training first.")
        return None, None

    print(f"\n{'='*60}")
    print(f"Multi-Material Pipeline")
    print(f"{'='*60}")
    print(f"  PLY: {ply_path}")
    print(f"  SAM: {sam_checkpoint}")
    print(f"  Views: {num_views}")
    print(f"  Materials: {num_materials}")

    # Load trained gaussian model
    from my_3dgs.gaussian import GaussianModel
    from my_3dgs.dataset import read_nerf_synthetic

    gaussians = GaussianModel(sh_degree=3)
    # We need to load the actual trained parameters from the PLY
    # For now, we re-initialize and then the projection only needs xyz
    # (which we read from the PLY)

    from plyfile import PlyData
    ply_data = PlyData.read(ply_path)
    verts = ply_data['vertex']
    xyz = np.stack([verts['x'], verts['y'], verts['z']], axis=-1)

    # Create minimal gaussian model with just xyz
    gaussians._xyz = torch.tensor(xyz, device=device)
    N = xyz.shape[0]
    print(f"  Points: {N}")

    # ---- Load cameras ----
    train_cameras, nerf_norm, _, _ = read_nerf_synthetic(
        dataset_path, white_background=True, eval=False,
    )
    print(f"  Cameras: {len(train_cameras)}")

    # ---- Initialize SAM ----
    segmenter = SAMSegmenter(sam_checkpoint, device="cuda")
    classifier = MaskClassifier(num_materials=num_materials)

    # ---- Segment + classify + project ----
    point_material_ids, material_configs, views_data = assign_materials_from_views(
        gaussians, train_cameras, segmenter, classifier,
        num_views=num_views, num_materials=num_materials,
        material_hint=material_hint,
    )

    if point_material_ids is None:
        return None, None

    # ---- Save ----
    pc_dir = os.path.join(output_dir, "point_cloud", f"iteration_{iterations}")
    save_per_point_materials(pc_dir, point_material_ids, material_configs)

    # Also generate multi-material config
    config_path = os.path.join(output_dir, "config_multi.json")
    generate_multi_material_config(material_configs, config_path)

    # ---- Visualize ----
    if visualize:
        vis_dir = os.path.join(output_dir, "mask_vis")
        os.makedirs(vis_dir, exist_ok=True)
        for view_idx, (cam, image_bgr, masks) in enumerate(views_data):
            if masks:
                vis_path = os.path.join(vis_dir, f"view_{view_idx:02d}.jpg")
                visualize_masks(image_bgr, masks, vis_path)
                print(f"  Visualization saved: {vis_path}")

    print(f"\n{'='*60}")
    print(f"  Multi-Material Pipeline Complete")
    print(f"{'='*60}")

    return point_material_ids, material_configs


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-material 3DGS segmentation")
    parser.add_argument("--dataset", type=str, default="chair")
    parser.add_argument("--iterations", type=int, default=7000)
    parser.add_argument("--num_views", type=int, default=6)
    parser.add_argument("--num_materials", type=int, default=4)
    parser.add_argument("--sam_checkpoint", type=str, default=None)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()

    base_dir = os.path.dirname(__file__)
    dataset_path = os.path.join(base_dir, "..", "nerf_synthetic", args.dataset)
    output_dir = os.path.join(base_dir, "output", args.dataset)

    run_multi_material_pipeline(
        dataset_path, output_dir,
        iterations=args.iterations,
        sam_checkpoint=args.sam_checkpoint,
        num_views=args.num_views,
        num_materials=args.num_materials,
        visualize=args.visualize,
    )
