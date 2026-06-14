"""NeRF synthetic dataset reader for 3DGS."""

import os
import json
import numpy as np
from PIL import Image
from pathlib import Path
from .camera import Camera, fov2focal, focal2fov


def read_nerf_synthetic(path, white_background=True, eval=False, extension=".png"):
    """
    Read a NeRF synthetic (Blender) dataset.

    Args:
        path: dataset root (contains transforms_train.json)
        white_background: composite RGBA over white
        eval: if True, separate train/test; if False, all frames for training
        extension: image file extension

    Returns:
        train_cameras: list of Camera objects
        nerf_normalization: dict with 'translate' and 'radius'
        ply_path: path to input point cloud (for random init)
        num_pts: number of random points to initialize
    """
    # Read transforms_train.json
    train_json = os.path.join(path, "transforms_train.json")
    test_json = os.path.join(path, "transforms_test.json")

    print(f"Reading Training Transforms from {train_json}")
    train_cam_infos = read_cameras_from_transforms(path, train_json, white_background, extension)

    # Test cameras
    test_cam_infos = []
    if os.path.exists(test_json):
        print(f"Reading Test Transforms from {test_json}")
        test_cam_infos = read_cameras_from_transforms(path, test_json, white_background, extension)

    # If not eval, add test cameras to training
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    # Compute normalization
    nerf_normalization = get_nerf_norm(train_cam_infos)

    # Random point cloud
    ply_path = os.path.join(path, "points3d.ply")
    num_pts = 100_000

    return train_cam_infos, nerf_normalization, ply_path, num_pts


def read_cameras_from_transforms(path, transforms_file, white_background, extension=".png"):
    """Parse transforms.json and create Camera objects."""
    cam_infos = []

    with open(os.path.join(path, transforms_file)) as f:
        contents = json.load(f)
        fovx = contents["camera_angle_x"]
        frames = contents["frames"]

        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # Camera-to-world matrix
            c2w = np.array(frame["transform_matrix"])
            # OpenGL → COLMAP: flip Y and Z
            c2w[:3, 1:3] *= -1

            # World-to-camera
            w2c = np.linalg.inv(c2w)
            # R stored transposed for CUDA glm
            R = np.transpose(w2c[:3, :3])
            T = w2c[:3, 3]

            # Load image with RGBA compositing
            image = Image.open(cam_name)
            im_data = np.array(image.convert("RGBA"))
            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.uint8), "RGB")

            # FOV
            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])

            # GT alpha mask
            gt_alpha = None
            if not white_background:
                gt_alpha = norm_data[:, :, 3:4]
                gt_alpha = np.transpose(gt_alpha, (2, 0, 1))
                import torch
                gt_alpha = torch.tensor(gt_alpha).float()

            # Convert image to tensor
            import torch
            image_tensor = torch.tensor(np.array(image) / 255.0).float()
            image_tensor = image_tensor.permute(2, 0, 1)  # C, H, W

            image_name = Path(cam_name).stem

            cam_info = Camera(
                colmap_id=idx,
                R=R, T=T,
                FoVx=fovx, FoVy=fovy,
                image=image_tensor,
                gt_alpha_mask=gt_alpha,
                image_name=image_name,
                uid=idx,
            )
            cam_infos.append(cam_info)

    return cam_infos


def get_nerf_norm(cam_infos):
    """Compute translation and radius for scene normalization."""
    import numpy as np
    from .camera import getWorld2View2

    cam_centers = []
    for cam in cam_infos:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    cam_centers = np.hstack(cam_centers)
    avg_center = np.mean(cam_centers, axis=1, keepdims=True)
    center = avg_center
    dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
    diagonal = np.max(dist)
    radius = diagonal * 1.1

    translate = -center.flatten()
    return {"translate": translate, "radius": radius}
