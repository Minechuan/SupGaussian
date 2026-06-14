"""Camera model with NeRF/COLMAP coordinate transforms."""

import torch
import numpy as np


def fov2focal(fov, pixels):
    """Convert field of view (radians) to focal length in pixels."""
    return pixels / (2 * np.tan(fov / 2))


def focal2fov(focal, pixels):
    """Convert focal length in pixels to field of view (radians)."""
    return 2 * np.arctan(pixels / (2 * focal))


def getWorld2View2(R, t, translate=np.array([0.0, 0.0, 0.0]), scale=1.0):
    """Build world-to-view matrix (OpenGL convention)."""
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    """Build OpenGL-style projection matrix."""
    tanHalfFovY = np.tan(fovY / 2)
    tanHalfFovX = np.tan(fovX / 2)
    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = np.zeros((4, 4))
    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class Camera:
    """Single training camera."""

    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid, trans=np.array([0.0, 0.0, 0.0]), scale=1.0,
                 data_device="cuda"):
        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R  # already transposed for glm
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        # Image
        self.original_image = image.clamp(0.0, 1.0)
        self.image_width = image.shape[2]
        self.image_height = image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask

        self.original_image = self.original_image.to(data_device)

        # Camera matrices
        self.world_view_transform = torch.tensor(
            getWorld2View2(R, T, trans, scale)
        ).transpose(0, 1).float().to(data_device)
        self.projection_matrix = torch.tensor(
            getProjectionMatrix(znear=0.01, zfar=100.0, fovX=FoVx, fovY=FoVy)
        ).transpose(0, 1).float().to(data_device)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(
                self.projection_matrix.unsqueeze(0)
            )
        ).squeeze(0)
        self.camera_center = torch.tensor(
            np.linalg.inv(getWorld2View2(R, T, trans, scale))[:3, 3]
        ).float().to(data_device)


class MiniCam:
    """Minimal camera for densification checks (no image data)."""

    def __init__(self, width, height, fovx, fovy, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVx = fovx
        self.FoVy = fovy
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        self.camera_center = torch.linalg.inv(world_view_transform)[3, :3]
