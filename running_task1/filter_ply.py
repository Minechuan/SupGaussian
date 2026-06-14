import os
import sys
import numpy as np
from plyfile import PlyData, PlyElement


def filter_ply(input_path, output_path, opacity_threshold=0.1):
    ply = PlyData.read(input_path)
    verts = ply['vertex']

    op_logit = np.array(verts['opacity'])
    op_sigmoid = 1.0 / (1.0 + np.exp(-op_logit))

    mask = op_sigmoid >= opacity_threshold

    x = np.array(verts['x'])[mask]
    y = np.array(verts['y'])[mask]
    z = np.array(verts['z'])[mask]
    nx = np.zeros_like(x)
    ny = np.zeros_like(y)
    nz = np.zeros_like(z)
    f_dc_0 = np.array(verts['f_dc_0'])[mask]
    f_dc_1 = np.array(verts['f_dc_1'])[mask]
    f_dc_2 = np.array(verts['f_dc_2'])[mask]
    op = op_logit[mask]
    scale_0 = np.array(verts['scale_0'])[mask]
    scale_1 = np.array(verts['scale_1'])[mask]
    scale_2 = np.array(verts['scale_2'])[mask]
    rot_0 = np.array(verts['rot_0'])[mask]
    rot_1 = np.array(verts['rot_1'])[mask]
    rot_2 = np.array(verts['rot_2'])[mask]
    rot_3 = np.array(verts['rot_3'])[mask]

    f_rest_count = sum(1 for name in verts.data.dtype.names if name.startswith('f_rest_'))
    extra_cols = []
    for i in range(f_rest_count):
        extra_cols.append(np.array(verts[f'f_rest_{i}'])[mask])

    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),
    ]
    for i in range(f_rest_count):
        dtype.append((f'f_rest_{i}', 'f4'))
    dtype += [
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
    ]

    data = np.concatenate([
        x[:, None], y[:, None], z[:, None],
        nx[:, None], ny[:, None], nz[:, None],
        f_dc_0[:, None], f_dc_1[:, None], f_dc_2[:, None],
    ] + [c[:, None] for c in extra_cols] + [
        op[:, None],
        scale_0[:, None], scale_1[:, None], scale_2[:, None],
        rot_0[:, None], rot_1[:, None], rot_2[:, None], rot_3[:, None],
    ], axis=1)

    elements = np.empty(len(x), dtype=dtype)
    elements[:] = list(map(tuple, data))
    vertex_element = PlyElement.describe(elements, 'vertex')
    PlyData([vertex_element]).write(output_path)
    return len(x), len(ply['vertex']['x'])


def find_ply_path(output_dir, iteration=None):
    pc_dir = os.path.join(output_dir, "point_cloud")
    if not os.path.exists(pc_dir):
        return None
    iterations = []
    for d in os.listdir(pc_dir):
        if d.startswith("iteration_"):
            try:
                iterations.append(int(d.split("_")[1]))
            except ValueError:
                pass
    if not iterations:
        return None
    if iteration:
        target = iteration
    else:
        target = max(iterations)
    ply_path = os.path.join(pc_dir, f"iteration_{target}", "point_cloud.ply")
    if os.path.exists(ply_path):
        return ply_path
    return None


def main():
    output_root = os.path.join(os.path.dirname(__file__), "output")
    for obj_id in sorted(os.listdir(output_root)):
        obj_dir = os.path.join(output_root, obj_id)
        if not os.path.isdir(obj_dir):
            continue
        ply_path = find_ply_path(obj_dir, iteration=30000)
        if ply_path is None:
            ply_path = find_ply_path(obj_dir)
        if ply_path is None:
            print(f"No PLY found for {obj_id}")
            continue
        for thresh in [0.1, 0.3, 0.5]:
            out_path = os.path.join(obj_dir, f"point_cloud_filtered_{int(thresh*100)}.ply")
            kept, total = filter_ply(ply_path, out_path, opacity_threshold=thresh)
            print(f"{obj_id} thr={thresh:.1f}: kept {kept}/{total} ({100*kept/total:.1f}%)")


if __name__ == "__main__":
    main()
