import os
import json
from material_database import get_material_params


def generate_config(material_name, output_path, object_center=None):
    params = get_material_params(material_name)

    if object_center is None:
        object_center = [1.0, 1.0, 1.0]

    config = {
        "opacity_threshold": 0.005,
        "rotation_degree": [],
        "rotation_axis": [],
        "substep_dt": 5e-5,
        "frame_dt": 2e-2,
        "frame_num": 100,
        "nu": params["nu"],
        "E": params["E"],
        "material": params["material"],
        "density": params["density"],
        "g": [0, 0, -9.8],
        "grid_v_damping_scale": 0.999,
        "rpic_damping": 0.01,
        "grid_lim": 2.5,
        "n_grid": 64,
        "scale": 0.8,
        "boundary_conditions": [
            {
                "type": "cuboid",
                "point": [
                    object_center[0],
                    object_center[1],
                    0.15,
                ],
                "size": [1.5, 1.5, 0.1],
                "velocity": [0, 0, 0],
                "start_time": 0,
                "end_time": 1e3,
                "reset": 1,
            },
            {
                "type": "bounding_box",
            },
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
    }

    if "yield_stress" in params:
        config["yield_stress"] = params["yield_stress"]
    if "friction_angle" in params:
        config["friction_angle"] = params["friction_angle"]
    if "hardening" in params:
        config["hardening"] = params["hardening"]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"  Config saved to {output_path} (material={params['material']}, E={params['E']}, nu={params['nu']}, density={params['density']})")
    return config


def generate_configs_for_all(object_materials, output_root):
    for obj_id, material_name in object_materials.items():
        output_path = os.path.join(output_root, obj_id, "config.json")
        generate_config(material_name, output_path)


if __name__ == "__main__":
    output_root = os.path.join(os.path.dirname(__file__), "output")
    test_materials = {
        "chair": "jelly",
    }
    generate_configs_for_all(test_materials, output_root)
