import os
import sys
import subprocess
import argparse


def run_simulation(model_path, config_path, output_path=None):
    physg_dir = os.path.join(os.path.dirname(__file__), "..", "PhysGaussian")
    gs_sim = "gs_simulation.py"

    if output_path is None:
        output_path = os.path.join(model_path, "sim_output")

    os.makedirs(output_path, exist_ok=True)

    model_path = os.path.abspath(model_path)
    config_path = os.path.abspath(config_path)
    output_path = os.path.abspath(output_path)

    cmd = [
        sys.executable, gs_sim,
        "--model_path", model_path,
        "--config", config_path,
        "--output_path", output_path,
        "--output_ply",
        "--render_img",
    ]
    print(f"Running: {' '.join(cmd)}")
    print(f"Working directory: {physg_dir}")
    result = subprocess.run(cmd, cwd=physg_dir)
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--object_id", type=str, default="chair")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    output_root = os.path.join(os.path.dirname(__file__), "output")

    if args.object_id:
        obj_ids = [args.object_id]
    elif args.all:
        obj_ids = sorted(os.listdir(output_root))
    else:
        obj_ids = sorted(os.listdir(output_root))
        if obj_ids:
            obj_ids = [obj_ids[0]]

    for obj_id in obj_ids:
        model_path = os.path.join(output_root, obj_id)
        # Prefer multi-material config if available, else single-material
        config_path = os.path.join(output_root, obj_id, "config_multi.json")
        if not os.path.exists(config_path):
            config_path = os.path.join(output_root, obj_id, "config.json")
        if not os.path.exists(config_path):
            print(f"Skipping {obj_id}: config.json not found. Run generate_config.py first.")
            continue
        print(f"\nRunning simulation for {obj_id}...")
        ret = run_simulation(model_path, config_path)
        if ret != 0:
            print(f"Simulation failed for {obj_id}")


if __name__ == "__main__":
    main()
