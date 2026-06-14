import os
import sys
import subprocess


def train_single_object(data_dir, output_dir, iterations=30000):
    gs_dir = os.path.join(os.path.dirname(__file__), "..", "PhysGaussian", "gaussian-splatting")
    train_script = os.path.join(gs_dir, "train.py")

    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        sys.executable, train_script,
        "-s", data_dir,
        "-m", output_dir,
        "--iterations", str(iterations),
        "-w",
        "--port", "6009",
    ]
    print(f"  Running: {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = gs_dir + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(cmd, env=env, cwd=gs_dir)
    if result.returncode != 0:
        print(f"  Training failed for {data_dir} (exit code {result.returncode})")
        return False
    print(f"  Training completed, output: {output_dir}")
    return True


def train_all_objects(data_root, output_root, iterations=30000):
    object_ids = sorted(os.listdir(data_root))
    for obj_id in object_ids:
        data_dir = os.path.join(data_root, obj_id)
        output_dir = os.path.join(output_root, obj_id)
        if not os.path.isdir(data_dir):
            continue
        if not os.path.exists(os.path.join(data_dir, "sparse")):
            print(f"Skipping {obj_id}: no sparse data found")
            continue
        print(f"Training 3DGS for {obj_id}...")
        train_single_object(data_dir, output_dir, iterations)


if __name__ == "__main__":
    data_root = os.path.join(os.path.dirname(__file__), "data")
    output_root = os.path.join(os.path.dirname(__file__), "output")
    train_all_objects(data_root, output_root)
