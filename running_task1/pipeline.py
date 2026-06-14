import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from train_3dgs import train_single_object
from generate_config import generate_config
from material_database import get_material_params, list_materials


def find_latest_ply(output_dir):
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
    latest = max(iterations)
    ply_path = os.path.join(pc_dir, f"iteration_{latest}", "point_cloud.ply")
    if os.path.exists(ply_path):
        return ply_path
    return None


def auto_infer_material(dataset_dir, model=None, processor=None):
    """Use Qwen2.5-VL to infer material from dataset images."""
    from infer_physics import infer_material_from_images, infer_from_nerf_dataset, load_model

    print("  (no --material specified, auto-inferring from images...)")
    if model is None or processor is None:
        model, processor = load_model()
    image_paths = infer_from_nerf_dataset(dataset_dir)
    material = infer_material_from_images(image_paths, model, processor, verbose=True)
    return material


def main():
    parser = argparse.ArgumentParser(description="NeRF synthetic -> 3DGS -> PhysGaussian pipeline")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-infer", action="store_true")
    parser.add_argument("--iterations", type=int, default=30000)
    parser.add_argument("--dataset", type=str, default="chair",
                        help="Dataset name under nerf_synthetic/ (default: chair)")
    parser.add_argument("--material", type=str, default=None,
                        help="Material type for PhysGaussian sim. If not specified, auto-infer from images. "
                             "Use --list-materials to see all options.")
    parser.add_argument("--list-materials", action="store_true",
                        help="List all available material presets and exit.")
    parser.add_argument("--object-center", type=float, nargs=3, default=None,
                        help="Object center [x y z] in MPM space (default: [1.0, 1.0, 1.0])")
    args = parser.parse_args()

    if args.list_materials:
        from material_database import print_all_materials
        print_all_materials()
        return

    base_dir = os.path.dirname(__file__)
    dataset_dir = os.path.join(base_dir, "..", "nerf_synthetic", args.dataset)
    output_dir = os.path.join(base_dir, "output", args.dataset)

    if not os.path.isdir(dataset_dir):
        print(f"Dataset not found: {dataset_dir}")
        sys.exit(1)

    print(f"Processing dataset: {args.dataset}")
    print(f"Source: {dataset_dir}")

    if not args.skip_train:
        print("\n=== Step 1: Training 3DGS (native Blender/NeRF mode) ===")
        train_single_object(dataset_dir, output_dir, iterations=args.iterations)

    # Determine material
    if args.material:
        material = args.material
        print(f"\n=== Step 2: Using specified material: {material} ===")
    else:
        print(f"\n=== Step 2: Auto-inferring material from images ===")
        try:
            material = auto_infer_material(dataset_dir)
        except Exception as e:
            print(f"  Auto-inference failed: {e}")
            print(f"  Falling back to 'plasticine'")
            material = "plasticine"

    print(f"\n=== Step 3: Generating PhysGaussian config ===")
    object_center = args.object_center if args.object_center else [1.0, 1.0, 1.0]
    config_path = os.path.join(output_dir, "config.json")
    generate_config(material, config_path, object_center=object_center)

    print(f"\n{'='*60}")
    print(f"  Pipeline Complete")
    print(f"{'='*60}")
    ply_path = find_latest_ply(output_dir)
    params = get_material_params(material)
    print(f"\n  Object:   {args.dataset}")
    print(f"    PLY:      {ply_path}")
    print(f"    Config:   {config_path}")
    print(f"    Material: {material}")
    print(f"    MPM type: {params['material']}")
    print(f"    E={params['E']:.1e}, nu={params['nu']}, density={params['density']}")
    if 'yield_stress' in params:
        print(f"    yield_stress={params['yield_stress']:.1e}")
    if 'friction_angle' in params:
        print(f"    friction_angle={params['friction_angle']}°")
    if 'hardening' in params:
        print(f"    hardening={params['hardening']}")

    print(f"\nTo run PhysGaussian simulation:")
    print(f"  python run_simulation.py --object_id {args.dataset}")


if __name__ == "__main__":
    main()
