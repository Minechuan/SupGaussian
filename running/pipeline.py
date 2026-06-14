import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(__file__))

from train_3dgs import train_single_object
from generate_config import generate_config
from material_database import get_material_params, list_materials
from train_my_3dgs import train as train_my_3dgs


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
    parser.add_argument("--iterations", type=int, default=7000,
                        help="Training iterations (default: 7000 for my backend, 30000 for original)")
    parser.add_argument("--dataset", type=str, default="chair",
                        help="Dataset name under nerf_synthetic/ (default: chair)")
    parser.add_argument("--material", type=str, default=None,
                        help="Material type for PhysGaussian sim. If not specified, auto-infer from images. "
                             "Use --list-materials to see all options.")
    parser.add_argument("--list-materials", action="store_true",
                        help="List all available material presets and exit.")
    parser.add_argument("--object-center", type=float, nargs=3, default=None,
                        help="Object center [x y z] in MPM space (default: [1.0, 1.0, 1.0])")
    parser.add_argument("--gs-backend", type=str, default="my", choices=["my", "physgaussian"],
                        help="3DGS training backend: 'my' (own implementation) or 'physgaussian' (original). "
                             "Default: my")
    parser.add_argument("--multi-material", action="store_true",
                        help="Enable multi-material segmentation: SAM segments object parts → "
                             "per-point material assignment. Requires sam_vit_b.pth in running/")
    parser.add_argument("--num-materials", type=int, default=4,
                        help="Number of material clusters for multi-material mode (default: 4)")
    parser.add_argument("--num-seg-views", type=int, default=6,
                        help="Number of views to segment in multi-material mode (default: 6)")
    parser.add_argument("--material-hint", type=str, default=None,
                        help="Comma-separated material hints for multi-material mode, "
                             "e.g. 'wood,metal,fabric'. Restricts classification to these types.")
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
        if args.gs_backend == "my":
            print("\n=== Step 1: Training 3DGS (my implementation) ===")
            print(f"  Backend: my own 3DGS ({args.iterations} iterations)")
            train_my_3dgs(dataset_dir, output_dir, iterations=args.iterations)
        else:
            print("\n=== Step 1: Training 3DGS (PhysGaussian original) ===")
            print(f"  Backend: PhysGaussian ({args.iterations} iterations)")
            train_single_object(dataset_dir, output_dir, iterations=args.iterations)

    if args.multi_material:
        # ---- Multi-material pipeline ----
        print(f"\n=== Step 2: Multi-Material Segmentation & Assignment ===")
        from segment_material import run_multi_material_pipeline

        sam_ckpt = os.path.join(base_dir, "sam_vit_b.pth")
        hint_list = args.material_hint.split(",") if args.material_hint else None
        point_ids, mat_configs = run_multi_material_pipeline(
            dataset_dir, output_dir,
            iterations=args.iterations,
            sam_checkpoint=sam_ckpt,
            num_views=args.num_seg_views,
            num_materials=args.num_materials,
            visualize=True,
            material_hint=hint_list,
        )

        if point_ids is None:
            print("Multi-material pipeline failed. Falling back to single material.")
            material = "plasticine"
            print(f"\n=== Step 3: Generating PhysGaussian config (single material fallback) ===")
            object_center = args.object_center if args.object_center else [1.0, 1.0, 1.0]
            config_path = os.path.join(output_dir, "config.json")
            generate_config(material, config_path, object_center=object_center)
        else:
            print(f"\n=== Step 3: Multi-material config already generated ===")
            config_path = os.path.join(output_dir, "config_multi.json")
            material = "multi"  # special marker
            # Use default material params for fallback display
            default_mat = list(mat_configs.values())[0] if mat_configs else \
                          {"name": "plasticine", "material": "plasticine", "E": 1e10, "nu": 0.35, "density": 700}
    else:
        # ---- Single material pipeline ----
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

    if args.multi_material and point_ids is not None:
        # Multi-material summary
        print(f"\n  Object:   {args.dataset}")
        print(f"    PLY:      {ply_path}")
        print(f"    Config:   {config_path}")
        print(f"    Material IDs: {output_dir}/point_cloud/iteration_{args.iterations}/material_ids.npy")
        print(f"    Material Map: {output_dir}/point_cloud/iteration_{args.iterations}/material_map.json")
        print(f"\n  Regions detected:")
        for mat_id, params in mat_configs.items():
            count = (point_ids == int(mat_id)).sum()
            pct = 100 * count / len(point_ids)
            print(f"    [{mat_id}] {params['name']}: {count} points ({pct:.1f}%) - "
                  f"MPM={params['mpm']}, E={params['E']:.1e}, nu={params['nu']}")
        print(f"\n  Multi-material PLY: {output_dir}/point_cloud/iteration_{args.iterations}/point_cloud_multi.ply")
    else:
        # Single material summary
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
