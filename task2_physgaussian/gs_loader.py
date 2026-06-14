"""
gs_loader.py — load a teammate's 3DGS .ply + PhysGaussian config.json and turn
them into particle arrays this demo's MPM solver can ingest.

Why a hand-written .ply parser: this env only has numpy + taichi (no plyfile,
no torch). A 3DGS .ply is a text header followed by a flat float32 block, so a
~20-line numpy reader is all we need and adds zero dependencies.

Pipeline (see prepare_particles):
  read ply -> opacity filter -> opacity-weighted downsample (200k -> ~10k)
           -> Z-up (PhysGaussian) to Y-up (this demo) -> normalize into [0,1]^3
  read config -> material string + E/nu (E rescaled to the demo's stable range)

The heavy lifting is plain numpy; main.py only injects the arrays into Taichi
fields and never has to know about the .ply layout.
"""

import os
import json
import numpy as np

# SH degree-0 -> RGB constant (3DGS convention): rgb = f_dc * C0 + 0.5
SH_C0 = 0.28209479177387814


def read_gaussian_ply(path):
    """Parse a 3DGS binary_little_endian .ply.

    Returns a dict with:
      xyz     (N,3) float32 positions
      opacity (N,)  float32 in [0,1] (sigmoid already applied)
      rgb     (N,3) float32 in [0,1] (SH DC term -> color)
      scale   (N,3) float32 (real scale, exp already applied)
    """
    with open(path, "rb") as f:
        magic = f.readline().strip()
        if magic != b"ply":
            raise ValueError(f"not a ply file: {path}")
        fmt = f.readline().strip()
        if b"binary_little_endian" not in fmt:
            raise ValueError(f"only binary_little_endian supported, got {fmt!r}")
        n = 0
        props = []
        while True:
            line = f.readline().strip()
            if line.startswith(b"element vertex"):
                n = int(line.split()[-1])
            elif line.startswith(b"property"):
                # property <type> <name> ; all 3DGS props are float32
                props.append(line.split()[-1].decode())
            elif line == b"end_header":
                break
        raw = f.read(n * len(props) * 4)
    data = np.frombuffer(raw, dtype=np.float32).reshape(n, len(props))
    idx = {name: i for i, name in enumerate(props)}

    def col(name):
        return data[:, idx[name]]

    xyz = np.stack([col("x"), col("y"), col("z")], axis=1).astype(np.float32)
    # opacity stored as logit -> sigmoid
    opacity = (1.0 / (1.0 + np.exp(-col("opacity")))).astype(np.float32)
    # color from SH DC term
    fdc = np.stack([col("f_dc_0"), col("f_dc_1"), col("f_dc_2")], axis=1)
    rgb = np.clip(fdc * SH_C0 + 0.5, 0.0, 1.0).astype(np.float32)
    # scale stored in log space -> exp
    scale = np.exp(np.stack([col("scale_0"), col("scale_1"), col("scale_2")],
                            axis=1)).astype(np.float32)
    return {"xyz": xyz, "opacity": opacity, "rgb": rgb, "scale": scale}


def load_config(path):
    """Read a PhysGaussian config.json. Returns the raw dict; callers pull out
    material / E / nu / density / opacity_threshold as needed."""
    with open(path, "r") as f:
        return json.load(f)


def _weighted_downsample(n, weights, n_target, seed=0):
    """Pick n_target indices out of n, with probability proportional to weights
    (here: opacity). Keeps the solid core, drops faint floaters. Deterministic
    given seed so reloads are reproducible. Returns the selected index array."""
    if n <= n_target:
        return np.arange(n)
    w = weights.astype(np.float64)
    s = w.sum()
    if s <= 0:
        p = None  # degenerate: fall back to uniform
    else:
        p = w / s
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=n_target, replace=False, p=p)


def prepare_particles(ply, cfg, n_target=10000, fill_frac=0.30,
                      drop_y=0.62, seed=0):
    """Turn raw gaussian data + config into demo-ready particle arrays.

    Steps: opacity filter -> opacity-weighted downsample -> Z-up to Y-up ->
    normalize+center into a box of side `fill_frac` centred at x=z=0.5 and
    resting with its centre near y=`drop_y` (so it falls like the built-in
    scenes). Returns (pos[M,3] float32 in [0,1]^3, rgb[M,3] float32).
    """
    xyz = ply["xyz"]
    opacity = ply["opacity"]
    rgb = ply["rgb"]

    # 1) opacity filter (config threshold, default matches PhysGaussian)
    thr = float(cfg.get("opacity_threshold", 0.005))
    keep = opacity > thr
    xyz, opacity, rgb = xyz[keep], opacity[keep], rgb[keep]

    # 2) opacity-weighted downsample to fit N_MAX budget
    sel = _weighted_downsample(len(xyz), opacity, n_target, seed=seed)
    xyz, rgb = xyz[sel], rgb[sel]

    # 3) PhysGaussian space is Z-up; this demo is Y-up. Map (x,y,z)->(x,z,y).
    pos = np.stack([xyz[:, 0], xyz[:, 2], xyz[:, 1]], axis=1)

    # 4) normalize: center, scale longest extent to fill_frac, place in [0,1]^3
    lo, hi = pos.min(0), pos.max(0)
    center = (lo + hi) * 0.5
    extent = float((hi - lo).max())
    if extent <= 0:
        extent = 1.0
    pos = (pos - center) * (fill_frac / extent)        # centered, scaled
    pos[:, 0] += 0.5                                   # x -> middle
    pos[:, 2] += 0.5                                   # z -> middle
    pos[:, 1] += drop_y                                # y -> drop height
    pos = np.clip(pos, 0.02, 0.98).astype(np.float32)
    # normalization record, so callers can invert sim coords -> original 3DGS
    # world coords (Z-up). Inverse of steps 3+4 above:
    #   pos_swapped = (sim - offset) * extent / fill_frac + center
    #   world = unswap(pos_swapped)   where swap was (x,y,z)->(x,z,y)
    norm = {
        "axis_swap": "xyz_to_xzy",          # world(Z-up) (x,y,z) -> sim (x,z,y)
        "center": [float(c) for c in center],   # in swapped (Y-up) space
        "extent": float(extent),
        "fill_frac": float(fill_frac),
        "offset": [0.5, float(drop_y), 0.5],     # added after scaling, in sim space
        "clip": [0.02, 0.98],
    }
    return pos, rgb.astype(np.float32), norm


# ── material + units bridging (teammate's PhysGaussian world -> this demo) ──

# SERVER build: the demo now implements 6 constitutive models that match the
# teammate's set almost 1:1, so the mapping is faithful (no more "everything
# becomes soft jelly"). main.py ids: JELLY=0 SNOW=1 LIQUID=2 PLASTIC=3 SAND=4
# FOAM=5. plasticine/metal/wood -> PLASTIC (hold shape, differ by yield);
# snow -> SNOW; sand -> SAND; foam -> FOAM; jelly -> JELLY; liquid -> LIQUID.
JELLY, SNOW, LIQUID, PLASTIC, SAND, FOAM = 0, 1, 2, 3, 4, 5
MAT_MAP = {
    "jelly": JELLY,
    "plasticine": PLASTIC, "metal": PLASTIC, "wood": PLASTIC,
    "snow": SNOW,
    "sand": SAND,
    "foam": FOAM,
    "liquid": LIQUID,
}

# Per-material (yield, friction) for the plastic models, mirroring main.py's
# PLASTIC_PARAMS. None means "not a yield material" (use main.py default).
MAT_PLASTIC = {
    "plasticine": (2.5e3, 0.0),
    "metal":      (3.0e4, 0.0),
    "wood":       (1.5e4, 0.0),
    "foam":       (6.0e2, 0.0),
    "sand":       (0.0,   3.0),
}

# Real SI Young's modulus spans ~1e3 (whipped cream) .. ~2e11 (steel).
# The demo is CFL-stable only for E in ~[8e3, 2e5] (tuned for its DT/density).
# Map log-E linearly from the real range to the demo range so the *relative*
# softness ordering across materials is preserved (jelly stays softer than wood
# stays softer than metal) without blowing up the integrator.
_E_REAL_LO, _E_REAL_HI = 1.0e3, 2.0e11
_E_DEMO_LO, _E_DEMO_HI = 8.0e3, 2.0e5


def rescale_E(E_real):
    """Log-linear remap of a real SI Young's modulus into the demo's stable band."""
    e = float(np.clip(E_real, _E_REAL_LO, _E_REAL_HI))
    t = (np.log10(e) - np.log10(_E_REAL_LO)) / (np.log10(_E_REAL_HI) - np.log10(_E_REAL_LO))
    log_demo = np.log10(_E_DEMO_LO) + t * (np.log10(_E_DEMO_HI) - np.log10(_E_DEMO_LO))
    return float(10.0 ** log_demo)


def material_to_demo(cfg):
    """From a PhysGaussian config dict, return (mat_id, E_demo, nu, mat_str, yield, fric).
    SERVER build: mat_id in {0..5}; E rescaled to the stable band; for plastic
    materials a yield/friction pair carries the real 'hardness' so metal holds
    shape and plasticine dents. nu clamped to [0.05, 0.49]."""
    mat_str = str(cfg.get("material", "plasticine")).lower().strip()
    mat_id = MAT_MAP.get(mat_str, PLASTIC)
    E_demo = rescale_E(cfg.get("E", 1.0e6))
    nu = float(np.clip(cfg.get("nu", 0.3), 0.05, 0.49))
    yld, fric = MAT_PLASTIC.get(mat_str, (1.0e9, 0.0))
    return mat_id, E_demo, nu, mat_str, yld, fric


def resolve_gs_paths(dataset, base=None):
    """Locate a dataset's .ply + config.json. Returns (ply_path, config_path).

    `dataset` may be:
      - a direct path to a .ply file        -> used as-is (config looked up beside it)
      - a directory containing point_cloud.ply
      - a short name (e.g. 'chair')         -> searched in known locations below

    Search order for a short name (first hit wins):
      1. <script_dir>/data/<name>/point_cloud.ply        (flat layout, shipped in the package)
      2. <script_dir>/data/<name>/point_cloud/iteration_*/point_cloud.ply
      3. <script_dir>/../secsion1/running/output/<name>/...  (teammate's training tree)
    config.json is taken from the same folder as the .ply, else one level up.
    On failure returns (None, None) and the caller prints the searched paths via
    `gs_search_report(dataset)`.
    """
    here = os.path.dirname(os.path.abspath(__file__))

    # (0) direct path: a .ply file or a dir holding point_cloud.ply
    if dataset and (dataset.endswith(".ply") or os.path.sep in dataset
                    or os.path.isabs(dataset)):
        cand = dataset
        if os.path.isdir(cand):
            cand = os.path.join(cand, "point_cloud.ply")
        if os.path.exists(cand):
            return cand, _find_config_near(cand)

    roots = [base] if base else [
        os.path.join(here, "data"),
        os.path.join(here, "..", "secsion1", "running", "output"),
    ]
    for root in roots:
        out = os.path.join(root, dataset)
        # flat layout: data/<name>/point_cloud.ply
        flat = os.path.join(out, "point_cloud.ply")
        if os.path.exists(flat):
            return flat, _find_config_near(flat)
        # nested layout: <name>/point_cloud/iteration_*/point_cloud.ply
        pc_dir = os.path.join(out, "point_cloud")
        if os.path.isdir(pc_dir):
            iters = []
            for d in os.listdir(pc_dir):
                if d.startswith("iteration_"):
                    c = os.path.join(pc_dir, d, "point_cloud.ply")
                    if os.path.exists(c):
                        try:
                            iters.append((int(d.split("_")[1]), c))
                        except ValueError:
                            pass
            if iters:
                ply = max(iters)[1]
                return ply, _find_config_near(ply)
    return None, None


def _find_config_near(ply_path):
    """config.json in the same dir as the .ply, else one or two levels up."""
    d = os.path.dirname(os.path.abspath(ply_path))
    for cand in (os.path.join(d, "config.json"),
                 os.path.join(d, "..", "config.json"),
                 os.path.join(d, "..", "..", "config.json")):
        if os.path.exists(cand):
            return os.path.abspath(cand)
    return None


def gs_search_report(dataset):
    """Human-readable list of where we looked, for a helpful 'not found' message."""
    here = os.path.dirname(os.path.abspath(__file__))
    return "\n".join([
        f"  - {os.path.join(here, 'data', dataset, 'point_cloud.ply')}",
        f"  - {os.path.join(here, 'data', dataset, 'point_cloud', 'iteration_*', 'point_cloud.ply')}",
        f"  - {os.path.join(here, '..', 'secsion1', 'running', 'output', dataset, '...')}",
        f"  (or pass a direct path: --gs /abs/path/to/point_cloud.ply,",
        f"   or --ply <file> --config <file>)",
    ])




