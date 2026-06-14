"""state_export.py — dump per-frame MPM particle state in EXACTLY the format of
official PhysGaussian's `save_data_at_frame` (mpm_solver_warp/engine_utils.py).

Field-level, naming-level, shape-level, and file-naming alignment:

  simulation_ply/
    sim_0000000000.h5    # frame 0 = initial state, then 1..N
    sim_0000000000.ply
    ...
    transform.json       # EXTRA (not in official): records the sim<->world
                         # normalization so downstream can recover 3DGS world coords.

Official .h5 datasets (all stored TRANSPOSED, i.e. channels-first):
    x        (3, N)  float  particle position   [sim space, NOT un-normalized]
    v        (3, N)  float  particle velocity
    f_tensor (9, N)  float  deformation gradient F, row-major flatten of 3x3
    C        (9, N)  float  affine velocity (APIC), row-major flatten of 3x3
    time     (1, 1)  float  current sim time

Official .ply: binary_little_endian, element vertex N, properties float x/y/z.

Like the official code, `x` here is the raw MPM sim-space coordinate at save
time (no inverse transform applied) — that is what `save_data_at_frame` writes.
"""
import os
import json
import numpy as np

# h5py is optional: only needed for --export-h5. Imported lazily in dump().
try:
    import h5py
    _HAVE_H5PY = True
except Exception:
    _HAVE_H5PY = False


def _write_ply_xyz(filename, position):
    """Binary little-endian PLY with only float x/y/z — byte-identical layout to
    official particle_position_to_ply()."""
    position = np.ascontiguousarray(position, dtype=np.float32)
    n = position.shape[0]
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    if os.path.exists(filename):
        os.remove(filename)
    with open(filename, "wb") as f:
        f.write(header.encode())
        f.write(position.tobytes())


def _write_h5(filename, x, v, F, C, t):
    """Mirror official save_data_at_frame's h5 branch. All arrays stored
    transposed to channels-first, matching the official `.transpose()` calls."""
    if not _HAVE_H5PY:
        raise RuntimeError("h5py not installed; `pip install h5py` or use --export-ply only.")
    if os.path.exists(filename):
        os.remove(filename)
    with h5py.File(filename, "w") as nf:
        nf.create_dataset("x", data=x.T.astype(np.float64))            # (3, N)
        nf.create_dataset("time", data=np.array([[float(t)]]))         # (1, 1)
        nf.create_dataset("f_tensor", data=F.reshape(-1, 9).T.astype(np.float64))  # (9, N)
        nf.create_dataset("v", data=v.T.astype(np.float64))            # (3, N)
        nf.create_dataset("C", data=C.reshape(-1, 9).T.astype(np.float64))         # (9, N)


class StateExporter:
    """Per-frame dumper aligned to official PhysGaussian output.

    Usage (headless loop):
        exp = StateExporter(out_dir, n, want_ply=True, want_h5=True, norm=gs_norm)
        exp.dump(0, x_np, v_np, F_np, C_np, t=0.0)        # initial state
        for frame in range(N):
            ...substeps...
            exp.dump(frame + 1, x_np, v_np, F_np, C_np, t=sim_time)
        exp.finalize()
    """

    def __init__(self, out_dir, n_particles, want_ply=True, want_h5=True, norm=None):
        # Official writes into "<output_path>/simulation_ply"; match that.
        self.dir = os.path.join(out_dir, "simulation_ply")
        os.makedirs(self.dir, exist_ok=True)
        self.n = int(n_particles)
        self.want_ply = want_ply
        self.want_h5 = want_h5
        self.norm = norm
        self.frames = 0
        if want_h5 and not _HAVE_H5PY:
            raise RuntimeError(
                "--export-h5 needs h5py. Install it (`pip install h5py`) "
                "or pass --export-ply only.")

    def dump(self, frame, x, v, F, C, t):
        """x:(N,3) v:(N,3) F:(N,3,3) C:(N,3,3) — already sliced to live particles."""
        stem = os.path.join(self.dir, f"sim_{int(frame):010d}")
        if self.want_ply:
            _write_ply_xyz(stem + ".ply", x[: self.n])
        if self.want_h5:
            _write_h5(stem + ".h5", x[: self.n], v[: self.n],
                      F[: self.n], C[: self.n], t)
        self.frames += 1

    def finalize(self):
        """Write the extra transform.json (sim<->world recovery) and a manifest."""
        meta = {
            "format": "PhysGaussian save_data_at_frame (strict)",
            "n_particles": self.n,
            "frames_written": self.frames,
            "file_pattern": "sim_{frame:010d}.{ply,h5}",
            "frame0_is_initial_state": True,
            "h5_datasets": {
                "x": "(3, N) position, sim space",
                "v": "(3, N) velocity",
                "f_tensor": "(9, N) deformation gradient F, row-major 3x3",
                "C": "(9, N) affine velocity (APIC), row-major 3x3",
                "time": "(1, 1) sim time",
            },
            "normalization": self.norm,  # null for procedural scenes
        }
        with open(os.path.join(self.dir, "transform.json"), "w") as f:
            json.dump(meta, f, indent=2)
        return self.dir
