"""
PhysGaussian (Taichi reproduction) — SERVER / offline-render build
==================================================================
Original: Xie et al., PhysGaussian, CVPR 2024 (https://arxiv.org/abs/2311.12198)

This build targets a headless NVIDIA-CUDA server. It keeps the interactive
window mode of the dev build, and adds:
  - configurable backend (PG_ARCH=cuda|vulkan|cpu, auto-detect by default)
  - large capacity (N_MAX up to full 3DGS clouds; GRID configurable via PG_GRID)
  - rounder ellipsoid splats (icosphere subdiv via PG_ICO_SUBDIV)
  - real constitutive models: hard elastoplastic (plasticine/metal keep shape),
    sand (Drucker-Prager), foam — so landing no longer just "slumps"
  - HEADLESS offline rendering: --headless --frames N --out video.mp4
    (simulate + render frames to PNG, mux to mp4 via ffmpeg, orbiting camera)

Interactive run (needs a display + GPU backend):
  python main.py [--scene ...] [--preset ...] [--gs <dataset>]

Headless offline render (server):
  PG_ARCH=cuda python main.py --gs chair --headless --frames 360 \
      --out chair.mp4 --gs-particles 120000

Controls (interactive) — same as dev build:
  RMB orbit · W/S zoom · LMB poke · Space pause · 1-5 material ·
  B/X/M/C/T scene · G/H gravity · F point/ellipsoid · O obstacle · R reset
"""

import argparse
import os
import numpy as np
import taichi as ti

# Backend selection (server build): prefer CUDA on an NVIDIA box, fall back to
# Vulkan, then CPU. Override with PG_ARCH=cuda|vulkan|cpu. Headless offline
# rendering still needs a GPU backend for ti.ui; CPU works for sim-only checks.
def _init_taichi():
    want = os.environ.get("PG_ARCH", "").lower().strip()
    order = {"cuda": [ti.cuda], "vulkan": [ti.vulkan], "cpu": [ti.cpu],
             "gpu": [ti.gpu]}.get(want, [ti.cuda, ti.vulkan, ti.cpu])
    last = None
    for arch in order:
        try:
            ti.init(arch=arch, default_fp=ti.f32, random_seed=0)
            print(f"[init] Taichi backend: {arch}")
            return
        except Exception as e:
            last = e
    raise RuntimeError(f"no usable Taichi backend (tried {order}): {last}")

_init_taichi()

# ─── Simulation constants ─────────────────────────────────────────────────────
# Server build: capacity and grid are env-configurable so a CUDA box can run the
# full 3DGS cloud at higher physical resolution. Defaults match the dev build.
N_MAX   = int(os.environ.get("PG_NMAX", "300000"))   # capacity (CUDA can handle 27万+)
GRID    = int(os.environ.get("PG_GRID", "64"))       # background grid resolution
DX      = 1.0 / GRID
INV_DX  = float(GRID)
# CFL: stable DT ∝ DX. The dev build tuned DT=1e-4 at GRID=64, so scale it down
# as the grid gets finer (DX shrinks) to keep the same stability margin.
DT      = 1.0e-4 * (64.0 / GRID)
SUBSTEP = int(os.environ.get("PG_SUBSTEP", "24"))    # substeps per rendered frame
P_VOL   = (DX * 0.5) ** 3
P_RHO   = 20.0           # high density lowers elastic wave speed c=√(E/ρ), which
                         # is what lets this larger DT stay CFL-stable.
P_MASS  = P_VOL * P_RHO
BOUND   = 3              # grid cells of sticky/slip boundary

# Material model ids
JELLY  = 0               # fixed-corotated elastic (springs back)
SNOW   = 1               # corotated + plasticity (snow)
LIQUID = 2               # weakly-compressible fluid
PLASTIC = 3              # hard elastoplastic, von Mises return mapping —
                         # metal / plasticine / wood: deforms then HOLDS shape
SAND   = 4               # granular, Drucker-Prager return mapping (sand/soil)
FOAM   = 5               # crushable foam: permanent volumetric compaction

# ─── Particle (= Gaussian) state ───────────────────────────────────────────────
x   = ti.Vector.field(3, ti.f32, N_MAX)   # position  (Gaussian center)
v   = ti.Vector.field(3, ti.f32, N_MAX)   # velocity
C   = ti.Matrix.field(3, 3, ti.f32, N_MAX)  # affine velocity field (APIC)
F   = ti.Matrix.field(3, 3, ti.f32, N_MAX)  # deformation gradient
Jp  = ti.field(ti.f32, N_MAX)             # plastic volume ratio (snow)
mat = ti.field(ti.i32, N_MAX)             # material id
base_col = ti.Vector.field(3, ti.f32, N_MAX)  # rest albedo

# staging fields for injecting externally-loaded gaussians (gs_loader). We fill
# these from numpy via from_numpy(), then a kernel copies them into the live
# particle state. Kept separate so the .ply path never touches the hot loop.
_load_pos = ti.Vector.field(3, ti.f32, N_MAX)
_load_col = ti.Vector.field(3, ti.f32, N_MAX)

# ─── Background grid ───────────────────────────────────────────────────────────
grid_v = ti.Vector.field(3, ti.f32, (GRID, GRID, GRID))
grid_m = ti.field(ti.f32, (GRID, GRID, GRID))

# ─── Live (UI-controlled) parameters ───────────────────────────────────────────
p_E    = ti.field(ti.f32, ())   # Young's modulus
p_nu   = ti.field(ti.f32, ())   # Poisson ratio
p_grav = ti.field(ti.f32, ())   # gravity (negative = down)
p_damp = ti.field(ti.f32, ())   # velocity damping
p_yield = ti.field(ti.f32, ())  # plastic yield strength (PLASTIC): bigger = harder
p_fric = ti.field(ti.f32, ())   # sand friction angle proxy (SAND): bigger = steeper pile
n_par  = ti.field(ti.i32, ())   # active particle count

# centroid Y accumulator (written by g2p, read by Python for the poke plane)
centroid_y = ti.field(ti.f32, ())

# ─── External poke (mouse) ─────────────────────────────────────────────────────
poke_on  = ti.field(ti.i32, ())
poke_pos = ti.Vector.field(3, ti.f32, ())
poke_dir = ti.Vector.field(3, ti.f32, ())
POKE_R   = 0.09

# ─── Sphere obstacle (collision proxy, idea-3 flavour) ──────────────────────────
# A movable sphere collider. Enforced as a grid-velocity boundary condition in
# grid_op (the same mechanism as the floor/walls): grid nodes inside the sphere
# get their inward normal velocity removed, so the material slides around it.
obs_on  = ti.field(ti.i32, ())          # 0/1 toggle
obs_pos = ti.Vector.field(3, ti.f32, ())  # sphere center (world, [0,1]^3)
obs_r   = ti.field(ti.f32, ())          # sphere radius
obs_v   = ti.Vector.field(3, ti.f32, ())  # sphere velocity (so a *moving* sphere pushes material)

# ─── Render buffers: an ellipsoid mesh instanced per Gaussian ──────────────────
# A low-poly unit icosphere is deformed per particle by  G = F · diag(scale)
# so each Gaussian shows its true simulated anisotropy.
ELL_SCALE = ti.field(ti.f32, ())   # base Gaussian radius (UI "size")

def _make_icosphere(subdiv=1):
    """Unit icosphere (centered at origin, radius 1). Returns (verts, faces)."""
    t = (1.0 + 5.0 ** 0.5) / 2.0
    verts = [
        (-1,  t,  0), (1,  t,  0), (-1, -t,  0), (1, -t,  0),
        ( 0, -1,  t), (0,  1,  t), ( 0, -1, -t), (0,  1, -t),
        ( t,  0, -1), (t,  0,  1), (-t,  0, -1), (-t,  0,  1),
    ]
    verts = [np.array(p, np.float32) for p in verts]
    faces = [
        (0,11,5),(0,5,1),(0,1,7),(0,7,10),(0,10,11),
        (1,5,9),(5,11,4),(11,10,2),(10,7,6),(7,1,8),
        (3,9,4),(3,4,2),(3,2,6),(3,6,8),(3,8,9),
        (4,9,5),(2,4,11),(6,2,10),(8,6,7),(9,8,1),
    ]
    cache = {}

    def midpoint(a, b):
        key = (min(a, b), max(a, b))
        if key in cache:
            return cache[key]
        m = (verts[a] + verts[b]) * 0.5
        verts.append(m)
        idx = len(verts) - 1
        cache[key] = idx
        return idx

    for _ in range(subdiv):
        new_faces = []
        for a, b, c in faces:
            ab, bc, ca = midpoint(a, b), midpoint(b, c), midpoint(c, a)
            new_faces += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = new_faces

    V = np.stack([p / np.linalg.norm(p) for p in verts]).astype(np.float32)
    Fc = np.array(faces, np.int32)
    return V, Fc


_ICO_SUBDIV = int(os.environ.get("PG_ICO_SUBDIV", "0"))  # 0:12v/20f 1:42v/80f 2:162v/320f
_ICO_V, _ICO_F = _make_icosphere(subdiv=_ICO_SUBDIV)     # higher = rounder splats, heavier
VPP = _ICO_V.shape[0]          # vertices per particle
TPP = _ICO_F.shape[0]          # triangles per particle

# Unit-sphere template (constant) — local vertex directions (also the rest normals)
ico_v   = ti.Vector.field(3, ti.f32, VPP)

# Per-frame combined mesh (one deformed icosphere per Gaussian)
mesh_v  = ti.Vector.field(3, ti.f32, N_MAX * VPP)
mesh_n  = ti.Vector.field(3, ti.f32, N_MAX * VPP)
mesh_c  = ti.Vector.field(3, ti.f32, N_MAX * VPP)
mesh_id = ti.field(ti.i32, N_MAX * TPP * 3)

# Per-particle color for the fast point-splat path (scene.particles).
# The per-particle ellipsoid mesh above is the faithful "anisotropic Gaussian"
# view, but on MoltenVK its massive overlapping-triangle overdraw is ~70x
# slower than instanced point sprites — so points are the default render mode.
pt_c    = ti.Vector.field(3, ti.f32, N_MAX)

# Floor quad (two triangles)
floor_v = ti.Vector.field(3, ti.f32, 4)
floor_id = ti.field(ti.i32, 6)

# Single-point field used to render the obstacle sphere as one big particle.
obs_draw = ti.Vector.field(3, ti.f32, 1)

# ─── Helpers ───────────────────────────────────────────────────────────────────
@ti.func
def I3():
    return ti.Matrix.identity(ti.f32, 3)

@ti.func
def clampf(a, lo, hi):
    return ti.max(lo, ti.min(a, hi))


# ─── Constitutive model: Kirchhoff stress τ ────────────────────────────────────
# PhysGaussian uses standard MPM elastoplastic models. We implement the three
# the original paper demonstrates: fixed-corotated elasticity (jelly/rubber),
# snow plasticity, and a weakly-compressible fluid.
@ti.func
def kirchhoff_stress(Fp, mp, jp):
    E  = p_E[None]
    nu = p_nu[None]
    mu  = E / (2.0 * (1.0 + nu))
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    U, sig, V = ti.svd(Fp)
    J = Fp.determinant()
    R = U @ V.transpose()
    tau = ti.Matrix.zero(ti.f32, 3, 3)
    if mp == LIQUID:
        # only volumetric response; deviatoric part relaxes away
        tau = lam * J * (J - 1.0) * I3()
    elif mp == SNOW:
        h = ti.exp(10.0 * (1.0 - jp))      # plastic hardening
        tau = 2.0 * mu * h * (Fp - R) @ Fp.transpose() + lam * h * J * (J - 1.0) * I3()
    else:
        # JELLY / rubber / PLASTIC / SAND / FOAM: fixed-corotated elastic stress.
        # For the plastic materials, Fp here is the ELASTIC part of F (the g2p
        # return-mapping has already split off the plastic flow), so the same
        # elastic law gives the correct stress — the held shape comes from the
        # return mapping, not from a different stress formula.
        tau = 2.0 * mu * (Fp - R) @ Fp.transpose() + lam * J * (J - 1.0) * I3()
    return tau


@ti.kernel
def clear_grid():
    for I in ti.grouped(grid_m):
        grid_v[I] = ti.Vector.zero(ti.f32, 3)
        grid_m[I] = 0.0


@ti.kernel
def p2g():
    for p in range(n_par[None]):
        Xp = x[p] * INV_DX
        base = (Xp - 0.5).cast(ti.i32)
        fx = Xp - base.cast(ti.f32)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        tau = kirchhoff_stress(F[p], mat[p], Jp[p])
        # MLS-MPM momentum: stress + APIC affine term
        stress = (-DT * P_VOL * 4.0 * INV_DX * INV_DX) * tau
        affine = stress + P_MASS * C[p]
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offs = ti.Vector([i, j, k])
            dpos = (offs.cast(ti.f32) - fx) * DX
            wt = w[i][0] * w[j][1] * w[k][2]
            grid_v[base + offs] += wt * (P_MASS * v[p] + affine @ dpos)
            grid_m[base + offs] += wt * P_MASS


@ti.kernel
def grid_op():
    g = p_grav[None]
    damp = ti.exp(-p_damp[None] * DT)
    for I in ti.grouped(grid_m):
        if grid_m[I] > 0.0:
            vel = grid_v[I] / grid_m[I]
            vel.y += DT * g
            vel *= damp
            i, j, k = I
            # sticky walls + a floor that only blocks downward motion
            if i < BOUND and vel.x < 0: vel.x = 0.0
            if i > GRID - BOUND and vel.x > 0: vel.x = 0.0
            if j < BOUND and vel.y < 0: vel.y = 0.0
            if j > GRID - BOUND and vel.y > 0: vel.y = 0.0
            if k < BOUND and vel.z < 0: vel.z = 0.0
            if k > GRID - BOUND and vel.z > 0: vel.z = 0.0

            # sphere obstacle: slip boundary using RELATIVE velocity, so a
            # *moving* sphere pushes the material instead of passing through it.
            # We work in the sphere's frame (subtract obs_v), cancel the inward
            # normal part there, then add obs_v back — material ends up moving
            # with the sphere along the normal.
            if obs_on[None] == 1:
                node = ti.Vector([i, j, k]).cast(ti.f32) * DX
                rel  = node - obs_pos[None]
                dist = rel.norm()
                if dist < obs_r[None]:
                    nrm  = rel / (dist + 1e-8)        # outward normal
                    vrel = vel - obs_v[None]          # velocity in sphere frame
                    vn   = vrel.dot(nrm)
                    if vn < 0.0:                       # closing on the sphere
                        vrel -= vn * nrm               # cancel inward part (slip)
                        vel   = vrel + obs_v[None]
            grid_v[I] = vel


@ti.kernel
def g2p():
    centroid_y[None] = 0.0
    for p in range(n_par[None]):
        Xp = x[p] * INV_DX
        base = (Xp - 0.5).cast(ti.i32)
        fx = Xp - base.cast(ti.f32)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(ti.f32, 3)
        new_C = ti.Matrix.zero(ti.f32, 3, 3)
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offs = ti.Vector([i, j, k])
            dpos = (offs.cast(ti.f32) - fx) * DX
            wt = w[i][0] * w[j][1] * w[k][2]
            gv = grid_v[base + offs]
            new_v += wt * gv
            new_C += 4.0 * INV_DX * INV_DX * wt * gv.outer_product(dpos)

        # mouse poke: smooth radial impulse
        if poke_on[None] == 1:
            d = (x[p] - poke_pos[None]).norm()
            if d < POKE_R:
                new_v += ((1.0 - d / POKE_R) ** 2) * poke_dir[None]

        v[p] = new_v
        C[p] = new_C
        x[p] += DT * new_v

        # advect deformation gradient
        F_new = (I3() + DT * new_C) @ F[p]

        if mat[p] == LIQUID:
            # reset deformation to keep only volume → behaves like a fluid
            J = F_new.determinant()
            F_new = ti.Matrix.identity(ti.f32, 3) * J ** (1.0 / 3.0)
        elif mat[p] == SNOW:
            U, sig, Vv = ti.svd(F_new)
            Jp_new = Jp[p]
            sc = ti.Matrix.zero(ti.f32, 3, 3)
            for d in ti.static(range(3)):
                s = clampf(sig[d, d], 1.0 - 2.5e-2, 1.0 + 4.5e-3)   # snow plastic bounds
                Jp_new *= sig[d, d] / s
                sc[d, d] = s
            Jp[p] = clampf(Jp_new, 0.6, 20.0)
            F_new = U @ sc @ Vv.transpose()
        elif mat[p] == PLASTIC or mat[p] == FOAM:
            # von Mises return mapping in Hencky (log) strain space.
            # F stored is the ELASTIC deformation gradient; any deviatoric
            # strain beyond the yield threshold is shed as permanent plastic
            # flow. Result: the body deforms when hit, then HOLDS the new shape
            # (metal / plasticine / wood / foam) instead of springing back or
            # slumping. PLASTIC vs FOAM differ only by E / yield in the presets.
            U, sig, Vv = ti.svd(F_new)
            eps = ti.Vector.zero(ti.f32, 3)
            for d in ti.static(range(3)):
                eps[d] = ti.log(clampf(sig[d, d], 1e-4, 1e4))
            tr = eps[0] + eps[1] + eps[2]
            eps_hat = eps - (tr / 3.0) * ti.Vector([1.0, 1.0, 1.0])
            ehn = eps_hat.norm()
            mu_l = p_E[None] / (2.0 * (1.0 + p_nu[None]))
            yld = p_yield[None] / (2.0 * mu_l + 1e-9)   # strain-space yield radius
            sc = ti.Matrix.zero(ti.f32, 3, 3)
            if ehn > yld and ehn > 1e-9:
                new_eps = eps - (ehn - yld) * (eps_hat / ehn)   # back to surface
                for d in ti.static(range(3)):
                    sc[d, d] = ti.exp(new_eps[d])
                F_new = U @ sc @ Vv.transpose()
            # else: still elastic (within yield) → keep F_new unchanged
        elif mat[p] == SAND:
            # Drucker-Prager return mapping (cohesionless granular, Klár 2016).
            # Sand piles at a friction angle and cannot sustain tension.
            U, sig, Vv = ti.svd(F_new)
            eps = ti.Vector.zero(ti.f32, 3)
            for d in ti.static(range(3)):
                eps[d] = ti.log(clampf(sig[d, d], 1e-4, 1e4))
            tr = eps[0] + eps[1] + eps[2]
            eps_hat = eps - (tr / 3.0) * ti.Vector([1.0, 1.0, 1.0])
            ehn = eps_hat.norm()
            mu_l = p_E[None] / (2.0 * (1.0 + p_nu[None]))
            lam_l = p_E[None] * p_nu[None] / ((1.0 + p_nu[None]) * (1.0 - 2.0 * p_nu[None]))
            sc = ti.Matrix.zero(ti.f32, 3, 3)
            if tr > 0.0:
                # expansion: cohesionless → project to cone tip (no tension)
                for d in ti.static(range(3)):
                    sc[d, d] = 1.0
                F_new = U @ sc @ Vv.transpose()
            else:
                dg = ehn + (3.0 * lam_l + 2.0 * mu_l) / (2.0 * mu_l) * tr * p_fric[None]
                if dg > 0.0 and ehn > 1e-9:
                    new_eps = eps - (dg * eps_hat / ehn)
                    for d in ti.static(range(3)):
                        sc[d, d] = ti.exp(new_eps[d])
                    F_new = U @ sc @ Vv.transpose()
                # else: elastic (inside the cone) → keep F_new
        else:  # JELLY / rubber / putty : elastic
            # Anti-degeneracy clamp. Without this, a hard poke can drive det(F)
            # to ≤0 (an inverted / zero-volume element), after which the stress
            # blows up and particles get flung out ("detachment"). Clamp the
            # singular values so the element can compress/stretch a lot but can
            # never collapse or invert. This keeps the elastic response intact
            # in the normal range and only kicks in at extreme deformation.
            U, sig, Vv = ti.svd(F_new)
            sc = ti.Matrix.zero(ti.f32, 3, 3)
            for d in ti.static(range(3)):
                sc[d, d] = clampf(sig[d, d], 0.30, 3.0)
            F_new = U @ sc @ Vv.transpose()

        # Wide out-of-bounds guard ONLY. The real floor/wall collision happens on
        # the grid (grid_op). Clamping at the BC height here would stop particles
        # *before* they reach the grid boundary, so no contact velocity gradient
        # forms and the material never deforms — keep this guard well outside the
        # BC band so it never interferes, while still keeping the quadratic B-spline
        # stencil (base, base+1, base+2) inside [0, GRID-1].
        lo = 1.0 * DX
        hi = 1.0 - 2.0 * DX
        for d in ti.static(range(3)):
            x[p][d] = clampf(x[p][d], lo, hi)

        # sphere obstacle: hard position projection (anti-tunneling backstop).
        # The grid BC handles the soft contact; this guarantees no particle is
        # ever left inside the sphere even when it is dragged quickly across
        # resting material. Push any interior particle back out to the surface
        # and remove its remaining inward velocity relative to the sphere.
        if obs_on[None] == 1:
            rel  = x[p] - obs_pos[None]
            dist = rel.norm()
            if dist < obs_r[None]:
                nrm   = rel / (dist + 1e-8)
                x[p]  = obs_pos[None] + nrm * obs_r[None]
                vn    = (v[p] - obs_v[None]).dot(nrm)
                if vn < 0.0:
                    v[p] -= vn * nrm

        F[p] = F_new

        # accumulate centroid Y (atomic for parallel safety)
        ti.atomic_add(centroid_y[None], x[p].y)


def substep_once():
    clear_grid()
    p2g()
    grid_op()
    g2p()


# ─── Build the renderable mesh: one F-deformed ellipsoid per Gaussian ──────────
# This is the heart of "what you see is what you simulate": the rendered shape
# of Gaussian p is the unit sphere pushed through G = F_p · (ELL_SCALE).  As the
@ti.kernel
def build_points():
    # Fast render path: one color per particle, no mesh expansion.
    # Same J-based tint as build_mesh so the two modes look consistent.
    n = n_par[None]
    for p in range(n):
        J = F[p].determinant()
        bc = base_col[p]
        pt_c[p] = ti.Vector([
            clampf(bc[0] + 0.6 * ti.max(1.0 - J, 0.0), 0.0, 1.0),
            clampf(bc[1] - 0.3 * ti.abs(J - 1.0), 0.0, 1.0),
            clampf(bc[2] + 0.6 * ti.max(J - 1.0, 0.0), 0.0, 1.0),
        ])


# material stretches/compresses, F becomes anisotropic and so does the splat.
@ti.kernel
def build_mesh():
    s = ELL_SCALE[None]
    n = n_par[None]
    for p in range(n):
        Fp = F[p]
        # color: tint by local volume change J (compressed → warm, stretched → cool)
        J = Fp.determinant()
        bc = base_col[p]
        col = ti.Vector([
            clampf(bc[0] + 0.6 * ti.max(1.0 - J, 0.0), 0.0, 1.0),
            clampf(bc[1] - 0.3 * ti.abs(J - 1.0), 0.0, 1.0),
            clampf(bc[2] + 0.6 * ti.max(J - 1.0, 0.0), 0.0, 1.0),
        ])
        # transport: world offset = F · (s * unit_dir); normal = F^{-T} · dir
        Finv_T = Fp.inverse().transpose()
        for u in range(VPP):
            dir = ico_v[u]
            wpos = x[p] + Fp @ (s * dir)
            nrm = (Finv_T @ dir).normalized()
            idx = p * VPP + u
            mesh_v[idx] = wpos
            mesh_n[idx] = nrm
            mesh_c[idx] = col
    # rebuild index buffer for the active particles
    for p in range(n):
        for f in range(TPP):
            tri = p * TPP + f
            mesh_id[tri * 3 + 0] = p * VPP + _face0(f)
            mesh_id[tri * 3 + 1] = p * VPP + _face1(f)
            mesh_id[tri * 3 + 2] = p * VPP + _face2(f)


# Face lookups baked into Taichi funcs via constant fields (set at startup)
_f0 = ti.field(ti.i32, TPP)
_f1 = ti.field(ti.i32, TPP)
_f2 = ti.field(ti.i32, TPP)

@ti.func
def _face0(f): return _f0[f]
@ti.func
def _face1(f): return _f1[f]
@ti.func
def _face2(f): return _f2[f]


def upload_template():
    ico_v.from_numpy(_ICO_V)
    _f0.from_numpy(_ICO_F[:, 0].copy())
    _f1.from_numpy(_ICO_F[:, 1].copy())
    _f2.from_numpy(_ICO_F[:, 2].copy())
    # floor quad at y = floor height
    fy = (BOUND + 0.5) * DX
    floor_v.from_numpy(np.array([
        [0.0, fy, 0.0], [1.0, fy, 0.0], [1.0, fy, 1.0], [0.0, fy, 1.0]
    ], np.float32))
    floor_id.from_numpy(np.array([0, 1, 2, 0, 2, 3], np.int32))


# ─── Particle samplers ─────────────────────────────────────────────────────────
@ti.kernel
def fill_sphere(start: ti.i32, count: ti.i32,
                cx: ti.f32, cy: ti.f32, cz: ti.f32, r: ti.f32, mc: ti.i32,
                r0: ti.f32, g0: ti.f32, b0: ti.f32):
    for q in range(count):
        p = start + q
        # uniform sample inside a sphere
        u1, u2, u3 = ti.random(), ti.random(), ti.random()
        theta = 2.0 * 3.14159265 * u1
        phi = ti.acos(2.0 * u2 - 1.0)
        rr = r * u3 ** (1.0 / 3.0)
        x[p] = ti.Vector([cx + rr * ti.sin(phi) * ti.cos(theta),
                          cy + rr * ti.sin(phi) * ti.sin(theta),
                          cz + rr * ti.cos(phi)])
        v[p] = ti.Vector.zero(ti.f32, 3)
        F[p] = I3(); C[p] = ti.Matrix.zero(ti.f32, 3, 3); Jp[p] = 1.0; mat[p] = mc
        t = rr / r
        base_col[p] = ti.Vector([r0 * (0.5 + 0.5 * t), g0, b0 * (0.6 + 0.4 * (1.0 - t))])


@ti.kernel
def fill_box(start: ti.i32, count: ti.i32,
             cx: ti.f32, cy: ti.f32, cz: ti.f32,
             hx: ti.f32, hy: ti.f32, hz: ti.f32, mc: ti.i32,
             r0: ti.f32, g0: ti.f32, b0: ti.f32):
    for q in range(count):
        p = start + q
        lx = (ti.random() - 0.5) * 2.0 * hx
        ly = (ti.random() - 0.5) * 2.0 * hy
        lz = (ti.random() - 0.5) * 2.0 * hz
        x[p] = ti.Vector([cx + lx, cy + ly, cz + lz])
        v[p] = ti.Vector.zero(ti.f32, 3)
        F[p] = I3(); C[p] = ti.Matrix.zero(ti.f32, 3, 3); Jp[p] = 1.0; mat[p] = mc
        t = (ly + hy) / (2.0 * hy)
        base_col[p] = ti.Vector([r0, g0 * (0.4 + 0.6 * t), b0 * (1.0 - 0.4 * t)])


@ti.kernel
def fill_cylinder(start: ti.i32, count: ti.i32,
                  cx: ti.f32, cy: ti.f32, cz: ti.f32,
                  radius: ti.f32, half_h: ti.f32, mc: ti.i32,
                  r0: ti.f32, g0: ti.f32, b0: ti.f32):
    """Uniform fill of a vertical cylinder (axis = Y)."""
    for q in range(count):
        p = start + q
        ang = 2.0 * 3.14159265 * ti.random()
        rr  = radius * ti.sqrt(ti.random())     # sqrt → uniform over the disk area
        ly  = (ti.random() - 0.5) * 2.0 * half_h
        x[p] = ti.Vector([cx + rr * ti.cos(ang), cy + ly, cz + rr * ti.sin(ang)])
        v[p] = ti.Vector.zero(ti.f32, 3)
        F[p] = I3(); C[p] = ti.Matrix.zero(ti.f32, 3, 3); Jp[p] = 1.0; mat[p] = mc
        t = (ly + half_h) / (2.0 * half_h)       # 0 bottom → 1 top
        base_col[p] = ti.Vector([r0 * (0.4 + 0.6 * t), g0 * (0.7 + 0.3 * t), b0])


@ti.kernel
def fill_torus(start: ti.i32, count: ti.i32,
               cx: ti.f32, cy: ti.f32, cz: ti.f32,
               R: ti.f32, r: ti.f32, mc: ti.i32,
               r0: ti.f32, g0: ti.f32, b0: ti.f32):
    """Rejection-fill a torus: major radius R (in XZ plane), tube radius r.
    Sample the bounding cylinder, keep the last point that lands inside the tube;
    32 tries makes the miss-everything probability negligible for normal R/r."""
    for q in range(count):
        p = start + q
        ax = cx; ay = cy; az = cz
        for _t in ti.static(range(32)):
            ang   = 2.0 * 3.14159265 * ti.random()
            rr_xz = (R + r) * ti.sqrt(ti.random())
            px = rr_xz * ti.cos(ang)
            pz = rr_xz * ti.sin(ang)
            py = (ti.random() - 0.5) * 2.0 * r
            d_ring = ti.sqrt((ti.sqrt(px * px + pz * pz) - R) ** 2 + py * py)
            if d_ring < r:
                ax = cx + px; ay = cy + py; az = cz + pz
        x[p] = ti.Vector([ax, ay, az])
        v[p] = ti.Vector.zero(ti.f32, 3)
        F[p] = I3(); C[p] = ti.Matrix.zero(ti.f32, 3, 3); Jp[p] = 1.0; mat[p] = mc
        a = ti.atan2(az - cz, ax - cx)
        t = 0.5 + 0.5 * ti.sin(a * 3.0)          # banded color around the ring
        base_col[p] = ti.Vector([r0 * t, g0 * (1.0 - 0.5 * t), b0])


@ti.kernel
def set_material(mc: ti.i32):
    for p in range(n_par[None]):
        mat[p] = mc
        F[p]  = I3()
        Jp[p] = 1.0
        C[p]  = ti.Matrix.zero(ti.f32, 3, 3)  # fix: reset APIC affine to avoid phantom momentum


@ti.kernel
def inject_particles(count: ti.i32, mc: ti.i32):
    """Copy staged (pos,col) from _load_* into live particle state, with fresh
    physics (v=0, F=I, C=0, Jp=1). Used by the .ply loader path."""
    for p in range(count):
        x[p] = _load_pos[p]
        base_col[p] = _load_col[p]
        v[p] = ti.Vector.zero(ti.f32, 3)
        F[p] = I3()
        C[p] = ti.Matrix.zero(ti.f32, 3, 3)
        Jp[p] = 1.0
        mat[p] = mc


# (E, nu, material) per preset. E capped at real-time DT stability ceiling.
PRESETS = {
    "jelly":  (3.0e4, 0.30, JELLY),
    "rubber": (6.0e4, 0.42, JELLY),
    "putty":  (2.0e4, 0.48, JELLY),   # near-incompressible, slow creep
    "snow":   (8.0e4, 0.20, SNOW),
    "liquid": (8.0e3, 0.40, LIQUID),
    # server build: real elastoplastic / granular materials that hold shape.
    # NOTE: E is capped by CFL stability (~2e5 ceiling at this DT). "Hardness"
    # for metal/wood comes from YIELD strength, not from a bigger E — cranking
    # E past the ceiling makes the integrator explode (learned the hard way).
    "plasticine": (8.0e4, 0.30, PLASTIC),   # soft metal-clay: dents and stays
    "metal":      (1.2e5, 0.35, PLASTIC),   # stiff + very high yield: barely deforms
    "wood":       (1.0e5, 0.30, PLASTIC),   # hard, holds its shape
    "sand":       (6.0e4, 0.30, SAND),      # granular pile
    "foam":       (1.2e4, 0.10, FOAM),      # crushable, low yield
}

# Plastic yield strength (PLASTIC/FOAM) and sand friction proxy (SAND). Only
# read for those materials; harmless defaults for the rest. Bigger yield = harder
# (metal barely yields), bigger fric = steeper / less runny sand pile.
PLASTIC_PARAMS = {
    "plasticine": (2.5e3, 0.0),
    "metal":      (3.0e4, 0.0),
    "wood":       (1.5e4, 0.0),
    "foam":       (6.0e2, 0.0),
    "sand":       (0.0,   3.0),
}
DEFAULT_YIELD = 1.0e9   # effectively "never yields" for purely-elastic presets


def init_params(preset):
    E, nu, mc = PRESETS[preset]
    p_E[None] = E
    p_nu[None] = nu
    p_grav[None] = -9.8
    p_damp[None] = 1.0
    yld, fric = PLASTIC_PARAMS.get(preset, (DEFAULT_YIELD, 0.0))
    p_yield[None] = yld
    p_fric[None]  = fric
    ELL_SCALE[None] = 0.018
    poke_on[None] = 0
    # sphere obstacle: default OFF, parked just below the object as a "target"
    obs_on[None]  = 0
    obs_pos[None] = [0.5, 0.30, 0.5]
    obs_r[None]   = 0.10
    obs_v[None]   = [0.0, 0.0, 0.0]
    return mc


def apply_preset(preset):
    """Switch material live (keys 1-9 / panel): set E, nu, yield, friction AND
    re-tag every particle. Unlike init_params it does not move the object or
    reset gravity/obstacle — it only changes what the existing object is made of."""
    E, nu, mc = PRESETS[preset]
    p_E[None], p_nu[None] = E, nu
    yld, fric = PLASTIC_PARAMS.get(preset, (DEFAULT_YIELD, 0.0))
    p_yield[None] = yld
    p_fric[None]  = fric
    set_material(mc)


def init_scene(scene, preset):
    mc = init_params(preset)
    if scene == "ball":
        n_par[None] = 3000                     # 3000 ≈ 37fps vs 4000 ≈ 22fps; still looks full
        fill_sphere(0, 3000, 0.5, 0.62, 0.5, 0.13, mc, 0.35, 0.55, 0.95)
    elif scene == "box":
        n_par[None] = 3000
        fill_box(0, 3000, 0.5, 0.55, 0.5, 0.15, 0.12, 0.15, mc, 0.95, 0.55, 0.25)
    elif scene == "multi":
        # two spheres at different X so they don't pile on each other after landing
        n_par[None] = 4000
        fill_sphere(0,    2000, 0.32, 0.72, 0.5, 0.10, JELLY, 0.30, 0.55, 0.95)
        fill_sphere(2000, 2000, 0.68, 0.72, 0.5, 0.10, mc,    0.95, 0.45, 0.25)
    elif scene == "cylinder":
        n_par[None] = 3000
        fill_cylinder(0, 3000, 0.5, 0.62, 0.5, 0.10, 0.14, mc, 0.25, 0.75, 0.55)
    elif scene == "torus":
        n_par[None] = 3000
        fill_torus(0, 3000, 0.5, 0.62, 0.5, 0.10, 0.045, mc, 0.90, 0.50, 0.20)


def init_gs_scene(ply_path, config_path, n_target=6000):
    """Load a teammate's 3DGS .ply + PhysGaussian config and inject it as the
    live object. Returns a dict of info for the HUD (material name, counts, E).
    Falls back gracefully (raises) if files are missing — caller handles it."""
    import gs_loader as L
    ply = L.read_gaussian_ply(ply_path)
    cfg = L.load_config(config_path) if config_path else {}
    pos, rgb, norm = L.prepare_particles(ply, cfg, n_target=min(n_target, N_MAX))
    mat_id, E_demo, nu, mat_str, yld, fric = L.material_to_demo(cfg)

    m = pos.shape[0]
    # stage into the load fields (pad the unused tail; only first m are read)
    pad_pos = np.zeros((N_MAX, 3), np.float32); pad_pos[:m] = pos
    pad_col = np.zeros((N_MAX, 3), np.float32); pad_col[:m] = rgb
    _load_pos.from_numpy(pad_pos)
    _load_col.from_numpy(pad_col)

    # set params like init_params, then inject
    p_E[None] = E_demo
    p_nu[None] = nu
    p_grav[None] = -9.8
    p_damp[None] = 1.0
    p_yield[None] = yld
    p_fric[None]  = fric
    ELL_SCALE[None] = 0.012      # smaller splats: dense real cloud looks solid
    poke_on[None] = 0
    obs_on[None] = 0
    obs_pos[None] = [0.5, 0.30, 0.5]
    obs_r[None] = 0.10
    obs_v[None] = [0.0, 0.0, 0.0]

    n_par[None] = m
    inject_particles(m, mat_id)
    return {"material": mat_str, "mat_id": mat_id, "count": m,
            "E_real": float(cfg.get("E", 0.0)), "E_demo": E_demo, "nu": nu,
            "yield": yld, "fric": fric,
            "norm": norm,
            "raw": len(ply["xyz"])}


# ─── Orbit camera (manual RMB-drag control around a target) ────────────────────
class OrbitCamera:
    def __init__(self):
        self.target = np.array([0.5, 0.35, 0.5], np.float32)
        self.dist = 1.8
        self.az = 0.6      # azimuth (rad)
        self.el = 0.35     # elevation (rad)

    def position(self):
        ce, se = np.cos(self.el), np.sin(self.el)
        ca, sa = np.cos(self.az), np.sin(self.az)
        off = self.dist * np.array([ce * sa, se, ce * ca], np.float32)
        return self.target + off

    def orbit(self, daz, dele):
        self.az += daz
        self.el = float(np.clip(self.el + dele, -1.5, 1.5))

    def zoom(self, dz):
        self.dist = float(np.clip(self.dist - dz, 0.5, 4.0))

    def forward(self):
        f = self.target - self.position()
        return f / (np.linalg.norm(f) + 1e-9)

    def right(self):
        f = self.forward()
        r = np.cross(f, np.array([0, 1, 0], np.float32))
        return r / (np.linalg.norm(r) + 1e-9)

    def ray_from_screen(self, sx, sy, fov_deg=45.0, aspect=1.0):
        """Screen pos in [0,1]^2 → (origin, dir) in world space."""
        ndc_x = (sx - 0.5) * 2.0
        ndc_y = (sy - 0.5) * 2.0
        thf = np.tan(np.radians(fov_deg) * 0.5)
        f = self.forward()
        r = self.right()
        u = np.cross(r, f)
        d = f + ndc_x * aspect * thf * r + ndc_y * thf * u
        d = d / (np.linalg.norm(d) + 1e-9)
        return self.position(), d.astype(np.float32)


# ─── Main ───────────────────────────────────────────────────────────────────────
WIN_W, WIN_H = 1024, 768


def _snapshot_state(n):
    """Pull live particle state off the GPU as numpy, sliced to n particles.
    Returns (x, v, F, C) with shapes (n,3),(n,3),(n,3,3),(n,3,3)."""
    xs = x.to_numpy()[:n]
    vs = v.to_numpy()[:n]
    Fs = F.to_numpy()[:n]
    Cs = C.to_numpy()[:n]
    return xs, vs, Fs, Cs


def render_headless(args, gs_info, cur_scene):
    """Offline render: simulate args.frames frames, render each to PNG with an
    optionally-orbiting camera, then mux to mp4 via ffmpeg. No interactive window.
    Uses an offscreen GGUI window (show_window=False) — needs a GPU backend."""
    import subprocess, tempfile, shutil
    res = args.res
    hi = args.hifi
    n0 = n_par[None]
    print(f"[headless] scene={cur_scene} particles={n0} frames={args.frames} "
          f"res={res} hifi={hi} backend-render={'ellipsoid' if hi else 'points'}")

    # ── optional: per-frame state export aligned to official PhysGaussian ──
    exporter = None
    if args.export_state:
        import state_export
        norm = gs_info.get("norm") if gs_info else None
        exporter = state_export.StateExporter(
            args.export_state, n0,
            want_ply=args.export_ply, want_h5=args.export_h5, norm=norm)
        print(f"[export] -> {exporter.dir} (ply={args.export_ply} h5={args.export_h5}) "
              f"format=official save_data_at_frame")

    # offscreen window: render targets exist, nothing is shown on screen
    window = ti.ui.Window("pg-offline", (res, res), show_window=False,
                          vsync=False)
    canvas = window.get_canvas()
    canvas.set_background_color((0.07, 0.08, 0.10))
    scene = window.get_scene()
    cam = OrbitCamera()
    ti_cam = ti.ui.Camera()

    tmp = tempfile.mkdtemp(prefix="pg_frames_")
    try:
        # frame 0 = initial state (matches official: save BEFORE first p2g2p)
        if exporter is not None:
            xs, vs, Fs, Cs = _snapshot_state(n0)
            exporter.dump(0, xs, vs, Fs, Cs, t=0.0)
        for f in range(args.frames):
            for _ in range(SUBSTEP):
                substep_once()
            # export AFTER this frame's substeps -> frame index f+1, like official
            if exporter is not None:
                nn = n_par[None]
                xs, vs, Fs, Cs = _snapshot_state(nn)
                exporter.dump(f + 1, xs, vs, Fs, Cs,
                              t=(f + 1) * SUBSTEP * DT)
            # orbit the camera over time (args.orbit rad/sec, frames at args.fps)
            cam.az = 0.6 + args.orbit * (f / float(args.fps))
            n = n_par[None]
            ti_cam.position(*cam.position())
            ti_cam.lookat(*cam.target)
            ti_cam.up(0, 1, 0)
            ti_cam.fov(45)
            scene.set_camera(ti_cam)
            scene.ambient_light((0.45, 0.45, 0.5))
            scene.point_light(pos=(0.5, 2.0, 1.5), color=(0.9, 0.9, 0.85))
            scene.point_light(pos=(2.0, 1.0, 0.0), color=(0.4, 0.4, 0.5))
            scene.mesh(floor_v, indices=floor_id, color=(0.18, 0.19, 0.22),
                       two_sided=True)
            if hi:
                build_mesh()
                scene.mesh(mesh_v, indices=mesh_id, per_vertex_color=mesh_c,
                           vertex_count=n * VPP, index_count=n * TPP * 3)
            else:
                build_points()
                scene.particles(x, radius=ELL_SCALE[None], per_vertex_color=pt_c,
                                index_count=n)
            canvas.scene(scene)
            window.save_image(os.path.join(tmp, f"frame_{f:05d}.png"))
            if f % 30 == 0:
                print(f"[headless] frame {f}/{args.frames}")
        if exporter is not None:
            d = exporter.finalize()
            print(f"[export] wrote {exporter.frames} frames of state to {d} "
                  f"(+ transform.json)")
        # mux PNG sequence -> mp4 with ffmpeg
        if shutil.which("ffmpeg") is None:
            print(f"[headless] ffmpeg not found. Frames are in {tmp}/frame_*.png — "
                  f"mux them manually. Keeping the folder.")
            return
        cmd = ["ffmpeg", "-y", "-framerate", str(args.fps),
               "-i", os.path.join(tmp, "frame_%05d.png"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-crf", "18", args.out]
        print(f"[headless] muxing -> {args.out}")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[headless] ffmpeg failed:\n{r.stderr[-800:]}\nFrames kept in {tmp}")
            return
        print(f"[headless] wrote {args.out} ({args.frames} frames @ {args.fps}fps)")
    finally:
        # clean up frames only on success (ffmpeg present & ok); else leave them
        if shutil.which("ffmpeg") is not None and os.path.exists(args.out):
            shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="ball",
                    choices=["ball", "box", "multi", "cylinder", "torus"])
    ap.add_argument("--preset", default="jelly", choices=list(PRESETS.keys()))
    ap.add_argument("--gs", default=None,
                    help="Load a 3DGS object: a short name (e.g. 'chair', searched in "
                         "./data/<name>/), OR a direct path to a point_cloud.ply / its "
                         "folder. Overrides --scene.")
    ap.add_argument("--ply", default=None, help="Explicit .ply path (with --config).")
    ap.add_argument("--config", default=None, help="Explicit PhysGaussian config.json path.")
    ap.add_argument("--gs-particles", type=int, default=6000,
                    help="Target particle count after downsampling a loaded .ply. "
                         "Lower = smoother/faster frame counter (e.g. 4000 ~20fps), "
                         "higher = denser/more faithful. Server build raises the cap "
                         "via PG_NMAX; pass 120000+ on a CUDA box. Default 6000.")
    # ── headless offline rendering (server) ──
    ap.add_argument("--headless", action="store_true",
                    help="No window: simulate N frames, render to PNG, mux to mp4. "
                         "Needs a GPU backend (cuda/vulkan) for offscreen rendering.")
    ap.add_argument("--frames", type=int, default=360,
                    help="Headless: number of frames to render. Default 360 (12s@30fps).")
    ap.add_argument("--out", default="out.mp4",
                    help="Headless: output video path (mp4 via ffmpeg). Default out.mp4.")
    ap.add_argument("--fps", type=int, default=30, help="Headless: output video fps.")
    ap.add_argument("--orbit", type=float, default=0.5,
                    help="Headless: camera orbit speed (rad/sec). 0 = static camera.")
    ap.add_argument("--res", type=int, default=1280,
                    help="Headless: square render resolution (e.g. 1280, 1920).")
    ap.add_argument("--hifi", action="store_true",
                    help="Render F-deformed anisotropic ellipsoids (faithful, heavier) "
                         "instead of point splats. Recommended for offline video.")
    ap.add_argument("--export-state", default=None, metavar="DIR",
                    help="Headless: dump per-frame particle state into DIR/simulation_ply/ "
                         "in EXACTLY official PhysGaussian format (sim_{frame:010d}.h5/.ply "
                         "+ transform.json). Frame 0 is the initial state.")
    ap.add_argument("--export-ply", action="store_true",
                    help="With --export-state: write per-frame .ply (binary xyz). "
                         "Defaults on if neither --export-ply nor --export-h5 given.")
    ap.add_argument("--export-h5", action="store_true",
                    help="With --export-state: write per-frame .h5 (x/v/f_tensor/C/time). "
                         "Needs h5py.")
    args = ap.parse_args()
    # default: if exporting but no format flag, emit both (mirror official defaults)
    if args.export_state and not (args.export_ply or args.export_h5):
        args.export_ply = True
        args.export_h5 = True

    upload_template()
    cur_scene, cur_preset = args.scene, args.preset
    gs_info = None
    # ── .ply load path takes precedence over procedural scenes ──
    ply_path, config_path = args.ply, args.config
    if args.gs and not ply_path:
        import gs_loader as L
        ply_path, config_path = L.resolve_gs_paths(args.gs)
        if not ply_path:
            print(f"[gs] dataset '{args.gs}' NOT FOUND. Looked in:\n"
                  f"{L.gs_search_report(args.gs)}\n"
                  f"[gs] falling back to procedural scene '{cur_scene}'. "
                  f"(this is why you may see a ball instead of your object)")
    if ply_path:
        try:
            gs_info = init_gs_scene(ply_path, config_path, n_target=args.gs_particles)
            cur_scene = f"gs:{args.gs or os.path.basename(ply_path)}"
            cur_preset = gs_info["material"]
            print(f"[gs] loaded {gs_info['raw']} gaussians -> {gs_info['count']} particles | "
                  f"material={gs_info['material']} (id {gs_info['mat_id']}) | "
                  f"E {gs_info['E_real']:.1e} -> {gs_info['E_demo']:.0f} | nu={gs_info['nu']}")
        except Exception as e:
            print(f"[gs] failed to load ({e}); falling back to procedural scene.")
            gs_info = None
            init_scene(cur_scene, cur_preset)
    else:
        init_scene(cur_scene, cur_preset)

    # ── headless offline render path: no interactive window ──
    if args.export_state and not args.headless:
        print("[export] --export-state implies --headless; running offline.")
        args.headless = True
    if args.headless:
        render_headless(args, gs_info, cur_scene)
        return

    window = ti.ui.Window("PhysGaussian (Taichi)", (WIN_W, WIN_H),
                          vsync=(os.environ.get("PG_NOVSYNC") != "1"))
    canvas = window.get_canvas()
    canvas.set_background_color((0.07, 0.08, 0.10))
    scene = window.get_scene()
    cam = OrbitCamera()
    ti_cam = ti.ui.Camera()

    paused = False
    frame = 0
    last_mouse = None
    aspect = WIN_W / WIN_H
    hi_fidelity = (os.environ.get("PG_HIFI") == "1") or args.hifi  # F toggles F-deformed ellipsoids (slower, faithful)
    prev_obs = np.array(obs_pos[None].to_numpy(), np.float32)  # for obstacle velocity

    print("PhysGaussian running.  RMB orbit · W/S zoom · LMB poke · 1-5 material · B/X/M/C/T scene · Space pause · R reset · ESC quit")

    _maxframes = int(os.environ.get("PG_MAXFRAMES", "0"))
    _loops = 0
    while window.running:
        _loops += 1
        if _maxframes and _loops > _maxframes:
            window.running = False
        mx, my = window.get_cursor_pos()

        # ── keyboard: ONE event drain, dispatch by type (the original bug) ──
        for e in window.get_events(ti.ui.PRESS):
            k = e.key
            if k == ti.ui.ESCAPE:
                window.running = False
            elif k == ti.ui.SPACE:
                paused = not paused
            elif k == "r":
                if gs_info is not None:
                    init_gs_scene(ply_path, config_path, n_target=args.gs_particles)
                else:
                    init_scene(cur_scene, cur_preset)
                frame = 0
            elif k in ("1", "2", "3", "4", "5", "6", "7", "8", "9"):
                cur_preset = {"1": "jelly", "2": "rubber", "3": "putty",
                              "4": "snow",  "5": "liquid", "6": "plasticine",
                              "7": "metal", "8": "sand",   "9": "foam"}[k]
                apply_preset(cur_preset)
            elif k == "b":
                cur_scene = "ball";     gs_info = None; init_scene(cur_scene, cur_preset); frame = 0
            elif k == "x":
                cur_scene = "box";      gs_info = None; init_scene(cur_scene, cur_preset); frame = 0
            elif k == "m":
                cur_scene = "multi";    gs_info = None; init_scene(cur_scene, cur_preset); frame = 0
            elif k == "c":
                cur_scene = "cylinder"; gs_info = None; init_scene(cur_scene, cur_preset); frame = 0
            elif k == "t":
                cur_scene = "torus";    gs_info = None; init_scene(cur_scene, cur_preset); frame = 0
            elif k == "g":
                p_grav[None] = max(-30.0, p_grav[None] - 1.0)
            elif k == "h":
                p_grav[None] = min(10.0, p_grav[None] + 1.0)   # allow positive (antigravity)
            elif k == "o":
                obs_on[None] = 1 - obs_on[None]                 # toggle sphere obstacle
            elif k == "f":
                hi_fidelity = not hi_fidelity

        # ── held keys: continuous zoom ──
        if window.is_pressed("w"):
            cam.zoom(0.03)
        if window.is_pressed("s"):
            cam.zoom(-0.03)

        # ── held keys: move the obstacle sphere (arrows = X/Z, U/N = up/down) ──
        if obs_on[None] == 1:
            step = 0.006
            op = obs_pos[None]
            ox, oy, oz = op[0], op[1], op[2]
            if window.is_pressed(ti.ui.LEFT):  ox -= step
            if window.is_pressed(ti.ui.RIGHT): ox += step
            if window.is_pressed(ti.ui.UP):    oz -= step
            if window.is_pressed(ti.ui.DOWN):  oz += step
            if window.is_pressed("u"):         oy += step
            if window.is_pressed("n"):         oy -= step
            obs_pos[None] = [float(np.clip(ox, 0.05, 0.95)),
                             float(np.clip(oy, 0.05, 0.95)),
                             float(np.clip(oz, 0.05, 0.95))]

        # ── RMB drag → orbit ──
        if window.is_pressed(ti.ui.RMB):
            if last_mouse is not None:
                cam.orbit(-(mx - last_mouse[0]) * 4.0, (my - last_mouse[1]) * 3.0)
            last_mouse = (mx, my)
        else:
            last_mouse = None

        # ── LMB drag → poke: ray-cast to live centroid plane, push along view ──
        poke_on[None] = 0
        if window.is_pressed(ti.ui.LMB):
            ro, rd = cam.ray_from_screen(mx, my, aspect=aspect)
            # use the live centroid Y so poke works after the object has fallen
            poke_y = centroid_y[None] / max(n_par[None], 1)
            if abs(rd[1]) > 1e-4:
                t_hit = (poke_y - ro[1]) / rd[1]
                if 0.0 < t_hit < 10.0:
                    hit = ro + t_hit * rd
                    poke_pos[None] = [float(hit[0]), float(hit[1]), float(hit[2])]
                    # gentler impulse — the old 6.0/+3.0 push could drive F to
                    # inversion (det≤0) on repeated pokes and fling particles out.
                    push = cam.forward() * 3.0
                    push[1] += 1.5
                    poke_dir[None] = [float(push[0]), float(push[1]), float(push[2])]
                    poke_on[None] = 1

        # ── obstacle velocity: from frame-to-frame displacement, scaled to the
        #    sim time advanced this frame (SUBSTEP*DT) so a dragged sphere pushes
        #    material at the right speed. ──
        cur_obs = np.array(obs_pos[None].to_numpy(), np.float32)
        if obs_on[None] == 1 and not paused:
            ov = (cur_obs - prev_obs) / (SUBSTEP * DT)
            obs_v[None] = [float(ov[0]), float(ov[1]), float(ov[2])]
        else:
            obs_v[None] = [0.0, 0.0, 0.0]
        prev_obs = cur_obs

        # ── simulate ──
        if not paused:
            for _ in range(SUBSTEP):
                substep_once()
            frame += 1

        # ── build renderable + draw ──
        n = n_par[None]
        ti_cam.position(*cam.position())
        ti_cam.lookat(*cam.target)
        ti_cam.up(0, 1, 0)
        ti_cam.fov(45)
        scene.set_camera(ti_cam)
        scene.ambient_light((0.45, 0.45, 0.5))
        scene.point_light(pos=(0.5, 2.0, 1.5), color=(0.9, 0.9, 0.85))
        scene.point_light(pos=(2.0, 1.0, 0.0), color=(0.4, 0.4, 0.5))

        # floor
        scene.mesh(floor_v, indices=floor_id, color=(0.18, 0.19, 0.22), two_sided=True)

        # obstacle sphere: drawn as one big particle at obs_pos with radius obs_r
        if obs_on[None] == 1:
            obs_draw[0] = obs_pos[None]
            scene.particles(obs_draw, radius=obs_r[None], color=(0.85, 0.78, 0.45))

        # Gaussians.  Two render modes:
        #   fast  (default): instanced point splats — ~70x faster on MoltenVK.
        #   hi-fi (press F) : per-Gaussian F-deformed ellipsoids — the faithful
        #                     anisotropic view, but heavy due to triangle overdraw.
        if hi_fidelity:
            build_mesh()
            scene.mesh(mesh_v, indices=mesh_id, per_vertex_color=mesh_c,
                       vertex_count=n * VPP, index_count=n * TPP * 3)
        else:
            build_points()
            scene.particles(x, radius=ELL_SCALE[None], per_vertex_color=pt_c,
                            index_count=n)
        canvas.scene(scene)

        # ── control panel (replaces the flickering text overlay) ──
        gui = window.get_gui()
        with gui.sub_window("PhysGaussian", 0.02, 0.02, 0.32, 0.86):
            gui.text(f"scene   : {cur_scene}")
            gui.text(f"material: {cur_preset}")
            if gs_info is not None:
                gui.text(f"  (3DGS: {gs_info['raw']} -> {gs_info['count']} pts)")
                gui.text(f"  E {gs_info['E_real']:.0e} -> {gs_info['E_demo']:.0f}")
            gui.text(f"particles: {n}")
            gui.text(f"frame   : {frame}")
            cy_val = centroid_y[None] / max(n, 1)
            gui.text(f"centroid Y: {cy_val:.3f}")
            gui.text(("render  : ellipsoids (hi-fi)" if hi_fidelity
                      else "render  : points (fast)"))
            gui.text(f"obstacle: {'ON' if obs_on[None] else 'off'}")
            gui.text("PAUSED" if paused else "running")
            # material switch buttons (work even when keys are eaten by the panel)
            clicked = None
            for name in ("jelly", "rubber", "putty", "snow", "liquid",
                         "plasticine", "metal", "sand", "foam"):
                if gui.button(name):
                    clicked = name
            if clicked is not None:
                cur_preset = clicked
                apply_preset(cur_preset)
            # obstacle toggle button
            if gui.button("toggle obstacle (O)"):
                obs_on[None] = 1 - obs_on[None]
            p_E[None]    = gui.slider_float("Young E",  p_E[None], 1.0e3, 2.0e5)
            p_nu[None]   = gui.slider_float("Poisson",  p_nu[None], 0.05, 0.49)
            # plastic yield (metal/plasticine/wood/foam) and sand friction.
            # Only affect the corresponding materials; shown always for tuning.
            p_yield[None] = gui.slider_float("yield",  min(p_yield[None], 6.0e4), 0.0, 6.0e4)
            p_fric[None]  = gui.slider_float("sand fric", p_fric[None], 0.0, 8.0)
            p_grav[None] = gui.slider_float("gravity",  p_grav[None], -30.0, 10.0)
            p_damp[None] = gui.slider_float("damping",  p_damp[None], 0.0, 10.0)
            ELL_SCALE[None] = gui.slider_float("size",  ELL_SCALE[None], 0.004, 0.03)
            # obstacle controls
            op = obs_pos[None]
            nx = gui.slider_float("obs X", op[0], 0.05, 0.95)
            ny = gui.slider_float("obs Y", op[1], 0.05, 0.95)
            nz = gui.slider_float("obs Z", op[2], 0.05, 0.95)
            obs_pos[None] = [nx, ny, nz]
            obs_r[None]   = gui.slider_float("obs radius", obs_r[None], 0.04, 0.20)
            gui.text("RMB orbit  W/S zoom   LMB poke")
            gui.text("keys: 1-5 material (click if no focus)")
            gui.text("B/X/M/C/T scene  Space pause")
            gui.text("G/H gravity  F render  R reset")
            gui.text("O obstacle  arrows move  U/N up/down")

        window.show()


if __name__ == "__main__":
    main()

