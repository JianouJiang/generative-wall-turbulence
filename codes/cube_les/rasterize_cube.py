#!/usr/bin/env python3
"""
rasterize_cube.py -- convert nekRS spectral cube-array LES field files (.f#####)
into COMPACT regular-Cartesian float16 snapshot volumes for the genuinely-3D
wall->whole-field coupling dataset, and (watch mode) do it in a ROLLING fashion
so raw fields never overflow the pod disk.

Cartesian target grid (over the Coceal domain [0,2]x[0,4]x[0,2]) with the cube
[0.5,1.5]x[0,1]x[0.5,1.5] masked SOLID.  Interpolation = nearest-GLL (the map is
built ONCE from a reference field -- the mesh is static -- then reused; each
snapshot is then a fast gather).

Modes
-----
  build-map  <ref.fld>                 -> cube_interp_map.npz (idx, mask, grids)
  one        <field.fld> [<out.npy>]   -> rasterize a single field
  watch      <run_dir> <stack_dir>     -> poll for new *.f????? , rasterize into a
                                          growing float16 stack, archive/prune raw
                                          fields when disk gets tight.
"""
import os, sys, time, glob, json, shutil
import numpy as np
from pymech.neksuite import readnek

# ---- target Cartesian grid (1:2:1 aspect ~ isotropic spacing ~0.021) ----
NXC, NYC, NZC = 96, 192, 96
LX, LY, LZ = 2.0, 4.0, 2.0
CUBE = ((0.5, 1.5), (0.0, 1.0), (0.5, 1.5))     # x,y,z solid extents
HERE = os.path.dirname(os.path.abspath(__file__))
MAP = os.path.join(HERE, "cube_interp_map.npz")

def cart_grid():
    # cell centers
    xg = (np.arange(NXC) + 0.5) * LX / NXC
    yg = (np.arange(NYC) + 0.5) * LY / NYC
    zg = (np.arange(NZC) + 0.5) * LZ / NZC
    X, Y, Z = np.meshgrid(xg, yg, zg, indexing="ij")
    (x0, x1), (y0, y1), (z0, z1) = CUBE
    solid = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1) & (Z >= z0) & (Z <= z1)
    fluid = ~solid
    return xg, yg, zg, X, Y, Z, fluid

def gll_points(fld):
    d = readnek(fld)
    P = np.concatenate([e.pos.reshape(3, -1) for e in d.elem], axis=1)  # (3, Ngll)
    return d, P

def build_map(ref):
    from scipy.spatial import cKDTree
    print(f"[map] reading reference field {ref}")
    d, P = gll_points(ref)
    print(f"[map] Ngll={P.shape[1]:,}  building KDTree ...")
    tree = cKDTree(P.T)
    xg, yg, zg, X, Y, Z, fluid = cart_grid()
    tgt = np.stack([X[fluid], Y[fluid], Z[fluid]], axis=1)
    print(f"[map] querying {tgt.shape[0]:,} fluid targets ...")
    dist, idx = tree.query(tgt, k=1, workers=-1)
    print(f"[map] nearest-GLL dist: median={np.median(dist):.4f} p99={np.percentile(dist,99):.4f} max={dist.max():.4f}")
    np.savez(MAP, idx=idx.astype(np.int64), fluid=fluid,
             nx=NXC, ny=NYC, nz=NZC, xg=xg, yg=yg, zg=zg,
             cube=np.array(CUBE), nn_dist_med=float(np.median(dist)))
    print(f"[map] wrote {MAP}  ({fluid.sum():,} fluid cells)")

def rasterize(fld, m):
    d = readnek(fld)
    U = np.concatenate([e.vel[0].reshape(-1) for e in d.elem])
    V = np.concatenate([e.vel[1].reshape(-1) for e in d.elem])
    W = np.concatenate([e.vel[2].reshape(-1) for e in d.elem])
    idx = m["idx"]; fluid = m["fluid"]
    out = np.zeros((3, NXC, NYC, NZC), np.float32)
    for c, F in enumerate((U, V, W)):
        vol = np.zeros((NXC, NYC, NZC), np.float32)
        vol[fluid] = F[idx]
        out[c] = vol
    return out.astype(np.float16), float(d.time)

def watch(run_dir, stack_dir, min_free_gb=6.0, poll=15.0, keep_restart=2):
    os.makedirs(stack_dir, exist_ok=True)
    # auto-build the interp map from the first stable field if not present
    while not os.path.exists(MAP):
        flds = sorted(f for f in glob.glob(os.path.join(run_dir, "*.f?????"))
                      if os.path.getsize(f) > 0)
        if flds:
            s0 = os.path.getsize(flds[0]); time.sleep(2.0)
            if os.path.getsize(flds[0]) == s0:
                build_map(flds[0])
                break
        print("[watch] waiting for first field to build interp map ...")
        time.sleep(poll)
    m = np.load(MAP)
    seen = set()
    manifest = os.path.join(stack_dir, "manifest.jsonl")
    # Keep only the newest raw fields for restart.  Archiving every ~170 MB raw
    # checkpoint would exhaust the 50 GB data disk before the ~11 GB compact
    # 1,100-snapshot stack is complete.
    restart = os.path.join(run_dir, "restart"); os.makedirs(restart, exist_ok=True)
    print(f"[watch] polling {run_dir} for new fields -> {stack_dir}")
    while True:
        flds = sorted(f for f in glob.glob(os.path.join(run_dir, "*.f?????"))
                      if f not in seen and os.path.getsize(f) > 0)
        for f in flds:
            try:
                # skip if still being written (size changing)
                s0 = os.path.getsize(f); time.sleep(1.0)
                if os.path.getsize(f) != s0:
                    continue
                vol, t = rasterize(f, m)
                stem = os.path.splitext(os.path.basename(f))[0] + f"_t{t:.4f}.npy"
                np.save(os.path.join(stack_dir, stem), vol)
                with open(manifest, "a") as mf:
                    mf.write(json.dumps({"src": os.path.basename(f), "time": t, "npy": stem}) + "\n")
                seen.add(f)
                # Rolling restart retention: the compact raster is the science
                # dataset; at most ``keep_restart`` raw fields remain recoverable.
                free_gb = shutil.disk_usage(run_dir).free / 1e9
                dst = os.path.join(restart, os.path.basename(f))
                shutil.move(f, dst)
                kept = sorted(glob.glob(os.path.join(restart, "*.f?????")),
                              key=os.path.getmtime)
                for old in kept[:-keep_restart]:
                    os.remove(old)
                action = "restart-kept" if free_gb >= min_free_gb else "restart-kept/disk-low"
                print(f"[watch] rasterized {stem} t={t:.3f} | disk {free_gb:.1f}GB -> {action}")
            except Exception as ex:
                print(f"[watch] skip {f}: {ex}")
        time.sleep(poll)

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "build-map":
        build_map(sys.argv[2])
    elif cmd == "one":
        m = np.load(MAP)
        vol, t = rasterize(sys.argv[2], m)
        out = sys.argv[3] if len(sys.argv) > 3 else "snap.npy"
        np.save(out, vol)
        print(f"[one] wrote {out} shape={vol.shape} t={t:.4f} "
              f"|U|mean={np.abs(vol[0]).mean():.4f}")
    elif cmd == "watch":
        watch(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)
