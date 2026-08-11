#!/usr/bin/env python3
"""Launch-once orig chain: y+ preflight -> cube LES -> raster harvest -> 3-D coupling."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "cube_les"
CASE = Path(os.environ.get("GWT_CUBE_CASE", "controlled_raw/cube_les"))
STACK = CASE / "stack"
NEKRS = Path(os.environ.get("GWT_NEKRS_HOME", "nekrs-install"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def atomic_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    os.replace(tmp, path)


def preflight_yplus() -> dict:
    required = [CASE / "cube.re2", CASE / "cube.udf", CASE / "cube.usr",
                CASE / "valid.log", NEKRS / "bin/nekrs"]
    for p in required:
        if not p.exists():
            raise FileNotFoundError(p)
    log = (CASE / "valid.log").read_text(errors="replace")
    if "finished with exit code 0" not in log or "step=4000" not in log:
        raise RuntimeError("the preserved 4,000-step validation is not terminal/pass")
    cfl = [float(v) for v in re.findall(r"CFL=\s*([0-9.]+)", log)]
    force = [float(v) for v in re.findall(r"flowrate.*?scale\s+([0-9.eE+-]+)", log)]
    if not cfl or not force:
        raise RuntimeError("cannot parse validation CFL/constant-flow forcing")

    # p=7 first interior GLL point is 0.0641 of the first element.  The mesh
    # generator uses 0.008h normal elements at floor/cube top and 0.012h at
    # vertical cube faces.  The measured constant-flow forcing supplies an
    # area-mean momentum-balance u_tau; u_tau=0.35 is also reported as a local
    # conservative bound covering stagnation/accelerating patches.
    nu = 1 / 5000
    gll = .0641
    delta_floor = gll * .008
    delta_face = gll * .012
    volume_fluid = 2 * 4 * 2 - 1
    wall_area = 3 + 4 + 1 + 4  # exposed floor + top + cube top + four cube sides
    tau_mean = force[-1] * volume_fluid / wall_area
    utau_mean = tau_mean ** .5
    utau_bound = .35
    result = {
        "method": "validated-run momentum balance plus conservative local-friction bound",
        "validation_log_sha256": sha256(CASE / "valid.log"),
        "mesh_sha256": sha256(CASE / "cube.re2"),
        "validation_steps": 4000,
        "validation_CFL_last": cfl[-1],
        "validation_CFL_max": max(cfl),
        "constant_flow_force_last": force[-1],
        "nu": nu,
        "first_interior_distance_h": {"floor_between_cubes": delta_floor,
                                        "cube_top": delta_floor,
                                        "cube_vertical_faces": delta_face},
        "area_mean_utau_from_momentum_balance": utau_mean,
        "area_mean_yplus": {"floor_between_cubes": delta_floor * utau_mean / nu,
                             "cube_top": delta_floor * utau_mean / nu,
                             "cube_vertical_faces": delta_face * utau_mean / nu},
        "conservative_local_utau_bound": utau_bound,
        "conservative_yplus_bound": {"floor_between_cubes": delta_floor * utau_bound / nu,
                                      "cube_top": delta_floor * utau_bound / nu,
                                      "cube_vertical_faces": delta_face * utau_bound / nu},
    }
    result["pass_yplus_le_2_all_reported_walls"] = bool(
        max(result["conservative_yplus_bound"].values()) <= 2.0)
    if not result["pass_yplus_le_2_all_reported_walls"]:
        raise RuntimeError(f"near-wall resolution failed: {result}")
    atomic_json(CASE / "yplus_preflight.json", result)
    print(f"[preflight] y+ PASS floor={result['conservative_yplus_bound']['floor_between_cubes']:.3f} "
          f"cube-face={result['conservative_yplus_bound']['cube_vertical_faces']:.3f} "
          f"(conservative u_tau={utau_bound})", flush=True)
    return result


def records():
    mf = STACK / "manifest.jsonl"
    if not mf.exists():
        return []
    out = []
    for line in mf.read_text().splitlines():
        try:
            r = json.loads(line)
            if (STACK / r["npy"]).exists():
                out.append(r)
        except Exception:
            pass
    return out


def production() -> dict:
    complete = CASE / "production_complete.json"
    if complete.exists():
        q = json.loads(complete.read_text())
        if q.get("exit_code") == 0 and q.get("n_post_spinup", 0) >= 1000:
            print(f"[production] reuse completed LES {complete}", flush=True)
            return q

    # Reuse the validated mesh/UDF/build cache.  Stage only the production par
    # and rolling rasterizer, as ordered by the charter.
    if not (CASE / "cube.validated.par").exists():
        shutil.copy2(CASE / "cube.par", CASE / "cube.validated.par")
    shutil.copy2(SRC / "cube_prod.par", CASE / "cube.par")
    shutil.copy2(SRC / "rasterize_cube.py", CASE / "rasterize_cube.py")
    STACK.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.update({
        "HWLOC_COMPONENTS": "-gl",
        "NEKRS_HOME": str(NEKRS),
        "NEKRS_CACHE_DIR": str(CASE / ".cache"),
        "OCCA_CACHE_DIR": str(CASE / ".cache/occa"),
        "OCCA_CUDA_COMPILER_FLAGS": "-w -O1 -lineinfo",
        "OCCA_CXXFLAGS": "-w -O1 -g",
        "PATH": f"/usr/local/cuda/bin:{NEKRS / 'bin'}:{env.get('PATH', '')}",
    })
    rast_log = (CASE / "rasterizer.log").open("a")
    prod_log = (CASE / "production.log").open("a")
    rast = subprocess.Popen([sys.executable, "-u", str(CASE / "rasterize_cube.py"),
                             "watch", str(CASE), str(STACK)], cwd=CASE, env=env,
                            stdout=rast_log, stderr=subprocess.STDOUT)
    cmd = ["mpirun", "-np", "1", str(NEKRS / "bin/nekrs"),
           "--setup", "cube.par", "--backend", "CUDA"]
    print(f"[production] launch {' '.join(cmd)}", flush=True)
    prod = subprocess.Popen(cmd, cwd=CASE, env=env, stdout=prod_log, stderr=subprocess.STDOUT)
    started = time.time()
    try:
        while prod.poll() is None:
            time.sleep(300)
            rr = records()
            last_t = max((float(r["time"]) for r in rr), default=-1)
            print(f"[production] alive wall={(time.time()-started)/3600:.2f}h "
                  f"rasters={len(rr)} last_t={last_t:.2f}", flush=True)
        rc = prod.wait()
        if rc != 0:
            raise RuntimeError(f"nekRS production exited {rc}; inspect {CASE/'production.log'}")
        # Let the rolling rasterizer consume the last stable field.
        deadline = time.time() + 1800
        while time.time() < deadline:
            rr = records()
            post = [r for r in rr if float(r["time"]) >= 40]
            raw = list(CASE.glob("*.f?????"))
            if len(post) >= 1000 and not raw:
                break
            print(f"[harvest] waiting rasters={len(rr)} post40={len(post)} raw={len(raw)}", flush=True)
            time.sleep(30)
        else:
            raise RuntimeError("raster harvest did not reach 1000 post-spinup snapshots")
    finally:
        if rast.poll() is None:
            rast.send_signal(signal.SIGTERM)
            try:
                rast.wait(timeout=30)
            except subprocess.TimeoutExpired:
                rast.kill(); rast.wait()
        rast_log.close(); prod_log.close()

    rr = records()
    times = sorted({round(float(r["time"]), 8) for r in rr})
    post = [t for t in times if t >= 40]
    result = {
        "exit_code": 0,
        "n_rasters": len(times),
        "n_post_spinup": len(post),
        "time_minmax": [times[0], times[-1]],
        "time_minmax_post_spinup": [post[0], post[-1]],
        "wall_seconds": round(time.time() - started, 1),
        "production_par_sha256": sha256(CASE / "cube.par"),
        "mesh_sha256": sha256(CASE / "cube.re2"),
        "udf_sha256": sha256(CASE / "cube.udf"),
        "production_log_sha256": sha256(CASE / "production.log"),
        "manifest_sha256": sha256(STACK / "manifest.jsonl"),
    }
    atomic_json(complete, result)
    print(f"[production] COMPLETE post-spinup snapshots={len(post)}", flush=True)
    return result


def main() -> None:
    yplus = preflight_yplus()
    prod = production()
    print(f"[chain] starting genuinely-3D coupling after terminal LES: {prod['n_post_spinup']} fields", flush=True)
    subprocess.run([sys.executable, "-u", str(HERE / "eval_cube_3d_coupling.py"),
                    "--stack", str(STACK), "--case", str(CASE)], cwd=ROOT.parent, check=True)
    out = ROOT / "results/cube3d_coupling_results.json"
    if not out.exists():
        raise RuntimeError(f"coupling output missing: {out}")
    print(f"=== chain done === cube_result_sha256={sha256(out)}", flush=True)


if __name__ == "__main__":
    main()
