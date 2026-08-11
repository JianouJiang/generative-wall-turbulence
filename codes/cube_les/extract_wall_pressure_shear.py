#!/usr/bin/env python3
"""Native wall-face pressure + signed two-tangent shear extractor (v1, offline).

Reads a nekRS field file (.f#####) from the validated aligned-cube case and
extracts, on uniform face rasters, the native no-slip wall observations:

  - static pressure p at the wall surface,
  - signed wall-on-fluid tangential viscous traction
    (tau_t1, tau_t2) = -mu * du_t/dn from a
    one-sided first-GLL-layer difference (first interior distance ~5.1e-4 h,
    y+ ~ 0.4 per the validated-run preflight),

for the six wall surfaces of the Coceal cell: floor (y=0, outside the cube
footprint), top (y=4), cube top (y=1 over the footprint), cube x-faces
(x=0.5, x=1.5 for y<1), cube z-faces (z=0.5, z=1.5 for y<1).

Sign convention: the normal n points from SOLID into FLUID; tau is the traction
exerted BY the wall ON the fluid resolved on the two in-face tangent axes, each
reconstructed to global Cartesian components before storage (gauge-invariant).

`audit` mode integrates streamwise wall friction + cube-face pressure drag over
all surfaces and compares with the constant-flow-rate driving force estimate;
v1 single-snapshot gate = ratio in [0.5, 2.0] (instantaneous imbalance is real
physics under constant-flow-rate forcing; the time-mean closure over the
continuation record is the registered follow-up, with a spectral-derivative
upgrade if needed).

This extractor defines the BA-0 native observation vector for the Orig lane.
It is NOT applicable to the frozen velocity-only ds2 record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    from pymech.neksuite import readnek
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"pymech required: {exc}")

MU = 1.0 / 5000.0  # rho=1, nu=1/5000 (cube_prod.par)
TOL = 1e-6
# face name -> (axis, wall_coord, normal_sign_into_fluid, tangent axes)
FACES = {
    "floor": (1, 0.0, +1, (0, 2)),
    "top": (1, 4.0, -1, (0, 2)),
    "cube_top": (1, 1.0, +1, (0, 2)),
    "cube_xlo": (0, 0.5, -1, (1, 2)),
    "cube_xhi": (0, 1.5, +1, (1, 2)),
    "cube_zlo": (2, 0.5, -1, (0, 1)),
    "cube_zhi": (2, 1.5, +1, (0, 1)),
}
# raster resolution per face (uniform cell centres over the face extent)
NRAST = 96


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def gll_nodes(n: int) -> np.ndarray:
    """Gauss-Lobatto-Legendre nodes on [-1,1] (endpoints included)."""
    from numpy.polynomial import legendre

    inner = legendre.Legendre.basis(n - 1).deriv().roots()
    return np.concatenate([[-1.0], np.sort(inner), [1.0]])


def attach_geometry_from_re2(field, mesh_path: Path) -> None:
    """Reconstruct exact GLL coordinates for an axis-aligned box mesh.

    nekRS checkpoints omit geometry.  gen_cube_mesh.py builds every hex as an
    axis-aligned box, so GLL nodes are the tensor product of 1-D GLL points
    affinely mapped to the element bounds read from the .re2.  Element-order
    consistency with the field file is verified downstream by the no-slip gate
    (velocity at reconstructed wall nodes must vanish).
    """
    from pymech.neksuite import readre2

    mesh = readre2(str(mesh_path))
    if len(mesh.elem) != len(field.elem):
        raise SystemExit(
            f"mesh nel {len(mesh.elem)} != field nel {len(field.elem)}"
        )
    n = field.elem[0].vel.shape[-1]
    g = 0.5 * (gll_nodes(n) + 1.0)  # [0,1]
    for fe, me in zip(field.elem, mesh.elem):
        bounds = [(float(me.pos[c].min()), float(me.pos[c].max())) for c in range(3)]
        xs = [lo + (hi - lo) * g for lo, hi in bounds]
        Z, Y, X = np.meshgrid(xs[2], xs[1], xs[0], indexing="ij")
        fe.pos = np.stack([X, Y, Z]).astype(np.float64)


def face_extent(name: str):
    if name == "floor" or name == "top":
        return (0.0, 2.0), (0.0, 2.0)  # x, z
    if name == "cube_top":
        return (0.5, 1.5), (0.5, 1.5)  # x, z
    if name.startswith("cube_x"):
        return (0.0, 1.0), (0.5, 1.5)  # y, z
    return (0.5, 1.5), (0.0, 1.0)  # x, y for cube_z*


def in_face(name: str, pts: np.ndarray) -> np.ndarray:
    x, y, z = pts
    if name == "floor":
        inside_fp = (x > 0.5 - TOL) & (x < 1.5 + TOL) & (z > 0.5 - TOL) & (z < 1.5 + TOL)
        return ~inside_fp
    if name == "top":
        return np.ones_like(x, bool)
    if name == "cube_top":
        return (x > 0.5 - TOL) & (x < 1.5 + TOL) & (z > 0.5 - TOL) & (z < 1.5 + TOL)
    if name.startswith("cube_x"):
        return (y < 1.0 + TOL) & (z > 0.5 - TOL) & (z < 1.5 + TOL)
    return (y < 1.0 + TOL) & (x > 0.5 - TOL) & (x < 1.5 + TOL)


def collect_wall_samples(field):
    """Return per-face arrays of wall nodes: position, p, du_t/dn (one-sided)."""
    out = {k: {"pos": [], "p": [], "dudn": [], "dn": []} for k in FACES}
    for elem in field.elem:
        pos = elem.pos  # (3, lz, ly, lx)
        vel = elem.vel
        pres = elem.pres[0]
        for name, (axis, wall, sgn, _tang) in FACES.items():
            coords = pos[axis]
            # local array axes are (z, y, x) -> derivative axis index in array
            arr_axis = {0: 2, 1: 1, 2: 0}[axis]
            lo = np.take(coords, 0, axis=arr_axis)
            hi = np.take(coords, -1, axis=arr_axis)
            side = None
            if np.all(np.abs(lo - wall) < TOL):
                side = 0
            elif np.all(np.abs(hi - wall) < TOL):
                side = -1
            if side is None:
                continue
            inner = 1 if side == 0 else -2
            wall_pos = np.stack(
                [np.take(pos[c], side, axis=arr_axis) for c in range(3)]
            )  # (3, a, b)
            keep = in_face(name, wall_pos.reshape(3, -1)).reshape(wall_pos.shape[1:])
            # solid side check: fluid must lie on the normal side
            interior_coord = np.take(coords, inner, axis=arr_axis)
            fluid_side = np.sign(np.median(interior_coord - wall))
            if fluid_side != sgn:
                continue
            if not keep.any():
                continue
            # A wall face is only claimed where the in-face filter keeps it, and
            # there the no-slip velocity must vanish (element-ordering gate).
            uw = np.stack([np.take(vel[c], side, axis=arr_axis) for c in range(3)])
            out[name].setdefault("uslip", []).append(float(np.abs(uw[:, keep]).max()))
            inner2 = 2 if side == 0 else -3
            d1 = np.abs(np.take(coords, inner, axis=arr_axis) - wall)
            d2 = np.abs(np.take(coords, inner2, axis=arr_axis) - wall)
            pw = np.take(pres, side, axis=arr_axis)
            dudn = []
            for c in range(3):
                u1 = np.take(vel[c], inner, axis=arr_axis)
                u2 = np.take(vel[c], inner2, axis=arr_axis)
                # second-order non-uniform one-sided derivative with u(0)=0:
                # f'(0) = u1*d2/(d1*(d2-d1)) - u2*d1/(d2*(d2-d1))
                dd = np.maximum(d2 - d1, 1e-12)
                dudn.append(u1 * d2 / np.maximum(d1 * dd, 1e-12) - u2 * d1 / np.maximum(d2 * dd, 1e-12))
            dudn = np.stack(dudn)  # (3, a, b) gradient of each velocity comp
            dn = d1
            k = keep.reshape(-1)
            out[name]["pos"].append(wall_pos.reshape(3, -1)[:, k])
            out[name]["p"].append(pw.reshape(-1)[k])
            out[name]["dudn"].append(dudn.reshape(3, -1)[:, k])
            out[name]["dn"].append(dn.reshape(-1)[k])
    for name in out:
        slip = out[name].pop("uslip", [])
        for key in list(out[name]):
            if isinstance(out[name][key], list) and out[name][key]:
                out[name][key] = np.concatenate(out[name][key], axis=-1)
            elif isinstance(out[name][key], list):
                out[name][key] = np.zeros((3, 0) if key in ("pos", "dudn") else (0,))
        out[name]["max_wall_slip"] = float(max(slip)) if slip else float("nan")
    return out


def rasterize_face(name: str, samples: dict):
    (a0, a1), (b0, b1) = face_extent(name)
    axis, wall, sgn, tang = FACES[name]
    plane_axes = [c for c in range(3) if c != axis]
    pos = samples["pos"]
    if pos.shape[1] == 0:
        return None
    ga = (np.arange(NRAST) + 0.5) * (a1 - a0) / NRAST + a0
    gb = (np.arange(NRAST) + 0.5) * (b1 - b0) / NRAST + b0
    from scipy.spatial import cKDTree

    tree = cKDTree(np.stack([pos[plane_axes[0]], pos[plane_axes[1]]], 1))
    A, B = np.meshgrid(ga, gb, indexing="ij")
    dist, idx = tree.query(np.stack([A.ravel(), B.ravel()], 1), k=1, workers=-1)
    p = samples["p"][idx].reshape(NRAST, NRAST)
    # Signed viscous traction exerted BY the wall ON the fluid. With n the
    # solid->fluid normal and u_wall = 0, the fluid-domain outward normal is -n,
    # so t_visc = -mu * du_t/dn (wall drags the fluid backwards).
    tau = np.stack(
        [(-MU * samples["dudn"][c][idx]).reshape(NRAST, NRAST) for c in range(3)]
    )
    tau[axis] = 0.0  # keep tangential projection only
    if name == "floor":
        (x0c, x1c), (z0c, z1c) = (0.5, 1.5), (0.5, 1.5)
        fp = (A >= x0c) & (A <= x1c) & (B >= z0c) & (B <= z1c)
        valid = ~fp
    else:
        valid = np.ones_like(A, bool)
    return {
        "p": p.astype(np.float32),
        "tau": tau.astype(np.float32),
        "valid": valid,
        "grid_a": ga,
        "grid_b": gb,
        "nearest_dist_p99": float(np.percentile(dist, 99)),
        "area_cell": float((a1 - a0) / NRAST * (b1 - b0) / NRAST),
        "tangent_axes": tang,
        "normal_axis": axis,
        "normal_sign": sgn,
    }


def extract(field_path: Path, out_path: Path | None, mesh_path: Path | None = None):
    t0 = time.time()
    field = readnek(str(field_path))
    if float(np.abs(field.elem[0].pos).max()) == 0.0:
        if mesh_path is None:
            raise SystemExit("field has no geometry; pass --mesh cube.re2")
        attach_geometry_from_re2(field, mesh_path)
    samples = collect_wall_samples(field)
    rasters = {}
    for name in FACES:
        r = rasterize_face(name, samples[name])
        if r is not None:
            rasters[name] = r
    slip = {k: samples[k].get("max_wall_slip", float("nan")) for k in samples}
    meta = {
        "field": str(field_path),
        "field_sha256": sha256(field_path),
        "time": float(field.time),
        "extractor_sha256": sha256(Path(__file__)),
        "mesh": str(mesh_path) if mesh_path else None,
        "mesh_sha256": sha256(mesh_path) if mesh_path else None,
        "mu": MU,
        "convention": "n solid->fluid; tau = viscous traction ON the fluid, tangential, global Cartesian",
        "wall_nodes": {k: int(samples[k]["p"].shape[0]) for k in samples},
        "max_wall_slip": slip,
        "no_slip_gate_lt_1e-6": bool(
            max(v for v in slip.values() if v == v) < 1e-6
        ),
        "seconds": round(time.time() - t0, 2),
    }
    if out_path is not None:
        np.savez_compressed(
            out_path,
            meta=json.dumps(meta),
            **{
                f"{n}_{k}": v
                for n, r in rasters.items()
                for k, v in r.items()
                if isinstance(v, np.ndarray)
            },
        )
    return field, samples, rasters, meta


def audit(field_path: Path, mesh_path: Path | None = None):
    field, samples, rasters, meta = extract(field_path, None, mesh_path)
    fx_visc = 0.0
    fx_pres = 0.0
    detail = {}
    for name, r in rasters.items():
        axis = r["normal_axis"]
        dA = r["area_cell"]
        v = r["valid"]
        f_v = float(r["tau"][0][v].sum() * dA)  # streamwise viscous drag on fluid
        fx_visc += f_v
        f_p = 0.0
        if name in ("cube_xlo", "cube_xhi"):
            # Pressure traction on the fluid = -p * n_out(fluid) = +p * n_into_fluid.
            # cube_xlo (x=0.5): fluid at x<0.5, n_into_fluid_x = -1 -> windward drag < 0.
            n_into_fluid = -1.0 if name == "cube_xlo" else +1.0
            f_p = float((r["p"][v] * dA).sum() * n_into_fluid)
            fx_pres += f_p
        detail[name] = {"fx_viscous": f_v, "fx_pressure": f_p, "nodes": meta["wall_nodes"][name]}
    fx_wall_on_fluid = fx_visc + fx_pres
    # Constant-flow-rate balance: applied body force per unit mass f satisfies
    # f * V_fluid + Fx_wall_on_fluid = 0 in the statistically steady state.
    v_fluid = 2.0 * 4.0 * 2.0 - 1.0
    f_last = 0.0219556  # validated-run momentum-balance value (yplus_preflight)
    balance = {
        "fx_viscous_total": fx_visc,
        "fx_pressure_cube": fx_pres,
        "fx_wall_on_fluid": fx_wall_on_fluid,
        "driving_force_estimate": f_last * v_fluid,
        "ratio_drag_to_driving": abs(fx_wall_on_fluid) / (f_last * v_fluid),
        "instantaneous_note": (
            "single-snapshot balance need not close exactly; steady-state closure "
            "holds in the mean. v1 gate = ratio in [0.5, 2.0] single snapshot."
        ),
        "gate_ratio_in_0p5_2p0": bool(0.5 < abs(fx_wall_on_fluid) / (f_last * v_fluid) < 2.0),
    }
    utau_floor = None
    if "floor" in rasters:
        r = rasters["floor"]
        utau_floor = float(np.sqrt(np.abs(r["tau"][0][r["valid"]]).mean()))
    report = {
        "meta": meta,
        "detail": detail,
        "balance": balance,
        "area_mean_utau_floor": utau_floor,
        "preflight_area_mean_utau": 0.1656638162061951,
        "finite_all": bool(
            all(np.isfinite(r["p"]).all() and np.isfinite(r["tau"]).all() for r in rasters.values())
        ),
    }
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["extract", "audit"])
    ap.add_argument("field")
    ap.add_argument("--mesh", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    fp = Path(args.field)
    mp = Path(args.mesh) if args.mesh else None
    if args.mode == "extract":
        out = Path(args.out) if args.out else fp.with_suffix(".wall.npz")
        _, _, rasters, meta = extract(fp, out, mp)
        print(json.dumps({"meta": meta, "faces": sorted(rasters)}, indent=1))
    else:
        rep = audit(fp, mp)
        out = Path(args.out) if args.out else fp.parent / "wall_extractor_audit.json"
        out.write_text(json.dumps(rep, indent=1))
        print(json.dumps({"no_slip": rep["meta"]["max_wall_slip"],
                          "no_slip_gate": rep["meta"]["no_slip_gate_lt_1e-6"]}, indent=1))
        print(json.dumps(rep["balance"], indent=1))
        print(f"[audit] wrote {out}")
        sys.exit(0 if rep["finite_all"] else 1)
