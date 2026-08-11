#!/usr/bin/env python3
"""Native GLL-face viscous traction and y+ audit for the cube-array LES.

This postprocessor reads the retained nekRS fields and the native ``.re2``
mesh.  It differentiates the spectral velocity polynomial at actual no-slip
faces, excludes pressure/form drag, and reports separate area-weighted
statistics for the exposed floor, cube top, and four vertical cube faces.
It does not alter or continue the LES.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import Legendre
from pymech.neksuite import readnek, readre2


CASE = Path(os.environ.get("GWT_CUBE_CASE", "controlled_raw/cube_les"))
MESH = CASE / "cube.re2"
FIELDS = sorted((CASE / "restart").glob("*.f?????"))
OUT = Path(__file__).resolve().parents[1] / "results/cube3d_native_yplus.json"
NU = 1.0 / 5000.0
EXPECTED_MESH_SHA256 = "09a68e5c763ba500c5dfe9d7281ef2d34557b1c4632707fd36a03ee4ce0b4128"
TOL = 2e-8


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def lgl_nodes_weights_diff(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Legendre--Gauss--Lobatto nodes, weights, and nodal derivative matrix."""
    if n < 2:
        raise ValueError("need at least two GLL nodes")
    degree = n - 1
    p = Legendre.basis(degree)
    x = np.concatenate(([-1.0], p.deriv().roots(), [1.0])).astype(np.float64)
    x.sort()
    w = 2.0 / (degree * (degree + 1) * p(x) ** 2)
    bary = np.ones(n, dtype=np.float64)
    for j in range(n):
        bary[j] = 1.0 / np.prod(x[j] - np.delete(x, j))
    d = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i != j:
                d[i, j] = bary[j] / (bary[i] * (x[i] - x[j]))
        d[i, i] = -np.sum(np.delete(d[i], i))
    if not np.allclose(d @ x, 1.0, rtol=0, atol=2e-13):
        raise RuntimeError("GLL derivative self-check failed")
    return x, w, d


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cdf = np.cumsum(w) / np.sum(w)
    return float(v[min(np.searchsorted(cdf, q, side="left"), len(v) - 1)])


def face_record(
    vel: np.ndarray,
    *,
    normal_axis: int,
    endpoint: int,
    lengths: tuple[float, float, float],
    dmat: np.ndarray,
    quad: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return y+, |tau_v|, surface weights, and maximum no-slip speed."""
    # pymech arrays are ordered (component, z, y, x).
    array_axis = {0: 3, 1: 2, 2: 1}[normal_axis]
    tangential_components = [c for c in range(3) if c != normal_axis]
    grad = []
    for comp in tangential_components:
        u = vel[comp]
        du_dxi = np.tensordot(dmat[endpoint], u, axes=(0, array_axis - 1))
        grad.append(du_dxi * 2.0 / lengths[normal_axis])
    tau_mag = NU * np.sqrt(np.square(grad[0]) + np.square(grad[1]))
    u_tau = np.sqrt(tau_mag)
    xi_first = abs(float(((-1.0 if endpoint == 0 else 1.0)
                          - (-1.0 if endpoint == 0 else 1.0))))
    # Exact first-interior GLL distance for either endpoint.
    nodes, _, _ = lgl_nodes_weights_diff(len(dmat))
    ref_delta = nodes[1] - nodes[0]
    first_distance = 0.5 * lengths[normal_axis] * ref_delta
    yplus = first_distance * u_tau / NU

    if normal_axis == 0:      # x-normal face -> (z, y)
        weights = np.outer(quad, quad) * (lengths[2] * lengths[1] / 4.0)
        wall_speed = np.linalg.norm(vel[:, :, :, endpoint], axis=0)
    elif normal_axis == 1:    # y-normal face -> (z, x)
        weights = np.outer(quad, quad) * (lengths[2] * lengths[0] / 4.0)
        wall_speed = np.linalg.norm(vel[:, :, endpoint, :], axis=0)
    else:                     # z-normal face -> (y, x)
        weights = np.outer(quad, quad) * (lengths[1] * lengths[0] / 4.0)
        wall_speed = np.linalg.norm(vel[:, endpoint, :, :], axis=0)
    if yplus.shape != weights.shape:
        raise RuntimeError(f"face shape mismatch {yplus.shape} != {weights.shape}")
    return yplus.ravel(), tau_mag.ravel(), weights.ravel(), float(np.max(wall_speed))


def summarize(values: np.ndarray, weights: np.ndarray) -> dict[str, float]:
    return {
        "area_weighted_mean": float(np.sum(values * weights) / np.sum(weights)),
        "area_weighted_p50": weighted_quantile(values, weights, 0.50),
        "area_weighted_p95": weighted_quantile(values, weights, 0.95),
        "area_weighted_p99": weighted_quantile(values, weights, 0.99),
        "maximum": float(np.max(values)),
    }


def classify_faces(bounds: tuple[float, float, float, float, float, float]):
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    faces: list[tuple[str, int, int]] = []
    if abs(ymin) < TOL:
        faces.append(("floor_between_cubes", 1, 0))
    if (abs(ymin - 1.0) < TOL and xmin >= 0.5 - TOL and xmax <= 1.5 + TOL
            and zmin >= 0.5 - TOL and zmax <= 1.5 + TOL):
        faces.append(("cube_top", 1, 0))
    if (ymin >= -TOL and ymax <= 1.0 + TOL
            and zmin >= 0.5 - TOL and zmax <= 1.5 + TOL):
        if abs(xmax - 0.5) < TOL:
            faces.append(("cube_vertical_faces", 0, -1))
        if abs(xmin - 1.5) < TOL:
            faces.append(("cube_vertical_faces", 0, 0))
    if (ymin >= -TOL and ymax <= 1.0 + TOL
            and xmin >= 0.5 - TOL and xmax <= 1.5 + TOL):
        if abs(zmax - 0.5) < TOL:
            faces.append(("cube_vertical_faces", 2, -1))
        if abs(zmin - 1.5) < TOL:
            faces.append(("cube_vertical_faces", 2, 0))
    return faces


def main() -> None:
    if sha256(MESH) != EXPECTED_MESH_SHA256:
        raise RuntimeError("cube mesh hash mismatch")
    if len(FIELDS) < 2:
        raise RuntimeError("need the two retained terminal nekRS fields")
    mesh = readre2(str(MESH))
    field_payload = []
    pooled: dict[str, dict[str, list[np.ndarray]]] = {
        k: {"yplus": [], "tau": [], "weight": []}
        for k in ("floor_between_cubes", "cube_top", "cube_vertical_faces")
    }
    max_wall_speed = 0.0
    for field_path in FIELDS:
        fld = readnek(str(field_path))
        if len(fld.elem) != len(mesh.elem):
            raise RuntimeError("mesh/field element-count mismatch")
        n = int(fld.elem[0].vel.shape[-1])
        _, quad, dmat = lgl_nodes_weights_diff(n)
        by_class: dict[str, dict[str, list[np.ndarray]]] = {
            k: {"yplus": [], "tau": [], "weight": []} for k in pooled
        }
        face_counts = {k: 0 for k in pooled}
        for em, ef in zip(mesh.elem, fld.elem):
            x = np.asarray(em.pos[0]); y = np.asarray(em.pos[1]); z = np.asarray(em.pos[2])
            bounds = (float(x.min()), float(x.max()), float(y.min()), float(y.max()),
                      float(z.min()), float(z.max()))
            lengths = (bounds[1] - bounds[0], bounds[3] - bounds[2], bounds[5] - bounds[4])
            for wall_class, normal_axis, endpoint in classify_faces(bounds):
                yp, tau, weight, speed = face_record(
                    np.asarray(ef.vel, dtype=np.float64), normal_axis=normal_axis,
                    endpoint=endpoint, lengths=lengths, dmat=dmat, quad=quad)
                by_class[wall_class]["yplus"].append(yp)
                by_class[wall_class]["tau"].append(tau)
                by_class[wall_class]["weight"].append(weight)
                face_counts[wall_class] += 1
                max_wall_speed = max(max_wall_speed, speed)
        record = {"field": field_path.name, "field_sha256": sha256(field_path),
                  "time": float(fld.time), "face_counts": face_counts, "classes": {}}
        for wall_class, arrays in by_class.items():
            if not arrays["yplus"]:
                raise RuntimeError(f"no faces found for {wall_class}")
            yp = np.concatenate(arrays["yplus"])
            tau = np.concatenate(arrays["tau"])
            weight = np.concatenate(arrays["weight"])
            record["classes"][wall_class] = {
                "yplus": summarize(yp, weight),
                "viscous_traction_magnitude": summarize(tau, weight),
                "area": float(np.sum(weight)),
            }
            for key, arr in (("yplus", yp), ("tau", tau), ("weight", weight)):
                pooled[wall_class][key].append(arr)
        field_payload.append(record)
    if max_wall_speed > 2e-5:
        raise RuntimeError(f"native no-slip check failed: max wall speed {max_wall_speed}")

    result = {
        "_meta": {
            "script": Path(__file__).name,
            "script_sha256": sha256(Path(__file__)),
            "mesh": str(MESH),
            "mesh_sha256": sha256(MESH),
            "field_count": len(FIELDS),
            "times": [r["time"] for r in field_payload],
            "nu": NU,
            "method": "native no-slip GLL-face spectral derivatives; viscous traction only",
            "pressure_form_drag_excluded": True,
            "max_native_wall_speed": max_wall_speed,
            "complete": True,
        },
        "per_field": field_payload,
        "pooled_two_terminal_fields": {},
    }
    for wall_class, arrays in pooled.items():
        yp = np.concatenate(arrays["yplus"])
        tau = np.concatenate(arrays["tau"])
        weight = np.concatenate(arrays["weight"])
        result["pooled_two_terminal_fields"][wall_class] = {
            "yplus": summarize(yp, weight),
            "viscous_traction_magnitude": summarize(tau, weight),
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUT.with_suffix(OUT.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True))
    os.replace(tmp, OUT)
    print(json.dumps(result["pooled_two_terminal_fields"], indent=2))
    print(f"=== done ===\n[out] {OUT} sha256={sha256(OUT)}")


if __name__ == "__main__":
    main()
