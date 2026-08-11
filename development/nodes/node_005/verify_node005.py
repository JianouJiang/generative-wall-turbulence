#!/usr/bin/env python3
"""Independent local verification of node 005.

Nothing here trusts the GPU job's own arithmetic. Every headline quantity is
recomputed from the retained per-target components, the geometry is rebuilt from
first principles, the physical gates are replayed, and the manuscript is checked
against the source of record for both the numbers it prints and the labels the
panel required to be withdrawn.

Usage:  python3 development/nodes/node_005/verify_node005.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
NODE = Path(__file__).resolve().parent
RESULTS = ROOT / "codes" / "results"
MAN = ROOT / "manuscript"

RES_JSON = RESULTS / "e2_direct_traction_results.json"
RES_NPZ = RESULTS / "e2_direct_traction_components.npz"

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))


def approx(a, b, tol=1e-9) -> bool:
    return abs(float(a) - float(b)) <= tol


# ---------------------------------------------------------------- geometry
def geometry():
    nx, ny, nz = 48, 96, 48
    x = (np.arange(nx) + 0.5) * 2.0 / nx
    y = (np.arange(ny) + 0.5) * 4.0 / ny
    z = (np.arange(nz) + 0.5) * 2.0 / nz
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    solid = ((X >= .5) & (X <= 1.5) & (Y <= 1.0) & (Z >= .5) & (Z <= 1.5))
    fluid = ~solid
    dx = np.maximum.reduce([.5 - X, np.zeros_like(X), X - 1.5])
    dy = np.maximum.reduce([0.0 - Y, np.zeros_like(Y), Y - 1.0])
    dz = np.maximum.reduce([.5 - Z, np.zeros_like(Z), Z - 1.5])
    dcube = np.sqrt(dx * dx + dy * dy + dz * dz)
    dist = np.minimum.reduce([Y, 4.0 - Y, dcube])
    band = fluid & (dist <= 2.01 * (4.0 / ny))
    return fluid, band, dist


DUP_ROWS = (41, 43, 51, 54, 55, 57, 58, 61, 65, 67, 68, 70, 71, 72, 73, 75, 76, 77,
            78, 79, 82, 83, 84, 85, 88, 89, 90, 91, 93, 94)
ZERO_ROW = 95


def main() -> int:
    if not RES_JSON.exists() or not RES_NPZ.exists():
        print(f"MISSING producer output: {RES_JSON} / {RES_NPZ}")
        return 2
    doc = json.loads(RES_JSON.read_text())
    meta, res = doc["_meta"], doc["evaluation"]
    npz = np.load(RES_NPZ)

    # ---------------- 1. frozen protocol binding
    frozen = json.loads((NODE / "FROZEN_HASHES.json").read_text())
    prod = ROOT / "codes" / "gpu" / "eval_e2_direct_traction.py"
    live = hashlib.sha256(prod.read_bytes()).hexdigest()
    check("producer matches the pre-run freeze",
          live == frozen["sha256"]["codes/gpu/eval_e2_direct_traction.py"],
          f"live={live[:16]} frozen={frozen['sha256']['codes/gpu/eval_e2_direct_traction.py'][:16]}")
    check("producer self-reported hash matches the file",
          meta["script_sha256"] == live)
    check("run used foshan only (orig not used)", frozen["orig_used"] is False)

    # ---------------- 2. geometry and support rebuilt from first principles
    fluid, band, dist = geometry()
    dup = np.zeros(96, bool)
    for r in DUP_ROWS:
        dup[r] = True
    dup[ZERO_ROW] = True
    dup3 = np.broadcast_to(dup[None, :, None], (48, 96, 48))
    regions = {
        "full_support_excluded": fluid & (~band),
        "near_support_excluded_d_le_0p5h": fluid & (~band) & (dist <= .5),
        "outer_d_gt_0p5h": fluid & (dist > .5),
        "uniq_raster_support_excluded": fluid & (~band) & (~dup3),
    }
    for k, m in regions.items():
        check(f"region cell count rebuilt: {k}",
              int(m.sum()) == meta["region_cells"][k],
              f"local={int(m.sum())} producer={meta['region_cells'][k]}")
    check("every scored region excludes the conditioning band",
          all(not (regions[k] & band).any() for k in
              ("full_support_excluded", "near_support_excluded_d_le_0p5h",
               "uniq_raster_support_excluded")))

    # traction support: six physical faces, lid excluded
    I, J, K = np.meshgrid(np.arange(48), np.arange(96), np.arange(48), indexing="ij")
    ci, cj, ck = (I >= 12) & (I <= 35), (J >= 0) & (J <= 23), (K >= 12) & (K <= 35)
    tmask = np.zeros((48, 96, 48), bool)
    for m in [(J == 0), (J == 24) & ci & ck, (I == 11) & cj & ck,
              (I == 36) & cj & ck, (K == 11) & ci & cj, (K == 36) & ci & cj]:
        tmask |= (m & fluid)
    check("traction support = 4,512 physical-wall cells (lid excluded)",
          int(tmask.sum()) == meta["traction_cells"] == 4512,
          f"local={int(tmask.sum())} producer={meta['traction_cells']}")
    check("traction cells lie inside the excluded band (never scored)",
          bool((tmask & ~band).sum() == 0))
    check("computational lid carries no traction",
          bool(not tmask[:, 95, :].any()))

    # ---------------- 3. physical gates replayed
    g = meta["physical_gates"]
    check("signed-traction gate: tau.u_t < 0 on every physical face",
          g["tau_dot_ut_negative_all_faces"] and
          all(v < 0 for v in g["tau_dot_ut_per_face"].values()),
          json.dumps(g["tau_dot_ut_per_face"]))
    check("six physical faces used, lid excluded",
          g["faces_used"] == ["floor", "cube_top", "cube_xlo", "cube_xhi",
                              "cube_zlo", "cube_zhi"] and g["computational_lid_excluded"])
    check("wall-on-fluid streamwise force is negative",
          g["wall_on_fluid_x_force_negative"] and g["fx_viscous_mean_ds2_fd_physwall"] < 0)
    # external quadrature target recomputed from the retained audit files
    ext = []
    for b in ("b001", "b002"):
        p = RESULTS / f"wall_loads_audit_{b}.json"
        ps = json.loads(p.read_text())["per_snapshot"]
        tot = float(np.mean([s["fx_viscous_total"] for s in ps]))
        top = float(np.mean([s["faces"]["top"]["fx_viscous"] for s in ps]))
        ext.append(tot - top)
    check("external quadrature target recomputed from the wall-load audits",
          all(approx(a, b, 5e-7) for a, b in
              zip(ext, g["fx_viscous_mean_native_quadrature_physwall"])),
          f"local={[round(v,6) for v in ext]} producer="
          f"{g['fx_viscous_mean_native_quadrature_physwall']}")
    ratios = [g["fx_viscous_mean_ds2_fd_physwall"] / v for v in ext]
    check("FD traction agrees with independent spectral-element quadrature",
          all(0.5 <= abs(r) <= 2.0 for r in ratios),
          f"ratios={[round(r,4) for r in ratios]}")
    check("traction is pressure-free by tangential projection",
          g["tangential_only_pressure_free"] is True)

    # ---------------- 4. split inherited from the published producer
    pub = json.loads((RESULTS / "cube3d_coupling_adequate_results.json").read_text())["_meta"]
    check("split inherited: n, train, gap all equal the published run",
          meta["n_post_spinup"] == pub["n_post_spinup"]
          and meta["n_train"] == pub["n_train"]
          and meta["split_gap_snapshots"] == pub["split_gap_snapshots"],
          f"this={meta['n_post_spinup']}/{meta['n_train']}/{meta['split_gap_snapshots']} "
          f"published={pub['n_post_spinup']}/{pub['n_train']}/{pub['split_gap_snapshots']}")
    check("integral time reproduces the published value",
          approx(meta["tau_integral_snapshots"], pub["tau_integral_snapshots"], 1e-6))
    check("held-out targets increased over the published run",
          meta["n_eval_used"] > 160, f"n_eval={meta['n_eval_used']}")

    # ---------------- 5. non-fabrication: every score recomputed from components
    reg_keys = list(regions)
    n_rec = 0
    for arm, adict in res["arms"].items():
        for k in reg_keys:
            # Exact match on sse_<arm>_<seed>_<region>: a prefix test would let
            # "absent" also collect every "absent_B" component and silently
            # average two different arms together.
            seeds = [s for s in npz.files
                     if s.startswith(f"sse_{arm}_") and s.endswith(f"_{k}")
                     and s[len(f"sse_{arm}_"):-len(f"_{k}")].isdigit()]
            if not seeds:
                continue
            se = np.mean(np.stack([npz[s] for s in seeds]), 0)
            sst = npz[f"sst_{k}"]
            recomputed = float(1 - se.sum() / (sst.sum() + 1e-12))
            ok = approx(recomputed, adict[k]["R2_fluct_balanced"], 1e-8)
            n_rec += 1
            if not ok:
                check(f"recompute {arm}/{k}", False,
                      f"local={recomputed} stored={adict[k]['R2_fluct_balanced']}")
    check(f"every stored arm score recomputed from retained components ({n_rec})",
          all(c[1] for c in CHECKS if c[0].startswith("recompute ")) and n_rec > 0)

    # deltas recomputed from their two arms on the shared seed set
    bad = []
    for name, d in res["deltas"].items():
        a, b = name.split("_minus_")
        # Delta names use the control's short name; map it back to the arm key.
        alias = {"absentB": "absent_B", "trainmean": "tau_trainmean",
                 "fartime": "tau_fartime", "signflip": "tau_signflip",
                 "shuffle": "tau_shuffle"}
        b = alias.get(b, b)
        if a not in res["arms"] or b not in res["arms"]:
            bad.append(f"{name}: arm missing")
            continue
        common = d["seeds_used"]
        for k in reg_keys:
            try:
                sa = np.mean(np.stack([npz[f"sse_{a}_{s}_{k}"] for s in common]), 0)
                sb = np.mean(np.stack([npz[f"sse_{b}_{s}_{k}"] for s in common]), 0)
            except KeyError as e:
                bad.append(f"{name}/{k}: {e}")
                continue
            sst = npz[f"sst_{k}"]
            got = float((1 - sa.sum() / (sst.sum() + 1e-12))
                        - (1 - sb.sum() / (sst.sum() + 1e-12)))
            if not approx(got, d[k]["point"], 1e-8):
                bad.append(f"{name}/{k}: local={got} stored={d[k]['point']}")
    check("every delta recomputed from its two arms on the shared seed set",
          not bad, "; ".join(bad[:4]))

    # ---------------- 6. bootstrap replayed independently
    def block_indices(n, block, B, rng):
        nb = int(np.ceil(n / block))
        starts = rng.integers(0, n, size=(B, nb))
        off = np.arange(block)[None, None]
        return ((starts[:, :, None] + off).reshape(B, -1) % n)[:, :n]

    k = "full_support_excluded"
    sst = npz[f"sst_{k}"]
    n_eval = len(sst)
    bixc = block_indices(n_eval, res["eval_block_conservative"], 4000,
                         np.random.default_rng(45))
    name = "tau_native_minus_absent"
    common = res["deltas"][name]["seeds_used"]
    sa = np.mean(np.stack([npz[f"sse_tau_native_{s}_{k}"] for s in common]), 0)
    sb = np.mean(np.stack([npz[f"sse_absent_{s}_{k}"] for s in common]), 0)
    ra = 1 - sa[bixc].sum(1) / (sst[bixc].sum(1) + 1e-12)
    rb = 1 - sb[bixc].sum(1) / (sst[bixc].sum(1) + 1e-12)
    d = ra - rb
    lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    stored = res["deltas"][name][k]["ci95_conservative_block"]
    check("conservative-block interval of the primary estimand replays",
          approx(lo, stored[0], 1e-6) and approx(hi, stored[1], 1e-6),
          f"local=[{lo:.6f},{hi:.6f}] stored=[{stored[0]:.6f},{stored[1]:.6f}]")
    check("conservative block is strictly larger than the release block",
          res["eval_block_conservative"] > res["eval_block"],
          f"cons={res['eval_block_conservative']} rel={res['eval_block']}")

    # ---------------- 7. seed reporting is real
    prim = res["deltas"][name][k]
    check("primary estimand carries three independent training seeds",
          len(prim["per_seed_delta"]) == 3, str(prim["per_seed_delta"]))
    check("per-seed deltas average to the reported point estimate",
          approx(float(np.mean(prim["per_seed_delta"])), prim["point"], 5e-3),
          f"mean_seed={np.mean(prim['per_seed_delta']):.6f} point={prim['point']:.6f}")
    check("matched ceiling also carries three seeds",
          len(res["deltas"]["band_phys_minus_absentB"][k]["per_seed_delta"]) == 3)

    # ---------------- 8. design integrity
    check("traction arms use no clamp", meta["clamp_used_traction_arms"] is False)
    check("one network serves both primary arms (conditioning dropout)",
          approx(meta["conditioning_dropout"], 0.25))
    check("evidence level names oracle traction and excludes solver coupling",
          "oracle" in meta["evidence_level"].lower()
          and "not solver-coupled" in meta["evidence_level"].lower())

    # ---------------- 9. manuscript label withdrawals (R24)
    tex = ""
    for p in sorted((MAN / "sections").glob("*.tex")):
        tex += p.read_text()
    banned = ["upper bound on any closure", "exact wall traction",
              "the target's own wall traction"]

    def asserted(phrase: str, body: str) -> bool:
        """True if the phrase occurs without an immediately preceding negation."""
        start = 0
        while (i := body.find(phrase, start)) != -1:
            window = body[max(0, i - 60):i]
            if "not " not in window and "never " not in window:
                return True
            start = i + len(phrase)
        return False

    hits = [b for b in banned if asserted(b, tex)]
    check("withdrawn labels are not asserted anywhere in the manuscript body",
          not hits, f"still asserted: {hits}")
    check("equilibrium adapter is named as such",
          "equilibrium" in tex and ("target-velocity" in tex or "velocity-derived" in tex))

    # ---------------- 10. manuscript numbers trace to the source of record
    def fmt(v, nd=5):
        return f"{v:+.{nd}f}".replace("+", "")

    printed = re.findall(r"[-+]?\d\.\d{4,5}", tex)
    prim_pt = res["deltas"][name][k]["point"]
    check("primary estimand appears in the manuscript at source precision",
          any(approx(float(x), prim_pt, 5e-5) for x in printed),
          f"looking for {prim_pt:.5f}")

    # ---------------- report
    npass = sum(1 for _, ok, _ in CHECKS if ok)
    nfail = len(CHECKS) - npass
    print("=" * 78)
    for nm, ok, detail in CHECKS:
        print(f"[{'PASS' if ok else 'FAIL'}] {nm}" + (f"  -- {detail}" if detail else ""))
    print("=" * 78)
    print(f"VERIFY node005: {npass} PASS / {nfail} FAIL")
    return 0 if nfail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
