#!/usr/bin/env python3
"""Stage the derived arrays behind Figure 1 (overview) into a single npz.

Every array drawn by codes/figures/fig1_architecture.py that is not a direct
slice of an already-staged file is computed here, deterministically, from the
staged records, and written to manuscript/source_data/fig1_v2/ together with a
provenance ledger naming each upstream file and its SHA-256.
"""

import hashlib
import json
import pathlib

import numpy as np

PROJECT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = PROJECT / "manuscript/source_data"
OUT_DIR = SOURCE / "fig1_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

UPSTREAM = {
    "rep": SOURCE / "fig2/cube3d_representative_fields.npz",
    "pair": SOURCE / "fig2/cube3d_time_pair.npz",
    "channel": SOURCE / "fig_slot/channel_context_frame.npz",
    "hill": SOURCE / "fig11_reachgated/hill_context_frame.npz",
    "e2cube": SOURCE / "fig11_reachgated/e2_cube_reachgated_components.npz",
}

rep = np.load(UPSTREAM["rep"])
tp = np.load(UPSTREAM["pair"])
ch = np.load(UPSTREAM["channel"])
hill = np.load(UPSTREAM["hill"])
e2c = np.load(UPSTREAM["e2cube"])

# ------------------------------------------------------------- cube planes
u = np.asarray(rep["truth"], np.float64)[0]
uc = np.asarray(rep["correct"], np.float64)[0]
fluid = np.asarray(rep["fluid"], bool)
counts = fluid.sum(axis=(0, 2)).clip(min=1)
plane_mean = (u * fluid).sum(axis=(0, 2)) / counts
up = u - plane_mean[None, :, None]
ucp = uc - plane_mean[None, :, None]
vlim = float(np.percentile(np.abs(up[fluid]), 99.0))

mask_h = np.zeros((48, 48), bool)
mask_h[12:36, 12:36] = True
mask_v = np.zeros((48, 48), bool)
mask_v[0:24, 12:36] = True

plane_h = up[:, 4, :].T.copy()
plane_v = up[:, :48, 24].T.copy()
out3d_v = ucp[:, :48, 24].T.copy()
band_plane = up[:, 2, :].T.copy()
out30 = ucp[:, 30, :].T.copy()
tru30 = up[:, 30, :].T.copy()
gen_plane = ucp[:, 40, :].T.copy()
v_gen = float(np.percentile(np.abs(ucp[:, 40, :]), 99.0))

# --------------------------------------------------- traction (floor shear)
tau_plane = np.asarray(rep["truth"], np.float64)[0][:, 1, :].T.copy()
tau_vlim = float(np.percentile(np.abs(tau_plane[~mask_h]), 99.0))
tau_wi = np.asarray(tp["donor"], np.float64)[0][:, 1, :].T.copy()
rng_sh = np.random.default_rng(3)
tau_sh = tau_plane.copy()
tau_sh[~mask_h] = rng_sh.permutation(tau_plane[~mask_h])


def sm2(f, n=6):
    for _ in range(n):
        for axx in (0, 1):
            g = np.moveaxis(np.pad(f, [(2, 2) if a == axx else (0, 0)
                                       for a in range(2)], mode="edge"),
                            axx, 0)
            f = np.moveaxis((g[:-4] + 4 * g[1:-3] + 6 * g[2:-2]
                             + 4 * g[3:-1] + g[4:]) / 16.0, 0, axx)
    return f


tau_eq = sm2(tau_plane.copy())

# ------------------------------------------------ E2 closure-vs-record floor
floor_ok = np.ones((48, 48), bool)
floor_ok[12:36, 12:36] = False
tau_true_f = np.full((48, 48), np.nan)
tau_true_f[floor_ok] = np.asarray(e2c["tau_native_test"],
                                  np.float64)[0, 0][:1728]
tau_clos_f = np.full((48, 48), np.nan)
tau_clos_f[floor_ok] = np.asarray(e2c["tau_closure_test"],
                                  np.float64)[0, 0][:1728]
vlim_tau_f = float(np.nanpercentile(np.abs(tau_true_f), 99.0))

# --------------------------------------------------------------- channel
prof = np.asarray(ch["u_mean_profile"], np.float64)
up_ch = np.asarray(ch["u_plane"], np.float64) - prof[:, None]
vlim_ch = float(np.percentile(np.abs(up_ch), 99.0))
profile_pn = (prof[:100] / prof[:100].max()).copy()
profile_yn = np.linspace(0.0, 1.0, 100)

# ------------------------------------------------------------------ hill
solidH = np.asarray(hill["mask"], np.float64) < 0.5
u_stdH = np.asarray(hill["u_std"], np.float64).copy()
vlim_hill = float(np.percentile(np.abs(u_stdH[~solidH]), 99.0))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

XXn, YYn = np.meshgrid(np.linspace(0.0, 1.0, u_stdH.shape[1]),
                       np.linspace(0.0, 1.0, u_stdH.shape[0]))
tmp_fig, tmp_ax = plt.subplots()
cs = tmp_ax.contour(XXn, YYn, solidH.astype(float), levels=[0.5])
segs = [seg for lev in cs.allsegs for seg in lev if len(seg) >= 4]
plt.close(tmp_fig)
parts = []
for seg in segs:
    parts.append(np.asarray(seg, np.float64))
    parts.append(np.full((1, 2), np.nan))
hill_boundary_xy = np.vstack(parts[:-1])

# ------------------------------------------------- 3-D streamlines (panel a)
DX, DY = 0.55, 0.34
BT = 0.045
zc, dz = 0.30, 0.35
V3 = np.mean([np.asarray(f, np.float64)
              for f in (rep["truth"], tp["target"], tp["donor"])], axis=0)
V3[:, ~fluid] = 0.0
counts3 = fluid.sum(axis=(0, 2)).clip(min=1)
ub_prof = (V3[0] * fluid).sum(axis=(0, 2)) / counts3


def sm3(f):
    for ax3 in (0, 1, 2):
        g = np.moveaxis(np.pad(f, [(2, 2) if a == ax3 else (0, 0)
                                   for a in range(3)], mode="edge"), ax3, 0)
        f = np.moveaxis((g[:-4] + 4 * g[1:-3] + 6 * g[2:-2] + 4 * g[3:-1]
                         + g[4:]) / 16.0, 0, ax3)
    return f


Vs = np.stack([sm3(sm3(sm3(V3[c]))) for c in range(3)])


def tri(f, xi, yi, zi):
    x0 = int(np.clip(np.floor(xi), 0, 46)); fx = np.clip(xi - x0, 0, 1)
    y0 = int(np.clip(np.floor(yi), 0, 94)); fy = np.clip(yi - y0, 0, 1)
    z0 = int(np.clip(np.floor(zi), 0, 46)); fz = np.clip(zi - z0, 0, 1)
    c = f[x0:x0 + 2, y0:y0 + 2, z0:z0 + 2]
    c = c[0] * (1 - fx) + c[1] * fx
    c = c[0] * (1 - fy) + c[1] * fy
    return c[0] * (1 - fz) + c[1] * fz


B_FLOOR = 0.35 * float(np.median(ub_prof[8:48]))
CUBE_LO = np.array([0.5 - BT, 0.0, zc - BT])
CUBE_HI = np.array([1.5 + BT, 1.0 + BT, zc + dz + BT])
CUBE_MID = 0.5 * (CUBE_LO + CUBE_HI)


def adv3(p):
    xi, yi, zi = p[0] * 24.0, p[1] * 24.0, p[2] * 47.0
    base = max(float(np.interp(yi, np.arange(96.0), ub_prof)), B_FLOOR) / 24.0
    F = np.array([tri(Vs[0], xi, yi, zi) / 24.0 - base,
                  tri(Vs[1], xi, yi, zi) / 24.0,
                  tri(Vs[2], xi, yi, zi) / 47.0])
    nF = np.linalg.norm(F); cap = 0.55 * base
    if nF > cap:
        F *= cap / nF
    a = np.array([base, 0.0, 0.0]) + F
    c = np.clip(p, CUBE_LO, CUBE_HI); d = p - c; dn = float(np.linalg.norm(d))
    M = 0.13
    if dn < 1e-9:
        below = p - CUBE_LO; above = CUBE_HI - p
        j = int(np.argmin(np.minimum(below, above)))
        n = np.zeros(3); n[j] = -1.0 if below[j] < above[j] else 1.0
        a = n * base
    elif dn < M:
        n = d / dn
        w = float(np.clip((M - dn) / (M - 0.04), 0.0, 1.0))
        vn = float(a @ n)
        if vn < 0.0:
            a = a - w * vn * n
        t = c - CUBE_MID; t[int(np.argmax(np.abs(n)))] = 0.0
        t[1] = max(t[1], 0.0)
        nt = float(np.linalg.norm(t))
        if nt > 1e-9:
            a = a + np.array([0.30, 0.30, 0.60]) * base * w * (t / nt)
    if p[1] < 0.06:
        a[1] -= min(a[1], 0.0) * (1 - p[1] / 0.06)
    if p[2] < 0.04:
        a[2] -= min(a[2], 0.0) * (1 - p[2] / 0.04)
    if p[2] > 0.96:
        a[2] -= max(a[2], 0.0) * (1 - (1 - p[2]) / 0.04)
    return a


front_parts, back_parts, arrow_rows = [], [], []
arw = 0
for ys in (0.17, 0.34, 0.54, 0.78, 1.02, 1.30, 1.62):
    for zs in (0.08, 0.24, 0.40, 0.56, 0.72, 0.88):
        pnt = np.array([0.02, ys, zs]); pts3 = [pnt.copy()]
        for _ in range(650):
            a1 = adv3(pnt); n1 = np.linalg.norm(a1)
            if n1 < 1e-10:
                break
            a2 = adv3(pnt + 0.009 * a1 / n1); n2 = np.linalg.norm(a2)
            if n2 < 1e-10:
                break
            pnt = pnt + 0.018 * (a1 / n1 + a2 / n2) / 2.0
            if np.all(pnt > CUBE_LO) and np.all(pnt < CUBE_HI):
                below = pnt - CUBE_LO; above = CUBE_HI - pnt
                j = int(np.argmin(np.minimum(below, above)))
                if below[j] < above[j]:
                    pnt[j] = CUBE_LO[j] - 0.005
                else:
                    pnt[j] = CUBE_HI[j] + 0.005
            pnt[1] = max(pnt[1], 0.01)
            pnt[2] = float(np.clip(pnt[2], 0.005, 0.995))
            pts3.append(pnt.copy())
            if pnt[0] > 1.985 or pnt[1] > 1.96:
                break
        if len(pts3) < 20:
            continue
        P = np.asarray(pts3)
        for _ in range(2):
            P[1:-1] = 0.5 * P[1:-1] + 0.25 * (P[:-2] + P[2:])
        sx = P[:, 0] + DX * P[:, 2]
        sy = P[:, 1] + DY * P[:, 2]
        xy = np.column_stack([sx, sy])
        front = zs < 0.475 or ys > 0.95
        (front_parts if front else back_parts).append(xy)
        (front_parts if front else back_parts).append(
            np.full((1, 2), np.nan))
        if front and ys <= 1.35:
            st = (0.30, 0.85, 1.38, 1.78)[arw % 4]; arw += 1
            j = int(np.argmin(np.abs(P[:, 0] - st)))
            if 1 <= j < len(P) - 4:
                arrow_rows.append([sx[j], sy[j], sx[j + 3], sy[j + 3]])

stream_front_xy = np.vstack(front_parts[:-1])
stream_back_xy = np.vstack(back_parts[:-1])
stream_arrows = np.asarray(arrow_rows, np.float64)

# -------------------------------------------------------------------- write
payload = {
    "plane_h": plane_h, "plane_v": plane_v, "mask_h": mask_h,
    "mask_v": mask_v, "out3d_v": out3d_v, "band_plane": band_plane,
    "out30": out30, "tru30": tru30, "gen_plane": gen_plane,
    "tau_plane": tau_plane, "tau_wi": tau_wi, "tau_sh": tau_sh,
    "tau_eq": tau_eq, "tau_true_f": tau_true_f, "tau_clos_f": tau_clos_f,
    "up_ch": up_ch, "profile_pn": profile_pn, "profile_yn": profile_yn,
    "u_stdH": u_stdH, "solidH": solidH,
    "hill_boundary_xy": hill_boundary_xy,
    "stream_front_xy": stream_front_xy, "stream_back_xy": stream_back_xy,
    "stream_arrows": stream_arrows,
    "vlim": np.float64(vlim), "tau_vlim": np.float64(tau_vlim),
    "v_gen": np.float64(v_gen), "vlim_tau_f": np.float64(vlim_tau_f),
    "vlim_ch": np.float64(vlim_ch), "vlim_hill": np.float64(vlim_hill),
}
out_npz = OUT_DIR / "fig1_overview_derived.npz"
np.savez_compressed(out_npz, **payload)


def sha(path):
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


provenance = {
    "produces": out_npz.name,
    "produced_sha256": sha(out_npz),
    "upstream": {k: {"path": str(p.relative_to(PROJECT)), "sha256": sha(p)}
                 for k, p in UPSTREAM.items()},
    "recipes": {
        "planes": "truth/correct u-component minus per-height fluid-mean "
                  "profile of the truth frame; transposed index planes at "
                  "y-index 4 (matching band), 2 (in-band), 30 (above cube), "
                  "40 (generator icon) and mid-span x-index 24 cropped to "
                  "the lowest 48 heights.",
        "tau_plane": "signed near-wall streamwise shear proxy: truth u at "
                     "y-index 1, cube footprint 12:36 masked.",
        "tau_variants": "wrong-instant = same plane from the donor frame of "
                        "cube3d_time_pair; shuffled = fixed-seed (rng 3) "
                        "permutation of the off-footprint values; "
                        "equilibrium = six passes of a separable 5-point "
                        "binomial smoother.",
        "tau_floor_maps": "first 1,728 entries (C-order over the floor grid "
                          "with the 12:36 footprint removed) of "
                          "tau_native_test / tau_closure_test frame 0, "
                          "component 0.",
        "streamlines": "unit-speed Heun tracer on the three-snapshot mean "
                       "field (truth + time-pair target/donor), triple "
                       "5-point binomial smoothing, forward drift = "
                       "height-mean profile floored at 0.35 median, "
                       "fluctuations capped at 0.55 base, no-penetration "
                       "slip on the drawn cube inflated by the band shell, "
                       "42 seeds on a 7x6 (y,z) grid, oblique projection "
                       "x+0.55z / y+0.34z, split front/back at seed depth "
                       "0.475 (or height 0.95).",
        "hill_boundary": "matplotlib 0.5-level contour of the hill solid "
                         "mask on the unit square.",
        "channel": "u_plane minus u_mean_profile per wall-normal index; "
                   "profile icon = lower 100 points normalised by their "
                   "maximum.",
    },
}
(OUT_DIR / "fig1_overview_derived_provenance.json").write_text(
    json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("staged", out_npz)
