#!/usr/bin/env python3
"""
wall_closure.py -- public, inference-only adapter for the frozen familyclass wall closure.

EVIDENCE BOUNDARY
-----------------
This module implements the closure forward map and two explicit conditioning seams.  It does not
turn an offline reconstruction into a solver-coupled wall model:

* ``predict_tau_w`` emits signed wall stress from four local outer-flow features and the fixed
  branch embedding contained in the selected header.
* ``reconstruct_u`` is the optional Reichardt profile bridge used by the older velocity-band runs.
* ``build_direct_tau_condition`` writes predicted wall stress into the direct-tau conditioning
  tensor used by ``eval_wallstress_cond.py``; this is feature conditioning, not a Navier--Stokes
  momentum boundary condition.
* ``extract_resolved_features`` obtains candidate local features from one resolved coarse/LES
  snapshot. It does not reconstruct the family/set branch input used in calibration, and its
  pressure-gradient and displacement-thickness definitions differ from the frozen exporter's
  definitions. It is therefore not, by itself, a validated runtime seam. The historical
  ``case*_closure_inputs.json`` files were extracted from DNS means and are retained only as
  clearly labelled a-priori diagnostics.

Only a flow solver that consumes the predicted ``tau_w`` during time evolution would constitute a
solver-coupled/a-posteriori wall model; no such solver loop is implemented in this repository.

WHAT THIS IS (and is NOT):
  * The physics-grounded closure C is REUSED verbatim from the familyclass champion
    (`familyclass_wall_model/codes/x2fam_weights.h`, md5 782b91ed..., the committed deploy weights,
    member = beyond-training Krank periodic hill at Re=10595). We do NOT retrain it -- INFERENCE
    ONLY. It emits the signed
    skin friction  Cf = 2 tau_w / U_e^2  from four wall-stress-FREE outer features
        f = [ pi1=U_m/U_e , pi2=y_m/delta99 , Pi_p=(dp/dx) delta99/U_e^2 , Lambda=delta* (dp/dx)/U_e^2 ]
    (deployed form: trunk MLP 4->64->64->16 with erf-GELU, dotted with the frozen member branch
    embedding X2_EMB; node004_x2_export.py:210-218). tau_w = 0.5 Cf U_e^2 (rho=1, kinematic).
  * The near-wall band u(y) is then reconstructed by the STANDARD equilibrium wall-model (Reichardt)
    profile from the predicted tau_w ONLY -- DNS-FREE (no DNS profile, no DNS matching height):
        u_tau = sqrt(|tau_w|),   y+ = y u_tau / nu,   u(y) = sign(tau_w) u_tau * reichardt(y+)
    (round010_eqwm_baseline_regime.py:56-68). This is the make-or-break DNS-free guard (L0 skeptic
    watch-item W1): the reconstruction uses ONLY predicted tau_w + nu(=1/Re) + a generic profile.

The header is parsed at runtime so the bytes are traceable; no network parameters are copied by hand.
"""
import os, re, hashlib
import numpy as np

try:
    from scipy.special import erf as _erf
except Exception:                                   # numpy fallback (vectorized erf via math)
    import math
    _erf = np.vectorize(math.erf)

HERE = os.path.dirname(os.path.abspath(__file__))
# Prefer the co-located committed copy (syncs to the GPU pod); fall back to the familyclass source.
_LOCAL_HEADER = os.path.join(HERE, "x2fam_weights.h")
_SRC_HEADER = os.path.abspath(os.path.join(HERE, "..", "..", "..", "familyclass_wall_model", "codes", "x2fam_weights.h"))
DEFAULT_HEADER = _LOCAL_HEADER if os.path.exists(_LOCAL_HEADER) else _SRC_HEADER

KAPPA, C_REICH = 0.41, 7.8                          # round010_eqwm_baseline_regime.py:56-57

# --------------------------------------------------------------------- header parser (byte-traceable)
def parse_header(path=None):
    """Parse the familyclass DeepONet C header into a dict of numpy arrays / scalars.
    Recognises `#define X2_NAME VAL` and `static const double X2_NAME[N] = {...};`."""
    path = path or DEFAULT_HEADER
    txt = open(path).read()
    W = {}
    for name, val in re.findall(r"#define\s+(X2_\w+)\s+([-\d.eE+]+)", txt):
        W[name] = float(val) if any(c in val for c in ".eE") else int(val)
    for name, val in re.findall(r"static const double\s+(X2_\w+)\s*=\s*([-\d.eE+]+)\s*;", txt):
        W[name] = float(val)                              # scalar `static const double X2_MY = ...;`
    for name, body in re.findall(r"static const double\s+(X2_\w+)\s*\[\s*\d+\s*\]\s*=\s*\{([^}]*)\}", txt):
        W[name] = np.array([float(v) for v in body.replace("\n", " ").split(",") if v.strip()], dtype=np.float64)
    W["_path"] = path
    W["_md5"] = hashlib.md5(open(path, "rb").read()).hexdigest()
    return W

def frozen_weight_provenance(path=None):
    """Return release-safe frozen-weight provenance (hashes + basename, never a machine path)."""
    path = path or DEFAULT_HEADER
    raw = open(path, "rb").read()
    W = _shape_weights(parse_header(path))
    return {
        "artifact": os.path.basename(path),
        "md5": hashlib.md5(raw).hexdigest(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "bytes": len(raw),
        "architecture": f"DeepONet trunk {W['NL']}->{W['H']}->{W['H']}->{W['P']} + frozen member embedding",
        "output": "signed Cf; tau_w = 0.5 * Cf * Ue^2 (rho=1)",
    }

def _shape_weights(W):
    """Reshape the flat header arrays into (out,in) matrices (row-major C indexing = torch weight)."""
    NL, H, P = int(W["X2_NLOC"]), int(W["X2_H"]), int(W["X2_P"])
    return dict(
        mu=W["X2_MU_L"], sd=W["X2_SD_L"], my=float(W["X2_MY"]), sy=float(W["X2_SY"]), b0=float(W["X2_B0"]),
        emb=W["X2_EMB"],
        W1=W["X2_W1"].reshape(H, NL), b1=W["X2_B1"],
        W2=W["X2_W2"].reshape(H, H),  b2=W["X2_B2"],
        W3=W["X2_W3"].reshape(P, H),  b3=W["X2_B3"],
        NL=NL, H=H, P=P)

def _gelu(v):
    return 0.5 * v * (1.0 + _erf(v / np.sqrt(2.0)))

# --------------------------------------------------------------------- closure forward: predict Cf / tau_w
def predict_Cf(feats, W=None):
    """feats: (n,4) array of [pi1, pi2, Pi_p, Lambda] (LOCAL_COLS order). Returns signed Cf (n,).
    Exact transcription of the deployed forward (node004_x2_export.py:210-218 / gen_x2fam_nut.py:64-90)."""
    W = W if isinstance(W, dict) and "W1" in W else _shape_weights(parse_header(W))
    x = (np.atleast_2d(np.asarray(feats, dtype=np.float64)) - W["mu"]) / W["sd"]     # (n,4) standardize
    h1 = _gelu(x @ W["W1"].T + W["b1"])                                              # (n,64)
    h2 = _gelu(h1 @ W["W2"].T + W["b2"])                                             # (n,64)
    t = h2 @ W["W3"].T + W["b3"]                                                     # (n,16)
    cf = (t @ W["emb"] + W["b0"]) * W["sy"] + W["my"]                                # (n,)
    return cf

def predict_tau_w(feats, Ue, W=None):
    """tau_w = 0.5 Cf U_e^2  (rho=1, kinematic). Ue: (n,) edge velocity per station."""
    return 0.5 * predict_Cf(feats, W) * np.asarray(Ue, dtype=np.float64) ** 2

# --------------------------------------------------------------------- DNS-free wall-model reconstruction
def reichardt_uplus(yplus):
    yplus = np.asarray(yplus, dtype=np.float64)
    return (np.log1p(KAPPA * yplus) / KAPPA
            + C_REICH * (1.0 - np.exp(-yplus / 11.0) - (yplus / 11.0) * np.exp(-yplus / 3.0)))

def reconstruct_u(tau_w, y, nu):
    """DNS-FREE: near-wall streamwise velocity from predicted tau_w ONLY.
    tau_w: scalar or (n,); y: wall-normal distances (m,); nu=1/Re. Returns u with sign of tau_w.
    Broadcasts to (n,m) if tau_w is (n,)."""
    tau_w = np.asarray(tau_w, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    u_tau = np.sqrt(np.abs(tau_w))
    if tau_w.ndim == 0:
        yplus = y * u_tau / nu
        return np.sign(tau_w) * u_tau * reichardt_uplus(yplus)
    yp = (y[None, :] * u_tau[:, None]) / nu                                          # (n,m)
    return np.sign(tau_w)[:, None] * u_tau[:, None] * reichardt_uplus(yp)

# --------------------------------------------------------------------- leakage-free runtime input seam
def _smooth_1d(values, width):
    """Small centred spatial stencil; uses only the current resolved snapshot."""
    values = np.asarray(values, dtype=np.float64)
    width = max(1, int(width))
    if width == 1 or values.size < 3:
        return values.copy()
    if width % 2 == 0:
        width += 1
    pad = width // 2
    return np.convolve(np.pad(values, pad, mode="edge"), np.ones(width) / width, mode="valid")

def extract_resolved_features(u_resolved, x, y, wall_row=None, mask=None, *,
                              matching_frac=0.10, x_stride=2, y_stride=2,
                              edge_smooth=5):
    """Extract closure inputs from ONE resolved coarse/LES streamwise-velocity snapshot.

    Parameters
    ----------
    u_resolved : (H,W) array
        Physical streamwise velocity available to a coarse solver at the current time.  A time or
        ensemble mean is deliberately rejected (the API accepts exactly two dimensions).
    x, y : (W,), (H,) arrays
        Resolved cell-centre coordinates.
    wall_row : (W,) integer array, optional
        First fluid row above the lower wall.  Defaults to zero for a flat wall.
    mask : (H,W) bool array, optional
        Resolved fluid mask.  Profiles stop at the first non-fluid cell.
    matching_frac : float
        y_m/delta99 used by the frozen closure (0.10 is its deployed convention).
    x_stride, y_stride : int
        Explicit coarse-sampling strides.  No discarded value re-enters the feature calculation.

    Returns
    -------
    dict with ``features`` in [pi1, pi2, Pi_p, Lambda] order, ``Ue``, station indices and
    provenance declaring that no temporal aggregation was used.
    """
    u = np.asarray(u_resolved, dtype=np.float64)
    if u.ndim != 2:
        raise ValueError("u_resolved must be one (H,W) runtime snapshot; time/ensemble arrays are not accepted")
    H, W = u.shape
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    if x.shape != (W,) or y.shape != (H,):
        raise ValueError(f"coordinate shapes must be x=({W},), y=({H},); got {x.shape}, {y.shape}")
    wr = np.zeros(W, dtype=int) if wall_row is None else np.asarray(wall_row, dtype=int)
    fluid = np.ones((H, W), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if wr.shape != (W,) or fluid.shape != (H, W):
        raise ValueError("wall_row/mask shape mismatch")
    xs = np.arange(0, W, max(1, int(x_stride)), dtype=int)
    if xs[-1] != W - 1:
        xs = np.r_[xs, W - 1]
    dy0 = float(np.median(np.diff(y))) if H > 1 else 1.0

    rows = []
    for j in xs:
        r0 = int(wr[j])
        if r0 < 0 or r0 >= H or not fluid[r0, j]:
            continue
        r1 = r0
        while r1 < H and fluid[r1, j] and np.isfinite(u[r1, j]):
            r1 += 1
        rr = np.arange(r0, r1, max(1, int(y_stride)), dtype=int)
        if rr.size == 0 or rr[-1] != r1 - 1:
            rr = np.r_[rr, r1 - 1]
        if rr.size < 4:
            continue
        eta = y[rr] - y[r0] + 0.5 * dy0
        up = u[rr, j]
        ie = int(np.argmax(np.abs(up)))
        if ie < 1 or abs(up[ie]) < 1e-10:
            continue
        Ue_signed = float(up[ie]); Ue = abs(Ue_signed)
        branch_eta, branch_u = eta[:ie + 1], up[:ie + 1]
        ratio = branch_u / Ue_signed
        hit = np.flatnonzero(ratio >= 0.99)
        ih = int(hit[0]) if hit.size else ie
        if ih == 0:
            d99 = float(branch_eta[0])
        else:
            q0, q1 = ratio[ih - 1], ratio[ih]
            a = float(np.clip((0.99 - q0) / (q1 - q0 + 1e-12), 0.0, 1.0))
            d99 = float(branch_eta[ih - 1] + a * (branch_eta[ih] - branch_eta[ih - 1]))
        d99 = max(d99, 0.5 * dy0)
        keep = eta <= d99
        dstar = float(np.trapz(1.0 - up[keep] / Ue_signed, eta[keep])) if keep.sum() >= 2 else 0.0
        ym = float(matching_frac) * d99
        Um = float(np.interp(ym, eta, up))
        rows.append({"j": int(j), "x": float(x[j]), "Ue": Ue, "Ue_signed": Ue_signed,
                     "delta99": d99, "dstar": dstar, "Um": Um})
    if len(rows) < 3:
        raise ValueError("fewer than three valid resolved stations; cannot estimate dp/dx")

    xr = np.array([r["x"] for r in rows]); ue_signed = np.array([r["Ue_signed"] for r in rows])
    ue_signed = _smooth_1d(ue_signed, edge_smooth)
    dpdx = -ue_signed * np.gradient(ue_signed, xr)  # resolved edge-Euler estimate, rho=1
    features, Ue, delta99, Um, ym = [], [], [], [], []
    for r, pg in zip(rows, dpdx):
        ue2 = r["Ue"] ** 2 + 1e-12
        features.append([r["Um"] / (r["Ue_signed"] + 1e-12), float(matching_frac),
                         float(pg) * r["delta99"] / ue2, float(pg) * r["dstar"] / ue2])
        Ue.append(r["Ue"])
        delta99.append(r["delta99"]); Um.append(r["Um"]); ym.append(float(matching_frac) * r["delta99"])
    return {
        "features": np.asarray(features, dtype=np.float64),
        "Ue": np.asarray(Ue, dtype=np.float64),
        "delta99": np.asarray(delta99, dtype=np.float64),
        "Um": np.asarray(Um, dtype=np.float64),
        "ym": np.asarray(ym, dtype=np.float64),
        "station_index": np.asarray([r["j"] for r in rows], dtype=np.int64),
        "x": xr,
        "dpdx": np.asarray(dpdx, dtype=np.float64),
        "provenance": {
            "source": "single resolved coarse/LES snapshot",
            "temporal_aggregation": "none",
            "target_dns_ensemble_mean": False,
            "x_stride": int(x_stride), "y_stride": int(y_stride),
            "edge_smooth_stations": int(edge_smooth), "matching_frac": float(matching_frac),
        },
    }

def predict_tau_w_from_resolved(u_resolved, x, y, wall_row=None, mask=None, *, W=None, **extract_kw):
    """Candidate local-feature seam: resolved snapshot -> features -> frozen-header tau_w.

    The returned stress is interpolated across all streamwise columns.  This remains an offline
    diagnostic and reuses the header's fixed member embedding; it is not a source-general or
    calibration-consistent runtime closure until the branch and local-feature mismatches are
    resolved. A time-evolving solver would still be required for solver coupling.
    """
    ext = extract_resolved_features(u_resolved, x, y, wall_row, mask, **extract_kw)
    Wh = _shape_weights(parse_header(W)) if not (isinstance(W, dict) and "W1" in W) else W
    tau_station = predict_tau_w(ext["features"], ext["Ue"], Wh)
    tau = np.interp(np.asarray(x, dtype=np.float64), ext["x"], tau_station)
    return tau, {**ext, "tau_station": tau_station, "weights": frozen_weight_provenance()}

def predict_eqwm_tau(Um, ym, nu, *, max_iter=64):
    """Classical equilibrium-wall-model stress from a resolved matching-point velocity.

    Inverts the same Reichardt law used by ``reconstruct_u`` with a monotone bisection in friction
    velocity.  Inputs are runtime resolved quantities; this is an offline EQWM baseline unless the
    returned stress is actually applied inside a time-evolving solver.
    """
    Um = np.asarray(Um, dtype=np.float64); ym = np.asarray(ym, dtype=np.float64)
    if Um.shape != ym.shape or nu <= 0 or np.any(ym <= 0):
        raise ValueError("Um/ym must have equal shape, ym>0 and nu>0")
    target = np.abs(Um)
    lo = np.zeros_like(target)
    hi = np.maximum(np.sqrt(target * float(nu) / (ym + 1e-12)) * 4.0, np.sqrt(target + 1e-12))
    def speed(ut):
        return ut * reichardt_uplus(ym * ut / float(nu))
    for _ in range(20):
        grow = speed(hi) < target
        if not np.any(grow):
            break
        hi[grow] *= 2.0
    for _ in range(int(max_iter)):
        mid = 0.5 * (lo + hi)
        low = speed(mid) < target
        lo[low] = mid[low]; hi[~low] = mid[~low]
    ut = 0.5 * (lo + hi)
    return np.sign(Um) * ut ** 2

def build_direct_tau_condition(tau_w, tau_mu, tau_sd, wall_row, mask, band_h=1):
    """Build the exact 4-channel direct-tau tensor used by the frozen CondUNet.

    ``tau_mu`` and ``tau_sd`` must come from the generator's training metadata; deriving them from
    the target test ensemble would be leakage.  Output shape is ``(N,4,H,W)`` with channels
    [standardised tau value, support mask, observed fraction, domain mask].
    """
    tau = np.asarray(tau_w, dtype=np.float64)
    if tau.ndim == 1:
        tau = tau[None, :]
    domain = np.asarray(mask, dtype=bool); H, Wd = domain.shape
    wr = np.asarray(wall_row, dtype=int)
    if tau.shape[1] != Wd or wr.shape != (Wd,):
        raise ValueError("tau_w/wall_row width does not match mask")
    if not np.isfinite(tau_mu) or not np.isfinite(tau_sd) or tau_sd <= 0:
        raise ValueError("tau_mu/tau_sd must be finite training-set normalisation constants")
    rows = np.arange(H)[:, None]
    support = ((rows >= wr[None, :]) & (rows < (wr + int(band_h))[None, :]) & domain)
    z = (tau - float(tau_mu)) / float(tau_sd)
    tval = z[:, None, None, :] * support[None, None]
    keep = np.broadcast_to(support[None, None], tval.shape)
    frac = float(support.sum() / max(1, domain.sum()))
    obsfrac = np.full_like(tval, frac)
    dom = np.broadcast_to(domain[None, None], tval.shape)
    return np.concatenate([tval, keep.astype(float), obsfrac, dom.astype(float)], axis=1).astype(np.float32)

# --------------------------------------------------------------------- CPU self-test on Case3 stations
if __name__ == "__main__":
    import json, argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default=os.path.join(HERE, "..", "..", "research", "round_004", "case3_closure_inputs.json"))
    ap.add_argument("--frac", type=float, default=0.10, help="matching-height fraction (training convention pi2=0.10)")
    args = ap.parse_args()
    W = _shape_weights(parse_header())
    print(f"[header] {DEFAULT_HEADER}\n[header] md5={parse_header()['_md5']}  NL={W['NL']} H={W['H']} P={W['P']}")
    rep = json.load(open(args.inputs))
    feats, Ues, xoh = [], [], []
    for st in rep["stations"]:
        hh = [h for h in st["heights"] if abs(h["frac"] - args.frac) < 1e-6]
        if not hh or st.get("Lambda") is None:
            continue
        feats.append([hh[0]["pi1"], hh[0]["pi2"], st["Pi_p"], st["Lambda"]]); Ues.append(st["Ue"]); xoh.append(st["x_over_h"])
    feats = np.array(feats); Ues = np.array(Ues)
    Cf = predict_Cf(feats, W); tau = predict_tau_w(feats, Ues, W)
    print(f"[predict] {len(feats)} stations (frac={args.frac})")
    print(f"[predict] Cf   range [{Cf.min():+.4e}, {Cf.max():+.4e}]  mean {Cf.mean():+.4e}  n_neg(reversed)={int((Cf<0).sum())}")
    print(f"[predict] tau_w range [{tau.min():+.4e}, {tau.max():+.4e}]  (kinematic, rho=1)")
    # DNS-free reconstruction demo at 3 near-wall cells (dy/h~0.047, y_d~(d+0.5)dy, Re_h=2800)
    h = 0.04016; dy = 3.0 * h / 64.0; nu_demo = 0.2 * h / 2800.0
    yb = (np.arange(3) + 0.5) * dy
    u_band = reconstruct_u(tau, yb, nu_demo)
    print(f"[reconstruct] nu~{nu_demo:.3e}  y_band/h={ (yb/h).round(4)}  u_band[st0]={u_band[0].round(4)}  "
          f"u_band[st-1]={u_band[-1].round(4)}")
    print("[selftest] OK -- closure forward + DNS-free reconstruction run clean on CPU.")
