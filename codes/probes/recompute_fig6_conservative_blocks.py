#!/usr/bin/env python3
"""Recompute Figure 6 intervals at the release's CONSERVATIVE bootstrap block.

Why this exists
---------------
The node-004 interface producer registered its uncertainty at the historical
block (``ceil(tau)`` snapshots, i.e. 49 evaluation samples at this stride), while
the rest of the release adopts the more conservative ``1.2551*tau`` rule. The
node-004 panel flagged the inconsistency:

    "The E2 preregistration also reverted to the historical 97.59-snapshot/block-49
     convention although the release elsewhere adopts the more conservative
     122.48-snapshot/block-57 convention."  -- champion verdict, section 5

Rather than restate that in prose, this probe recomputes every Figure 6 arm and
delta interval at the conservative block, using the SAME resampler the producers
use, from the hash-pinned per-target components. The frozen result JSON is never
mutated: the conservative statistics are written to a separate derived file, so
the original bytes and their pinned hash survive untouched.

The block length changes only the resampling dependence assumption, never the
point estimate, so the points are re-derived and asserted equal to the frozen
ones. The realised interval can be marginally narrower or wider than the release
one because it is drawn from a different random stream; what matters is whether
each frozen significance statement survives, and that flag is reported per
contrast and per region.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "manuscript" / "source_data" / "fig6_interface"
RESULTS = SRC / "e2_traction_interface_results.json"
COMPONENTS = SRC / "e2_traction_interface_components.npz"
OUT = SRC / "e2_traction_interface_conservative_blocks.json"

CONSERVATIVE_FACTOR = 1.2551  # release rule, identical to the node-005 producer
BOOT = 4000
SEED = 45  # the node-005 producer's conservative-block stream


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def block_indices(n: int, block: int, boot: int, rng) -> np.ndarray:
    """Circular moving-block resampler, byte-identical to the producers."""
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(boot, nb))
    off = np.arange(block)[None, None]
    return ((starts[:, :, None] + off).reshape(boot, -1) % n)[:, :n]


def main() -> int:
    doc = json.loads(RESULTS.read_text())
    ev = doc["evaluation"]
    z = np.load(COMPONENTS)

    test_idx = z["test_idx"]
    n = len(test_idx)
    stride = max(1, int(round(np.median(np.diff(test_idx))))) if n > 1 else 1

    # The producer stored the integral time it used; recover the block it implies.
    tau = float(doc["_meta"]["frozen_record"]["tau_integral_snapshots"]) if (
        "frozen_record" in doc["_meta"]
        and "tau_integral_snapshots" in doc["_meta"]["frozen_record"]
    ) else float(doc["_meta"].get("tau_integral_snapshots", 97.59))
    block_snapshots = int(np.ceil(tau))
    b_release = max(1, int(round(block_snapshots / stride)))
    b_cons = max(1, int(round(CONSERVATIVE_FACTOR * block_snapshots / stride)))
    if b_cons <= b_release:
        raise SystemExit("conservative block must exceed the release block")

    rng = np.random.default_rng(SEED)
    bix = block_indices(n, b_cons, BOOT, rng)

    regions = sorted({k.split("__")[2] for k in z.files if k.startswith("sse__")})
    arms = sorted({k.split("__")[1] for k in z.files if k.startswith("sse__")})

    def boot_r2(arm: str, region: str) -> np.ndarray:
        sse = z[f"sse__{arm}__{region}"]
        sst = z[f"sst__{region}"]
        return 1 - sse[bix].sum(1) / (sst[bix].sum(1) + 1e-12)

    boots = {a: {r: boot_r2(a, r) for r in regions} for a in arms}

    # Reproduction tolerance. The stored points were accumulated on the GPU in a
    # different summation order, so agreement is asserted at 1e-6 rather than
    # bitwise; the realised maximum deviation is reported, not hidden.
    POINT_TOL = 1e-6
    max_dev = 0.0
    out_arms: dict = {}
    for arm in arms:
        out_arms[arm] = {}
        for region in regions:
            sse, sst = z[f"sse__{arm}__{region}"], z[f"sst__{region}"]
            point = float(1 - sse.sum() / (sst.sum() + 1e-12))
            br = boots[arm][region]
            out_arms[arm][region] = {
                "R2_fluct_balanced": point,
                "ci95_conservative_block": [
                    float(np.percentile(br, 2.5)),
                    float(np.percentile(br, 97.5)),
                ],
            }
            stored = ev["arms"].get(arm, {}).get(region, {}).get("R2_fluct_balanced")
            if stored is not None:
                max_dev = max(max_dev, abs(stored - point))
                if abs(stored - point) > POINT_TOL:
                    raise SystemExit(
                        f"point estimate mismatch for {arm}/{region}: "
                        f"{point} != {stored} (frozen)"
                    )

    out_deltas: dict = {}
    for name in ev["deltas"]:
        a, b = name.split("_minus_")
        if a not in boots or b not in boots:
            continue
        out_deltas[name] = {}
        for region in regions:
            d = boots[a][region] - boots[b][region]
            lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
            stored = ev["deltas"][name][region]
            point = stored["point"]
            rel_lo, rel_hi = stored["ci95"]
            out_deltas[name][region] = {
                "point": point,
                "ci95_conservative_block": [lo, hi],
                "ci95_release_block": [rel_lo, rel_hi],
                "ci_positive_conservative": bool(lo > 0),
                "ci_negative_conservative": bool(hi < 0),
                "significance_preserved": bool(
                    (stored.get("ci_positive") and lo > 0)
                    or (stored.get("ci_negative") and hi < 0)
                    or (not stored.get("ci_positive") and not stored.get("ci_negative"))
                ),
                "interval_width_ratio_conservative_over_release": float(
                    (hi - lo) / max(rel_hi - rel_lo, 1e-12)
                ),
            }

    lost = [
        f"{name}/{region}"
        for name, per in out_deltas.items()
        for region, v in per.items()
        if not v["significance_preserved"]
    ]

    payload = {
        "_what": (
            "Figure 6 intervals recomputed at the release's conservative bootstrap "
            "block from the hash-pinned per-target components. The frozen result "
            "JSON is not modified; point estimates are re-derived and asserted "
            "equal to the frozen ones."
        ),
        "_why": (
            "The node-004 producer registered intervals at the historical block; "
            "the rest of the release uses the conservative 1.2551*tau rule. The "
            "panel required one convention."
        ),
        "inputs": {
            "results": {"path": str(RESULTS.relative_to(ROOT)), "sha256": sha256(RESULTS)},
            "components": {
                "path": str(COMPONENTS.relative_to(ROOT)),
                "sha256": sha256(COMPONENTS),
            },
        },
        "protocol": {
            "resampler": "circular moving block, identical implementation to the producers",
            "n_eval": int(n),
            "stride_snapshots": int(stride),
            "tau_integral_snapshots": tau,
            "block_snapshots_release": block_snapshots,
            "block_snapshots_conservative": float(CONSERVATIVE_FACTOR * block_snapshots),
            "eval_block_release": b_release,
            "eval_block_conservative": b_cons,
            "boot": BOOT,
            "seed": SEED,
            "point_reproduction_max_abs_deviation": max_dev,
            "point_reproduction_tolerance": POINT_TOL,
        },
        "arms": out_arms,
        "deltas": out_deltas,
        "significance_lost_under_conservative_block": lost,
    }
    OUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    print(f"n_eval={n} stride={stride} tau={tau}")
    print(f"release block={b_release} -> conservative block={b_cons}")
    for name in ("correct_physwall_minus_no_wall", "tau_lift_oracle_minus_no_wall"):
        v = out_deltas[name]["full_support_excluded"]
        print(
            f"  {name}: {v['point']:+.5f} "
            f"release {v['ci95_release_block']} -> "
            f"conservative {v['ci95_conservative_block']} "
            f"(width x{v['interval_width_ratio_conservative_over_release']:.2f})"
        )
    if lost:
        print(f"SIGNIFICANCE LOST at the conservative block: {lost}")
    else:
        print("every frozen significance statement survives the conservative block")
    print(f"[write] {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
