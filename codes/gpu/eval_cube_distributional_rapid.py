#!/usr/bin/env python3
"""One-shot proper-score analysis of the terminal genuine-cube posterior.

This script is inference-only.  It reuses the frozen 1,101-field cube memmap,
the terminal periodic rectified-flow checkpoint and the terminal 160-time test
subset.  Correct, no-wall and far-time-wall interventions use common random
numbers.  The sole primary endpoint and the complete secondary family are
frozen in development/nodes/node_006/preregistration.md.

Evidence level: held-out LES oracle near-wall-band propagation.  This is not
closure-conditioned reconstruction and not solver-coupled WMLES.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch


AP = argparse.ArgumentParser()
AP.add_argument("--base", type=int, default=48)
AP.add_argument("--members", type=int, default=8)
AP.add_argument("--sample-steps", type=int, default=32)
AP.add_argument("--boot", type=int, default=4000)
AP.add_argument("--block", type=int, default=49)
AP.add_argument("--seed", type=int, default=20260723)
AP.add_argument("--case", default="controlled_raw/cube_les")
AP.add_argument(
    "--checkpoint",
    default="codes/results/cube_periodic_topology/cube_periodic_topology.pt",
)
AP.add_argument(
    "--component-ref",
    default="codes/results/cube_periodic_topology/cube_periodic_topology_components.npz",
)
AP.add_argument(
    "--terminal-result",
    default="codes/results/cube_periodic_topology/cube_periodic_topology_results.json",
)
AP.add_argument(
    "--output-dir",
    default="codes/results/cube_distributional_rapid",
)
A = AP.parse_args()

EXPECTED = {
    "data": "8bac93f1537eab6667d692282b76c7bccd28f28965d35ea97668bcc2567bc45a",
    "times": "99e8e0a45cf6c361bcefc10b251ceac15bd60992d42255cf76c945eae9655482",
    "checkpoint": "6f507fd1fb97a7e52dd60a631507dbc37d2005fdd12abd856f165eed8f6135c2",
    "component_ref": "28264e996586f961b1a3cd8c369f494f62d39d4d18d73f0625fc4983e4ab3d18",
    "terminal_result": "0fc302421d622879e00ef16a14636fb8849ee65c0680c85491659788807f914d",
}
ARMS = ("correct", "no_wall", "far_time_wall")
PHYSICAL = (
    "component_spectrum_log_rmse",
    "reynolds_stress_profile_nrmse",
    "coverage80_abs_error",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> None:
    got = sha256(path)
    if got != expected:
        raise RuntimeError(f"{label} SHA-256 mismatch: {got} != {expected}")


# Import the exact terminal architecture without allowing this script's
# arguments to be consumed by the earlier producer.
saved_argv = sys.argv
sys.argv = ["eval_cube_periodic_topology.py", "--base", str(A.base)]
import eval_cube_periodic_topology as PERIODIC  # noqa: E402
sys.argv = saved_argv
BASE = PERIODIC.B


def circular_block_indices(n: int, block: int, draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(draws, nb))
    offsets = np.arange(block)[None, None, :]
    return ((starts[:, :, None] + offsets).reshape(draws, -1) % n)[:, :n]


@torch.no_grad()
def posterior_ensemble(
    model: torch.nn.Module,
    truth: np.ndarray,
    donor: np.ndarray,
    mu: np.ndarray,
    sd: np.ndarray,
    fluid: np.ndarray,
    band: np.ndarray,
    arm: str,
    seed: int,
) -> np.ndarray:
    """Return all physical-space posterior members for one held-out time."""
    dev = BASE.DEV
    ft = torch.tensor(fluid[None, None], dtype=torch.float32, device=dev)
    bt = torch.tensor(band[None, None], dtype=torch.float32, device=dev)
    muv = torch.tensor(mu[None, :, None, None, None], dtype=torch.float32, device=dev)
    sdv = torch.tensor(sd[None, :, None, None, None], dtype=torch.float32, device=dev)
    tr = torch.from_numpy(truth[None]).to(dev)
    trn = ((tr - muv) / sdv) * ft

    if arm == "correct":
        obs = trn
        cm = bt
    elif arm == "far_time_wall":
        dn = torch.from_numpy(donor[None]).to(dev)
        obs = ((dn - muv) / sdv) * ft
        cm = bt
    elif arm == "no_wall":
        obs = torch.zeros_like(trn)
        cm = torch.zeros_like(bt)
    else:
        raise KeyError(arm)

    obs = (obs * cm).repeat_interleave(A.members, 0)
    cm = cm.repeat_interleave(A.members, 0)
    fluid_m = ft.expand(A.members, -1, -1, -1, -1)
    generator = torch.Generator(device=dev).manual_seed(int(seed))
    x = torch.randn(
        (A.members, 3, 48, 96, 48),
        generator=generator,
        dtype=torch.float32,
        device=dev,
    ) * fluid_m
    zobs = x.clone()

    for j in range(A.sample_steps, 0, -1):
        t = j / A.sample_steps
        tnext = (j - 1) / A.sample_steps
        tv = torch.full((A.members,), t, dtype=torch.float32, device=dev)
        velocity = model(x, tv, obs, cm, fluid_m)
        x = (x + (tnext - t) * velocity) * fluid_m
        bridge = (1 - tnext) * obs + tnext * zobs
        x = x * (1 - cm) + bridge * cm

    physical = (x * sdv + muv) * fluid_m
    return physical.float().cpu().numpy()


def standardized(field: np.ndarray, mean_field: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (field - mean_field) / sd[:, None, None, None]


def energy_distances(
    ensemble_std: np.ndarray,
    truth_std: np.ndarray,
    outer: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Memberwise distances sufficient to recompute the empirical energy score."""
    x = np.ascontiguousarray(ensemble_std[:, :, outer].reshape(A.members, -1), dtype=np.float64)
    y = np.ascontiguousarray(truth_std[:, outer].reshape(-1), dtype=np.float64)
    root_d = math.sqrt(y.size)
    truth_distance = np.linalg.norm(x - y[None], axis=1) / root_d
    norm2 = np.einsum("md,md->m", x, x)
    sq = norm2[:, None] + norm2[None, :] - 2.0 * (x @ x.T)
    pair_distance = np.sqrt(np.maximum(sq, 0.0)) / root_d
    return truth_distance, pair_distance


def component_spectrum(field_std: np.ndarray, outer_y: np.ndarray) -> np.ndarray:
    """Streamwise component spectra on complete outer planes."""
    q = field_std[:, :, outer_y, :]
    fft = np.fft.rfft(q, axis=1)
    power = np.square(np.abs(fft), dtype=np.float64) / (q.shape[1] ** 2)
    return power.mean(axis=(2, 3))[:, 1:13]


def reynolds_profile(field_std: np.ndarray, outer_y: np.ndarray) -> np.ndarray:
    q = field_std[:, :, outer_y, :]
    u, v, w = q
    products = (u * u, v * v, w * w, u * v, u * w, v * w)
    return np.stack([p.mean(axis=(0, 2), dtype=np.float64) for p in products])


def spectrum_loss(pred: np.ndarray, truth: np.ndarray) -> float:
    eps = 1e-12
    return float(np.sqrt(np.mean(np.square(np.log10(pred + eps) - np.log10(truth + eps)))))


def profile_loss(pred: np.ndarray, truth: np.ndarray) -> float:
    num = np.mean(np.square(pred - truth))
    den = np.mean(np.square(truth))
    return float(np.sqrt(num / max(den, 1e-12)))


def coverage_terms(
    ensemble_std: np.ndarray,
    truth_std: np.ndarray,
    outer: np.ndarray,
) -> tuple[int, int, float, float, float]:
    vals = ensemble_std[:, :, outer]
    target = truth_std[:, outer]
    lo = np.quantile(vals, 0.1, axis=0, method="linear")
    hi = np.quantile(vals, 0.9, axis=0, method="linear")
    hits = int(((target >= lo) & (target <= hi)).sum())
    total = int(target.size)
    width_sum = float((hi - lo).sum(dtype=np.float64))
    coverage = hits / total
    return hits, total, width_sum, float(coverage), float(abs(coverage - 0.8))


def summarize_loss(
    values: dict[str, np.ndarray],
    bix: np.ndarray,
    ci: tuple[float, float] = (2.5, 97.5),
) -> dict:
    out: dict[str, object] = {"arms": {}, "improvements": {}}
    boots: dict[str, np.ndarray] = {}
    for arm in ARMS:
        vals = np.asarray(values[arm], dtype=np.float64)
        br = vals[bix].mean(axis=1)
        boots[arm] = br
        out["arms"][arm] = {
            "mean_loss": float(vals.mean()),
            "ci95": [float(np.percentile(br, ci[0])), float(np.percentile(br, ci[1]))],
        }
    for control in ("no_wall", "far_time_wall"):
        delta_vals = np.asarray(values[control]) - np.asarray(values["correct"])
        delta_boot = delta_vals[bix].mean(axis=1)
        lo, hi = np.percentile(delta_boot, ci)
        out["improvements"][f"correct_vs_{control}"] = {
            "control_minus_correct": float(delta_vals.mean()),
            "ci95": [float(lo), float(hi)],
            "ci_positive": bool(lo > 0),
            "secondary_multiplicity_label": control != "",
        }
    return out


def make_figure(result: dict, out_png: Path, out_pdf: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"correct": "#2166ac", "no_wall": "#777777", "far_time_wall": "#b2182b"}
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.45), constrained_layout=True)

    energy = result["primary_energy_score"]
    for i, arm in enumerate(ARMS):
        q = energy["arms"][arm]
        axes[0].errorbar(
            i,
            q["mean_loss"],
            yerr=[[q["mean_loss"] - q["ci95"][0]], [q["ci95"][1] - q["mean_loss"]]],
            fmt="o",
            color=colors[arm],
            capsize=3,
        )
    axes[0].set_xticks(range(3), ["correct", "no wall", "far-time"], rotation=18)
    axes[0].set_ylabel("outer energy score ↓")
    axes[0].set_title("a  Proper distributional score")

    controls = ("no_wall", "far_time_wall")
    for i, control in enumerate(controls):
        q = energy["improvements"][f"correct_vs_{control}"]
        axes[1].errorbar(
            i,
            q["control_minus_correct"],
            yerr=[[
                q["control_minus_correct"] - q["ci95"][0]
            ], [
                q["ci95"][1] - q["control_minus_correct"]
            ]],
            fmt="o",
            color=colors[control],
            capsize=3,
        )
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xticks(range(2), ["no wall", "far-time"], rotation=18)
    axes[1].set_ylabel("control − correct (positive is better)")
    axes[1].set_title("b  Paired outer improvement")

    x = np.arange(len(PHYSICAL))
    width = 0.34
    labels = ["spectra", "stress profiles", "80% coverage"]
    for j, control in enumerate(controls):
        points, low, high = [], [], []
        for metric in PHYSICAL:
            q = result["secondary_physical_statistics"][metric]["improvements"][
                f"correct_vs_{control}"
            ]
            points.append(q["control_minus_correct"])
            low.append(q["control_minus_correct"] - q["ci95"][0])
            high.append(q["ci95"][1] - q["control_minus_correct"])
        axes[2].bar(
            x + (j - 0.5) * width,
            points,
            width,
            color=colors[control],
            alpha=0.82,
            label=control.replace("_", " "),
        )
        axes[2].errorbar(
            x + (j - 0.5) * width,
            points,
            yerr=[low, high],
            fmt="none",
            ecolor="black",
            capsize=2,
            lw=0.8,
        )
    axes[2].axhline(0, color="black", lw=0.8)
    axes[2].set_xticks(x, labels, rotation=18)
    axes[2].set_ylabel("control − correct loss")
    axes[2].set_title("c  Preregistered physical family")
    axes[2].legend(frameon=False, fontsize=8)

    fig.savefig(out_png, dpi=240)
    fig.savefig(out_pdf)
    plt.close(fig)


def main() -> None:
    t0 = time.time()
    case = Path(A.case)
    data_path = case / "cube_ds2_float16.npy"
    times_path = case / "cube_ds2_times.npy"
    checkpoint = Path(A.checkpoint)
    component_ref = Path(A.component_ref)
    terminal_result = Path(A.terminal_result)
    out_dir = Path(A.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "cube_distributional_rapid_results.json"
    out_npz = out_dir / "cube_distributional_rapid_components.npz"
    out_png = out_dir / "fig_cube_distributional_rapid.png"
    out_pdf = out_dir / "fig_cube_distributional_rapid.pdf"

    require_hash(data_path, EXPECTED["data"], "cube memmap")
    require_hash(times_path, EXPECTED["times"], "cube physical times")
    require_hash(checkpoint, EXPECTED["checkpoint"], "periodic checkpoint")
    require_hash(component_ref, EXPECTED["component_ref"], "terminal component reference")
    require_hash(terminal_result, EXPECTED["terminal_result"], "terminal periodic result")

    if BASE.DEV.type != "cuda":
        raise RuntimeError("rapid cube inference requires CUDA")
    if A.members != 8 or A.sample_steps != 32 or A.block != 49 or A.boot != 4000:
        raise RuntimeError("members/sample-steps/block/bootstrap differ from preregistration")

    mm = np.load(data_path, mmap_mode="r")
    times = np.load(times_path)
    if mm.shape != (1101, 3, 48, 96, 48) or len(times) != len(mm):
        raise RuntimeError(f"unexpected frozen cube shape/times: {mm.shape}, {times.shape}")
    fluid, band, dist, _, y_grid, _ = BASE.geometry()
    outer = fluid & (dist > 0.5)
    outer_y = np.flatnonzero(outer.all(axis=(0, 2)))
    if len(outer_y) == 0:
        raise RuntimeError("no complete d/h>0.5 outer planes")

    frozen = np.load(component_ref)
    test_idx = np.asarray(frozen["test_idx"], dtype=np.int64)
    donor_idx = np.asarray(frozen["donor_idx"], dtype=np.int64)
    if len(test_idx) != 160 or not np.array_equal(donor_idx, np.roll(test_idx, 80)):
        raise RuntimeError("terminal test/donor index contract changed")
    train_idx = np.arange(660, dtype=np.int64)
    mu, sd, mean_field = BASE.prepare_stats(mm, train_idx, fluid)

    ckpt = torch.load(checkpoint, map_location=BASE.DEV, weights_only=False)
    if not ckpt.get("complete", False) or int(ckpt.get("step", -1)) != 90000:
        raise RuntimeError("checkpoint is not the frozen terminal 90k state")
    model = PERIODIC.PeriodicFlowUNet3D(base=A.base).to(BASE.DEV)
    model.load_state_dict(ckpt["ema"], strict=True)
    model.eval()

    n = len(test_idx)
    truth_spectrum = np.empty((n, 3, 12), dtype=np.float64)
    truth_profile = np.empty((n, 6, len(outer_y)), dtype=np.float64)
    truth_std_cache: list[np.ndarray] = []
    for pos, idx in enumerate(test_idx):
        truth = np.asarray(mm[idx], dtype=np.float32)
        tstd = standardized(truth, mean_field, sd)
        truth_std_cache.append(tstd)
        truth_spectrum[pos] = component_spectrum(tstd, outer_y)
        truth_profile[pos] = reynolds_profile(tstd, outer_y)

    payload: dict[str, np.ndarray] = {
        "test_idx": test_idx,
        "donor_idx": donor_idx,
        "test_time": times[test_idx].astype(np.float64),
        "sampler_seed": (910000 + np.arange(n)).astype(np.int64),
        "outer_y_index": outer_y.astype(np.int64),
        "truth_spectrum": truth_spectrum,
        "truth_reynolds_profile": truth_profile,
    }
    metric_values: dict[str, dict[str, np.ndarray]] = {
        "energy_score": {},
        **{metric: {} for metric in PHYSICAL},
    }

    for arm in ARMS:
        truth_distance = np.empty((n, A.members), dtype=np.float64)
        pair_distance = np.empty((n, A.members, A.members), dtype=np.float64)
        truth_term = np.empty(n, dtype=np.float64)
        pair_term = np.empty(n, dtype=np.float64)
        es = np.empty(n, dtype=np.float64)
        pred_spectrum = np.empty_like(truth_spectrum)
        pred_profile = np.empty_like(truth_profile)
        coverage_hits = np.empty(n, dtype=np.int64)
        coverage_total = np.empty(n, dtype=np.int64)
        coverage_width_sum = np.empty(n, dtype=np.float64)
        coverage = np.empty(n, dtype=np.float64)
        coverage_error = np.empty(n, dtype=np.float64)
        spectrum_rmse = np.empty(n, dtype=np.float64)
        profile_nrmse = np.empty(n, dtype=np.float64)

        for pos, (idx, donor) in enumerate(zip(test_idx, donor_idx)):
            truth = np.asarray(mm[idx], dtype=np.float32)
            donor_field = np.asarray(mm[donor], dtype=np.float32)
            ensemble = posterior_ensemble(
                model,
                truth,
                donor_field,
                mu,
                sd,
                fluid,
                band,
                arm,
                seed=910000 + pos,
            )
            estd = standardized(ensemble, mean_field[None], sd)
            tstd = truth_std_cache[pos]
            truth_distance[pos], pair_distance[pos] = energy_distances(estd, tstd, outer)
            truth_term[pos] = truth_distance[pos].mean()
            pair_term[pos] = pair_distance[pos].mean()
            es[pos] = truth_term[pos] - 0.5 * pair_term[pos]

            member_spectra = np.stack([component_spectrum(q, outer_y) for q in estd])
            pred_spectrum[pos] = member_spectra.mean(axis=0)
            spectrum_rmse[pos] = spectrum_loss(pred_spectrum[pos], truth_spectrum[pos])

            member_profiles = np.stack([reynolds_profile(q, outer_y) for q in estd])
            pred_profile[pos] = member_profiles.mean(axis=0)
            profile_nrmse[pos] = profile_loss(pred_profile[pos], truth_profile[pos])

            (coverage_hits[pos], coverage_total[pos], coverage_width_sum[pos],
             coverage[pos], coverage_error[pos]) = coverage_terms(estd, tstd, outer)
            if (pos + 1) % 10 == 0 or pos + 1 == n:
                print(
                    f"[rapid] arm={arm} {pos+1}/{n} "
                    f"ES={es[:pos+1].mean():.6f} elapsed={time.time()-t0:.0f}s",
                    flush=True,
                )

        payload[f"energy_truth_distance__{arm}"] = truth_distance
        payload[f"energy_pair_distance__{arm}"] = pair_distance
        payload[f"energy_truth_term__{arm}"] = truth_term
        payload[f"energy_pair_term__{arm}"] = pair_term
        payload[f"energy_score__{arm}"] = es
        payload[f"pred_spectrum__{arm}"] = pred_spectrum
        payload[f"spectrum_log_rmse__{arm}"] = spectrum_rmse
        payload[f"pred_reynolds_profile__{arm}"] = pred_profile
        payload[f"reynolds_profile_nrmse__{arm}"] = profile_nrmse
        payload[f"coverage80_hits__{arm}"] = coverage_hits
        payload[f"coverage80_total__{arm}"] = coverage_total
        payload[f"coverage80_width_sum__{arm}"] = coverage_width_sum
        payload[f"coverage80__{arm}"] = coverage
        payload[f"coverage80_abs_error__{arm}"] = coverage_error
        metric_values["energy_score"][arm] = es
        metric_values["component_spectrum_log_rmse"][arm] = spectrum_rmse
        metric_values["reynolds_stress_profile_nrmse"][arm] = profile_nrmse
        metric_values["coverage80_abs_error"][arm] = coverage_error

    np.savez_compressed(out_npz, **payload)
    bix = circular_block_indices(n, A.block, A.boot, A.seed)
    primary = summarize_loss(metric_values["energy_score"], bix)
    secondary = {
        metric: summarize_loss(metric_values[metric], bix) for metric in PHYSICAL
    }
    primary_pass = all(
        q["ci_positive"] for q in primary["improvements"].values()
    )
    physical_wins = {
        metric: all(
            q["ci_positive"] for q in secondary[metric]["improvements"].values()
        )
        for metric in PHYSICAL
    }
    physical_pass = sum(physical_wins.values()) >= 2
    terminal = json.loads(terminal_result.read_text())
    result = {
        "_meta": {
            "producer": Path(__file__).name,
            "producer_sha256": sha256(Path(__file__)),
            "preregistration": "development/nodes/node_006/preregistration.md",
            "inference_only": True,
            "device": str(BASE.DEV),
            "evidence_level": (
                "held-out LES oracle near-wall-band propagation; "
                "not closure-conditioned and not solver-coupled"
            ),
            "regime": terminal["_meta"]["regime"],
            "n_post_spinup": int(len(mm)),
            "n_eval": int(n),
            "test_span_integral_times": float(terminal["_meta"]["test_span_in_integral_times"]),
            "effective_test_events_approx": float(
                terminal["_meta"]["n_test_available"]
                / terminal["_meta"]["tau_integral_snapshots"]
            ),
            "low_event_count_warning": (
                "The test interval spans only about 3.5 estimated integral times; "
                "block resampling does not create independent physical events."
            ),
            "support": "solid- and supplied-band-excluded d/h>0.5 outer volume",
            "outer_cells": int(outer.sum()),
            "complete_outer_y_planes": int(len(outer_y)),
            "members": int(A.members),
            "sample_steps": int(A.sample_steps),
            "common_random_numbers": True,
            "sampler_seed_rule": "910000 + frozen test position, identical across arms",
            "bootstrap": {
                "kind": "circular moving physical-time blocks",
                "block_evaluation_samples": int(A.block),
                "draws": int(A.boot),
                "seed": int(A.seed),
            },
            "input_sha256": {
                "data": sha256(data_path),
                "times": sha256(times_path),
                "checkpoint": sha256(checkpoint),
                "component_ref": sha256(component_ref),
                "terminal_result": sha256(terminal_result),
            },
            "components": out_npz.name,
            "components_sha256": sha256(out_npz),
            "wall_seconds": round(time.time() - t0, 1),
        },
        "primary_energy_score": primary,
        "secondary_physical_statistics": secondary,
        "coverage_descriptives": {
            arm: {
                "coverage80": float(payload[f"coverage80__{arm}"].mean()),
                "mean_standardized_width": float(
                    payload[f"coverage80_width_sum__{arm}"].sum()
                    / payload[f"coverage80_total__{arm}"].sum()
                ),
            }
            for arm in ARMS
        },
        "registered_gates": {
            "primary_correct_beats_both_controls_ci95": bool(primary_pass),
            "physical_family_wins": physical_wins,
            "physical_at_least_two_of_three": bool(physical_pass),
            "rapid_distributional_claim_pass": bool(primary_pass and physical_pass),
        },
        "claim_boundary": {
            "terminal_outer_posterior_mean_R2": float(
                terminal["evaluation"]["arms"]["correct"]["outer_d_gt_0p5h"][
                    "R2_fluct_balanced"
                ]
            ),
            "positive_outer_pointwise_mean_claim": False,
            "distributional_claim_requires_registered_gates": True,
        },
    }
    out_json.write_text(json.dumps(result, indent=2, sort_keys=True))
    make_figure(result, out_png, out_pdf)
    digest = sha256(out_json)
    out_json.with_suffix(".sha256").write_text(f"{digest}  {out_json.name}\n")
    out_npz.with_suffix(".sha256").write_text(f"{sha256(out_npz)}  {out_npz.name}\n")
    out_png.with_suffix(".sha256").write_text(f"{sha256(out_png)}  {out_png.name}\n")
    out_pdf.with_suffix(".sha256").write_text(f"{sha256(out_pdf)}  {out_pdf.name}\n")
    print(
        f"=== done ===\n[out] {out_json} sha256={digest}\n"
        f"[components] {out_npz} sha256={sha256(out_npz)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
