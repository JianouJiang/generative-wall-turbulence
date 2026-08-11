#!/usr/bin/env python3
"""Matched test of periodic topology in the genuine-3-D cube posterior.

The preserved cube posterior uses zero padding on every convolutional axis even
though the LES cell is periodic in x and z.  This wrapper changes only that
boundary topology: x/z convolutional padding is circular and y padding remains
zero.  Data, chronological split, model width/depth, optimizer updates,
rectified-flow sampler and correct/no-wall/far-time-wall interventions are
otherwise inherited from the byte-frozen adequate-cube producer.

Evidence remains held-out LES oracle-band propagation, not closure-conditioned
reconstruction or solver coupling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


P = argparse.ArgumentParser()
P.add_argument("--smoke", action="store_true")
P.add_argument("--base", type=int, default=48)
P.add_argument("--steps-train", type=int, default=90000)
P.add_argument("--sample-steps", type=int, default=32)
P.add_argument("--members", type=int, default=8)
P.add_argument("--boot", type=int, default=4000)
P.add_argument("--stack", default="controlled_raw/cube_les/stack")
P.add_argument("--case", default="controlled_raw/cube_les")
A = P.parse_args()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Import the component-saving adequate wrapper without executing its main.
saved_argv = sys.argv
sys.argv = ["eval_cube_3d_coupling_adequate.py"]
import eval_cube_3d_coupling_adequate as Q  # noqa: E402
sys.argv = saved_argv
B = Q.B


class PeriodicXZConv3d(nn.Module):
    """Conv3d with circular x/z padding and zero wall-normal padding."""

    def __init__(self, ci: int, co: int, kernel_size: int, stride: int = 1,
                 padding: int = 0, bias: bool = True):
        super().__init__()
        self.padding = int(padding)
        self.conv = nn.Conv3d(ci, co, kernel_size, stride=stride, padding=0, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.padding
        if p:
            # Tensor axes are (N,C,x,y,z). F.pad lists z, y, x.
            x = F.pad(x, (p, p, 0, 0, 0, 0), mode="circular")
            x = F.pad(x, (0, 0, p, p, 0, 0), mode="constant", value=0.0)
            x = F.pad(x, (0, 0, 0, 0, p, p), mode="circular")
        return self.conv(x)


class PeriodicBlock3D(nn.Module):
    def __init__(self, ci: int, co: int, ce: int = 128):
        super().__init__()
        self.n1 = nn.GroupNorm(min(8, ci), ci)
        self.c1 = PeriodicXZConv3d(ci, co, 3, padding=1)
        self.e = nn.Linear(ce, co)
        self.n2 = nn.GroupNorm(min(8, co), co)
        self.c2 = PeriodicXZConv3d(co, co, 3, padding=1)
        self.skip = nn.Conv3d(ci, co, 1) if ci != co else nn.Identity()

    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        h = self.c1(F.silu(self.n1(x))) + self.e(e)[:, :, None, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class PeriodicFlowUNet3D(nn.Module):
    """Parameter-matched parent topology with physical x/z padding."""

    def __init__(self, base: int = A.base):
        super().__init__()
        base = int(A.base)
        self.temb = nn.Sequential(nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 128))
        self.inc = PeriodicXZConv3d(8, base, 3, padding=1)
        self.b0 = PeriodicBlock3D(base, base)
        self.d1 = PeriodicXZConv3d(base, 2 * base, 4, stride=2, padding=1)
        self.b1 = PeriodicBlock3D(2 * base, 2 * base)
        self.d2 = PeriodicXZConv3d(2 * base, 4 * base, 4, stride=2, padding=1)
        self.m1 = PeriodicBlock3D(4 * base, 4 * base)
        self.m2 = PeriodicBlock3D(4 * base, 4 * base)
        self.u1 = nn.ConvTranspose3d(4 * base, 2 * base, 4, 2, 1)
        self.ub1 = PeriodicBlock3D(4 * base, 2 * base)
        self.u0 = nn.ConvTranspose3d(2 * base, base, 4, 2, 1)
        self.ub0 = PeriodicBlock3D(2 * base, base)
        self.out = PeriodicXZConv3d(base, 3, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor,
                cmask: torch.Tensor, fluid: torch.Tensor) -> torch.Tensor:
        e = self.temb(B.noise_embed(t))
        h0 = self.b0(self.inc(torch.cat([x, cond, cmask, fluid], 1)), e)
        h1 = self.b1(self.d1(h0), e)
        m = self.m2(self.m1(self.d2(h1), e), e)
        u1 = self.ub1(torch.cat([self.u1(m), h1], 1), e)
        u0 = self.ub0(torch.cat([self.u0(u1), h0], 1), e)
        return self.out(F.silu(u0)) * fluid


def configure() -> tuple[Path, Path]:
    original_results = Path(B.RESULTS)
    run_dir = original_results / ("cube_periodic_topology_smoke" if A.smoke
                                  else "cube_periodic_topology")
    run_dir.mkdir(parents=True, exist_ok=True)
    tag = "cube_periodic_topology_smoke" if A.smoke else "cube_periodic_topology"
    B.RESULTS = run_dir
    B.OUT = run_dir / f"{tag}_results.json"
    B.FIG = run_dir / f"fig_{tag}.png"
    B.CKPT = run_dir / f"{tag}.pt"
    Q.COMP = run_dir / f"{tag}_components.npz"
    B.STACK = Path(A.stack)
    B.CASE = Path(A.case)
    B.A.smoke = False
    B.A.steps_train = 8 if A.smoke else int(A.steps_train)
    B.A.sample_steps = 3 if A.smoke else int(A.sample_steps)
    B.A.members = 2 if A.smoke else int(A.members)
    B.A.boot = 100 if A.smoke else int(A.boot)
    B.FlowUNet3D = PeriodicFlowUNet3D
    return original_results, run_dir


def main() -> None:
    original_results, run_dir = configure()
    # Inherited producer writes all sufficient statistics and uses common arm noise.
    B.main()
    result = json.loads(B.OUT.read_text())
    baseline_path = original_results / "cube3d_coupling_adequate_results.json"
    baseline = json.loads(baseline_path.read_text())
    regions = ("full_support_excluded", "near_support_excluded_d_le_0p5h",
               "outer_d_gt_0p5h")
    result["_meta"].update({
        "producer_script": Path(__file__).name,
        "producer_script_sha256": sha256(Path(__file__)),
        "mechanism": "exact circular x/z convolutional topology; zero y padding",
        "single_changed_factor": "convolutional padding topology",
        "parameter_matched_baseline": baseline_path.name,
        "parameter_matched_baseline_sha256": sha256(baseline_path),
        "baseline_correct_R2": {
            key: baseline["evaluation"]["arms"]["correct"][key]["R2_fluct_balanced"]
            for key in regions
        },
        "correct_R2_change_from_zero_padding": {
            key: (result["evaluation"]["arms"]["correct"][key]["R2_fluct_balanced"]
                  - baseline["evaluation"]["arms"]["correct"][key]["R2_fluct_balanced"])
            for key in regions
        },
        "components": Q.COMP.name,
        "components_sha256": sha256(Q.COMP),
        "common_sampler_noise_across_arms": True,
        "evidence_level": (
            "held-out LES oracle near-wall-band propagation; not closure-conditioned "
            "and not solver-coupled"
        ),
    })
    result["topology_gates"] = {
        "same_parameter_count_as_baseline": (
            int(result["training"]["n_parameters"])
            == int(baseline["training"]["n_parameters"])
        ),
        "positive_outer_absolute_skill": (
            result["evaluation"]["arms"]["correct"]["outer_d_gt_0p5h"]["ci95"][0] > 0
        ),
        "outer_correct_beats_no_wall": (
            result["evaluation"]["deltas"]["correct_minus_no_wall"]
            ["outer_d_gt_0p5h"]["ci95"][0] > 0
        ),
        "outer_correct_beats_wrong_wall": (
            result["evaluation"]["deltas"]["correct_minus_wrong_wall"]
            ["outer_d_gt_0p5h"]["ci95"][0] > 0
        ),
    }
    B.OUT.write_text(json.dumps(result, indent=2, sort_keys=True))
    digest = sha256(B.OUT)
    B.OUT.with_suffix(".sha256").write_text(f"{digest}  {B.OUT.name}\n")
    print(f"=== periodic-topology done ===\n[out] {B.OUT} sha256={digest}\n"
          f"[components] {Q.COMP} sha256={sha256(Q.COMP)}", flush=True)


if __name__ == "__main__":
    main()
