"""
eval_wallstress_cond.py -- DEV L3 (node_003) DECISIVE run: DIRECT wall-stress conditioning.

WHY (the binding L2-panel reservation, all three judges converged on it):
  On the cross-kind Case1 wall-jet the *equilibrium bridge* B is the bottleneck. Even the PERFECT
  DNS wall stress, reconstructed to a near-wall velocity band through the equilibrium (Reichardt)
  profile, does NOT move the whole (total) field: `passthrough` fails to beat `no_wall` CI-separated
  (Delta = +0.0014/+0.0032, ci_pos=False), the order correct>=passthrough>=closure is violated, and
  the wrong-shaped band even corrupts the mean field (passthrough mean-R2 0.933 < no_wall 0.937). The
  band_h sensitivity (0.83 -> 0.24 -> -0.17) pinned the equilibrium bridge as the cause.
  Judge3 (champion) named the decisive lever: "condition the generator DIRECTLY on predicted tau_w,
  bypassing the equilibrium-profile bridge" -- an L3+ experiment. This script RUNS it.

WHAT CHANGES vs eval_closure_arm_case1.py (everything else -- backbone, budget, split, VM lever,
block bootstrap -- is IDENTICAL, so the comparison is clean apples-to-apples on the same flow):
  * The near-wall conditioning is the WALL SHEAR STRESS tau_w(x) itself, imposed as a dedicated
    conditioning channel in the wall-adjacent band, INSTEAD of a reconstructed velocity band. The
    generator LEARNS the near-wall shape (and its propagation into the outer recirculation) from DNS
    during training, so it no longer relies on the (wrong) equilibrium profile. NO Reichardt bridge.
  * Training conditions on the INSTANTANEOUS DNS tau_w(x,t) (with empty-dropout p_empty); deployment
    feeds a STEADY tau_w(x) (mean DNS, or the a-priori closure) -- the honest steady-closure scenario.

Band-degradation ladder (same 6 arms, now on the tau_w channel):
   correct     = instantaneous DNS tau_w(x,t)         [in-distribution oracle ceiling]
   passthrough = mean DNS tau_w(x)  (= data climatology of the wall stress; the steady-perfect wall)
   closure     = a-priori familyclass-predicted mean tau_w(x)  (zero target wall data; cross-kind)
   wrong_mean  = spatially-FLAT tau_w (correct domain-average magnitude, no separation structure)
   random      = spatial shuffle of the instantaneous DNS tau_w (right magnitude, wrong location)
   no_wall     = empty conditioning
Pre-registered gates (the ones that FAILED with the equilibrium bridge are the make-or-break):
   G_passthrough_beats_no_wall_total  -- THE mechanism-repair gate (was False with the bridge)
   G_closure_beats_no_wall_{total,fluct}
   G_closure_beats_wrong_mean_total   -- the closure's a-priori x-structure beats a flat wall
   G_closure_ge_passthrough_total     -- a-priori closure matches the data-climatology wall stress
   G_correct_beats_no_wall_total ; G_order_... ; G_crps_beats_point ; G_phi_pos_total
phi = (R2_closure - R2_none)/(R2_passthrough - R2_none)  -- fraction of the steady-perfect-wall gain
   the deployable closure recovers; now WELL-DEFINED because passthrough-none is no longer ~0.

EVIDENCE NOTE (2026-07-17 audit): the historical closure-input JSON used here was extracted from the
target DNS time mean. The closure arm is therefore a byte-verified a-priori diagnostic, not a
leakage-free runtime deployment. ``eval_runtime_closure_conditioning.py`` is the decisive offline
serial-composition test: it extracts inputs independently from each resolved coarse snapshot. Neither
script is solver-coupled WMLES; scalar tau_w is a generator feature channel, not a momentum BC.

Output (per config): codes/results/wallstress_cond_<tag>_results.json (+ _fields.npz, + ckpts on pod).
DELIVERY: gpu_run.sh run codes/gpu/eval_wallstress_cond.py [args]; gpu_run.sh wait; '=== done ==='.
"""
import os, json, time, math, argparse, hashlib, sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

ap = argparse.ArgumentParser()
ap.add_argument("--smoke", action="store_true")
# configs = comma list of "npz:band_h:closure_inputs" ; first is primary (cross-kind Case1)
ap.add_argument("--configs", type=str, default=(
    "case1_grid_48x80.npz:1:research/round_010/case1_closure_inputs.json:2800:CROSS-KIND Case1 wall-jet,"
    "case3_grid_64x192_full.npz:1:research/round_004/case3_closure_inputs.json:2800:IN-KIND Case3 hill"))
ap.add_argument("--families", type=str, default="diffusion,flow_matching")
ap.add_argument("--steps_train", type=int, default=18000, help="SAME budget as the frozen Case1/Case3 runs")
ap.add_argument("--batch", type=int, default=24)
ap.add_argument("--ch", type=int, default=64)
ap.add_argument("--L_eval", type=int, default=16)
ap.add_argument("--steps", type=int, default=32)
ap.add_argument("--g0_steps", type=int, default=64)
ap.add_argument("--g0_samp", type=int, default=256)
ap.add_argument("--sig_tau", type=float, default=0.03, help="std-space jitter on the tau_w channel (train+eval)")
ap.add_argument("--p_empty", type=float, default=0.30)
ap.add_argument("--frac", type=float, default=0.10, help="closure matching-height fraction (pi2=0.10 convention)")
ap.add_argument("--vm_lambda", type=float, default=0.5)
ap.add_argument("--vm_every", type=int, default=250)
ap.add_argument("--vm_n", type=int, default=16)
ap.add_argument("--vm_steps", type=int, default=6)
args, _ = ap.parse_known_args()
if args.smoke:
    (args.steps_train, args.batch, args.L_eval, args.steps, args.g0_steps, args.g0_samp,
     args.vm_every, args.vm_n, args.vm_steps) = (120, 8, 4, 8, 8, 16, 40, 6, 4)

dev = "cuda" if torch.cuda.is_available() else "cpu"
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); PROJ = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "closure"))
from wall_closure import parse_header, _shape_weights, predict_Cf, predict_tau_w  # inference-only closure
OUT = os.path.join(ROOT, "results"); os.makedirs(OUT, exist_ok=True)
SIG_MIN, SIG_MAX, RHO, SIGMA_DATA = 0.02, 80.0, 7.0, 1.0
P_MEAN, P_STD = -1.2, 1.2
ARMS = ["correct", "passthrough", "closure", "wrong_mean", "random", "no_wall"]
t0 = time.time()
def log(*a): print(*a, flush=True)

# ============================================================ CondUNet (tau_w-conditioned)
N_COND = 1 + 1 + 1 + 1   # [tau_val, keep(mask), obsfrac, domain]
def noise_embed(sig, dim=128):
    half = dim // 2; freqs = torch.exp(-math.log(10000) * torch.arange(half, device=sig.device) / half)
    a = sig.float().log().view(-1, 1) * freqs.view(1, -1); return torch.cat([a.sin(), a.cos()], -1)
class Block(nn.Module):
    def __init__(s, ci, co, ce):
        super().__init__(); s.n1 = nn.GroupNorm(8, ci); s.c1 = nn.Conv2d(ci, co, 3, padding=1)
        s.emb = nn.Linear(ce, co); s.n2 = nn.GroupNorm(8, co); s.c2 = nn.Conv2d(co, co, 3, padding=1)
        s.skip = nn.Conv2d(ci, co, 1) if ci != co else nn.Identity()
    def forward(s, x, e):
        hh = s.c1(F.silu(s.n1(x))); hh = hh + s.emb(e)[:, :, None, None]
        hh = s.c2(F.silu(s.n2(hh))); return hh + s.skip(x)
class CondUNet(nn.Module):
    def __init__(s, C, ch=64, ce=128):
        super().__init__()
        s.emb = nn.Sequential(nn.Linear(ce, ce), nn.SiLU(), nn.Linear(ce, ce))
        s.in_c = nn.Conv2d(C + N_COND, ch, 3, padding=1)
        s.d1 = Block(ch, ch, ce); s.d2 = Block(ch, 2*ch, ce); s.d3 = Block(2*ch, 2*ch, ce)
        s.mid = Block(2*ch, 2*ch, ce)
        s.u3 = Block(4*ch, 2*ch, ce); s.u2 = Block(4*ch, ch, ce); s.u1 = Block(2*ch, ch, ce)
        s.out = nn.Conv2d(ch, C, 3, padding=1); s.dn = nn.AvgPool2d(2); s.up = nn.Upsample(scale_factor=2, mode="nearest")
    def forward(s, xin, sig, cond):
        e = s.emb(noise_embed(sig)); h0 = s.in_c(torch.cat([xin, cond], 1)); h1 = s.d1(h0, e)
        h2 = s.d2(s.dn(h1), e); h3 = s.d3(s.dn(h2), e); m = s.mid(h3, e)
        u3 = s.u3(torch.cat([m, h3], 1), e); u2 = s.u2(torch.cat([s.up(u3), h2], 1), e)
        u1 = s.u1(torch.cat([s.up(u2), h1], 1), e); return s.out(u1)
def precond(sig):
    s2 = sig**2 + SIGMA_DATA**2; return (SIGMA_DATA**2/s2, sig*SIGMA_DATA/s2.sqrt(), 1.0/s2.sqrt())
def denoise(model, x, sig, cond):
    cs, co, ci = precond(sig)
    return cs.view(-1,1,1,1)*x + co.view(-1,1,1,1)*model(ci.view(-1,1,1,1)*x, sig, cond)
def edm_sigmas(steps):
    i = torch.arange(steps, device=dev)
    sig = (SIG_MAX**(1/RHO) + i/(steps-1)*(SIG_MIN**(1/RHO) - SIG_MAX**(1/RHO)))**RHO
    return torch.cat([sig, torch.zeros(1, device=dev)])


class Flow:
    """One separated-flow config: data, tau_w channel machinery, closure prediction, arms."""
    def __init__(s, npz, band_h, closure_inputs, re_h, label):
        s.band_h = band_h; s.closure_inputs = closure_inputs; s.re_h = re_h; s.label = label
        z = np.load(os.path.join(ROOT, "data", npz)); s.npz = npz
        s.fields = torch.tensor(np.asarray(z["fields"], dtype=np.float32)).to(dev)
        s.DM = torch.tensor(np.asarray(z["mask"], dtype=np.float32)).to(dev)
        s.wall_row = torch.tensor(np.asarray(z["wall_row"]).astype(np.int64)).to(dev)
        s.mu = torch.tensor(np.asarray(z["mu"], dtype=np.float32)); s.sd = torch.tensor(np.asarray(z["sd"], dtype=np.float32))
        s.h = float(z["h"]); s.Ly = float(z["Ly"]); s.Lx = float(z["Lx"]); s.gx = np.asarray(z["grid_x"], dtype=np.float32)
        s.T, s.C, s.H, s.W = s.fields.shape
        s.dy = s.Ly / s.H; s.dx = s.Lx / s.W
        s.DMb = (s.DM > 0.5); s.DMcol = s.DMb.float()
        log(f"[data] {npz} {tuple(s.fields.shape)} h={s.h:.5f} dy/h={s.dy/s.h:.4f} fluid_frac={float(s.DM.mean()):.3f}")
        # decorrelation tau + SAME split convention as the frozen runs
        Xmean_full = s.fields.mean(0, keepdim=True)
        Ef = ((s.fields - Xmean_full)[:, :, s.DMb]**2).mean(dim=(1, 2)); Ef = (Ef - Ef.mean()).detach().cpu().numpy()
        def itau(sig, maxlag=200):
            ac = np.array([1.0] + [float(np.corrcoef(sig[:-k], sig[k:])[0, 1]) for k in range(1, min(maxlag, len(sig)//2))])
            zc = np.argmax(ac < 0.0) if np.any(ac < 0.0) else len(ac)
            return float(max(1.0, 1.0 + 2.0 * ac[1:zc].sum()))
        s.tau = round(itau(Ef)); s.n_tr = int(0.76 * s.T); s.gap = max(80, 3 * s.tau)
        s.Xtr = s.fields[:s.n_tr]
        s.te_idx = torch.arange(s.n_tr + s.gap, s.T, device=dev)
        if args.smoke: s.te_idx = s.te_idx[:24]
        s.n_test = int(s.te_idx.numel()); s.Xte = s.fields[s.te_idx]
        s.block = max(1, 2 * s.tau); s.n_eff = s.n_test / (2.0 * s.tau)
        s.Xmean = s.Xtr.mean(0, keepdim=True)
        log(f"[tau] tau={s.tau} n_tr={s.n_tr} n_test={s.n_test} block={s.block} N_eff~{s.n_eff:.1f}")
        with torch.no_grad():
            s.var_dns = torch.stack([s.Xtr[:, 0].var(0), s.Xtr[:, 1].var(0)], 0).clamp_min(1e-4)
        s._build_tau()

    # ---- instantaneous & mean DNS wall stress tau_w(x,t); closure prediction; standardization ----
    def _build_tau(s):
        mu0 = float(s.mu[0]); sd0 = float(s.sd[0])
        Uphys = s.fields[:, 0] * sd0 + mu0                     # (T,H,W) physical streamwise velocity
        wr = s.wall_row; bh = s.band_h
        yb = (torch.arange(bh, device=dev) + 0.5) * s.dy       # wall-normal distances of band cells
        denom = float((yb * yb).sum())
        Ub = float((s.fields[:, 0].mean(0) * sd0 + mu0)[s.DMb].mean()); nu = Ub * s.h / s.re_h
        s.nu = nu; s.Ub = Ub
        # gather band cells per column: rows wr[j] .. wr[j]+bh-1  (all valid for these grids)
        valid = (wr + bh < s.H)
        tau_inst = torch.zeros(s.T, s.W, device=dev)
        for d in range(bh):
            rows = (wr + d).clamp(0, s.H - 1)                  # (W,)
            ub_d = Uphys[:, rows, torch.arange(s.W, device=dev)]   # (T,W) velocity at band cell d
            tau_inst = tau_inst + ub_d * float(yb[d])
        tau_inst = nu * tau_inst / denom                       # least-squares through-origin slope * nu
        tau_inst[:, ~valid] = 0.0
        s.tau_inst = tau_inst                                  # (T,W) physical wall stress
        s.tau_valid = valid                                    # (W,)
        s.tau_mean = tau_inst[:s.n_tr].mean(0)                 # (W,) data-climatology wall stress
        vt = tau_inst[:s.n_tr][:, valid]
        s.tau_mu = float(vt.mean()); s.tau_sd = float(vt.std() + 1e-8)
        # ---- closure (a-priori) tau_w per column via reused familyclass DeepONet ----
        rep = json.load(open(os.path.join(PROJ, s.closure_inputs)))
        Wh = _shape_weights(parse_header())
        feats, Ue_st, st_x = [], [], []
        for st in rep["stations"]:
            hh = [q for q in st["heights"] if abs(q["frac"] - args.frac) < 1e-6]
            if not hh or st.get("Lambda") is None: continue
            feats.append([hh[0]["pi1"], hh[0]["pi2"], st["Pi_p"], st["Lambda"]]); Ue_st.append(st["Ue"]); st_x.append(st["x_over_h"])
        feats = np.array(feats); Ue_st = np.array(Ue_st); st_x = np.array(st_x)
        order = np.argsort(st_x); st_x, feats, Ue_st = st_x[order], feats[order], Ue_st[order]
        Cf_cl_st = predict_Cf(feats, Wh); tau_cl_st = predict_tau_w(feats, Ue_st, Wh)
        xoh_col = (s.gx - s.gx.min()) / s.h
        in_rng = (xoh_col >= st_x.min()) & (xoh_col <= st_x.max())
        tau_cl = np.interp(xoh_col, st_x, tau_cl_st)
        Cf_cl = np.interp(xoh_col, st_x, Cf_cl_st)
        s.tau_closure = torch.tensor(tau_cl, dtype=torch.float32, device=dev)     # (W,) physical
        # transfer metrics vs DNS mean tau_w (measured, reported straight)
        tdns = s.tau_mean.detach().cpu().numpy(); vnp = valid.detach().cpu().numpy()
        Ue_col = np.full(s.W, np.nan)
        Umean_phys = (s.fields[:, 0].mean(0) * sd0 + mu0).detach().cpu().numpy()
        wr_np = wr.detach().cpu().numpy()
        for j in range(s.W):
            col = Umean_phys[wr_np[j]:, j]; Ue_col[j] = np.max(np.abs(col)) if col.size else np.nan
        good = vnp & np.isfinite(Ue_col) & in_rng
        Cf_dns = 2 * tdns / (Ue_col**2 + 1e-12)
        s.transfer = dict(
            corr_tau=float(np.corrcoef(tdns[good], tau_cl[good])[0, 1]),
            corr_Cf=float(np.corrcoef(Cf_dns[good], Cf_cl[good])[0, 1]),
            sign_agree=float(np.mean(np.sign(tdns[good]) == np.sign(tau_cl[good]))),
            relRMSE_tau=float(np.sqrt(np.mean((tau_cl[good]-tdns[good])**2))/(np.sqrt(np.mean(tdns[good]**2))+1e-12)),
            n_neg_closure=int((tau_cl[good] < 0).sum()), n_neg_dns=int((tdns[good] < 0).sum()),
            n_good=int(good.sum()), nu=float(nu), Ub=float(Ub), tau_mu=s.tau_mu, tau_sd=s.tau_sd,
            tau_dns_mean=float(tdns[good].mean()), tau_closure_mean=float(tau_cl[good].mean()),
            in_range_frac=float(in_rng.mean()))
        s.in_rng = torch.tensor(in_rng, device=dev)
        log(f"[closure] {s.label} nu={nu:.3e} Ub={Ub:.4f} | transfer corr(tau)={s.transfer['corr_tau']:+.3f} "
            f"sign-agree={s.transfer['sign_agree']:.2f} relRMSE={s.transfer['relRMSE_tau']:.3f} "
            f"n_neg cl/dns={s.transfer['n_neg_closure']}/{s.transfer['n_neg_dns']} in_rng={s.transfer['in_range_frac']:.2f}")

    def std_tau(s, tau_col):    # (.,W) physical -> standardized, zeroed where invalid/out-of-range
        z = (tau_col - s.tau_mu) / s.tau_sd
        m = (s.tau_valid & s.in_rng).float()
        return z * m

    def band_keepmask(s, shift=0):
        wr = torch.roll(s.wall_row, shift, 0); dmask = torch.roll(s.DMb, shift, 1)
        rows = torch.arange(s.H, device=dev)[:, None]
        band = (rows >= wr[None, :]) & (rows < (wr + s.band_h)[None, :]) & dmask
        return band.float()

    def make_cond(s, tau_col_std, keep, jitter=True, seed=None):
        """tau_col_std: (n,W) standardized wall stress; keep: (H,W) band mask. -> (n,N_COND,H,W)."""
        n = tau_col_std.shape[0]; k = keep[None, None]
        tval = tau_col_std[:, None, None, :].expand(n, 1, s.H, s.W) * k
        if jitter and args.sig_tau > 0:
            g = None if seed is None else torch.Generator(device=dev).manual_seed(seed)
            noise = torch.randn(tval.shape, device=dev, generator=g)
            tval = tval + args.sig_tau * noise * k
        of = float(keep.sum() / max(1.0, s.DMcol.sum()))
        ofc = torch.full((n, 1, s.H, s.W), of, device=dev)
        kc = k.expand(n, 1, s.H, s.W); dmc = s.DMcol[None, None].expand(n, 1, s.H, s.W)
        return torch.cat([tval, kc, ofc, dmc], 1)

    def empty_cond(s, n):
        z = torch.zeros(n, 1, s.H, s.W, device=dev)
        dmc = s.DMcol[None, None].expand(n, 1, s.H, s.W)
        return torch.cat([z, z, z, dmc], 1)


# ============================================================ samplers (cond passed in, decoupled)
@torch.no_grad()
def sample(model, cond, family, H, W, C, DMcol, L, steps, chunk=128):
    gen = torch.Generator(device=dev).manual_seed(777); n = cond.shape[0]
    cond_all = cond.repeat_interleave(L, 0); NB = n*L; dm = DMcol[None, None]
    if family == "diffusion":
        x_all = torch.randn(NB, C, H, W, device=dev, generator=gen) * SIG_MAX; sig = edm_sigmas(steps)
    else:
        x_all = torch.randn(NB, C, H, W, device=dev, generator=gen); dt = 1.0/steps
    for b0 in range(0, NB, chunk):
        b1 = min(b0+chunk, NB); x = x_all[b0:b1]*dm; cond_b = cond_all[b0:b1]; m = b1-b0
        if family == "diffusion":
            for j in range(steps):
                sg = sig[j].expand(m); D = denoise(model, x, sg, cond_b); dxu = (x - D)/sig[j]
                x = (x + (sig[j+1]-sig[j])*dxu) * dm
        else:
            for j in range(steps):
                t = torch.full((m,), 1.0 - j*dt, device=dev).clamp(1e-3, 1.0)
                v = model(x, t, cond_b); x = (x - dt*v) * dm
        x_all[b0:b1] = x
    return x_all.reshape(n, L, C, H, W)

# ============================================================ variance-match short samplers (grad)
def vm_sample(family, model, fl, nsmp, steps, gen):
    cond = fl.empty_cond(nsmp); dm = fl.DMcol[None, None]
    if family == "diffusion":
        sig = edm_sigmas(steps); x = torch.randn(nsmp, fl.C, fl.H, fl.W, device=dev, generator=gen)*SIG_MAX*dm
        for j in range(steps):
            sg = sig[j].expand(nsmp); D = denoise(model, x, sg, cond); dxu = (x - D)/sig[j]
            x = (x + (sig[j+1]-sig[j])*dxu) * dm
    else:
        dt = 1.0/steps; x = torch.randn(nsmp, fl.C, fl.H, fl.W, device=dev, generator=gen)*dm
        for j in range(steps):
            t = torch.full((nsmp,), 1.0 - j*dt, device=dev).clamp(1e-3, 1.0)
            v = model(x, t, cond); x = (x - dt*v) * dm
    return x
def variance_match_loss(family, model, fl, gen):
    xg = vm_sample(family, model, fl, args.vm_n, args.vm_steps, gen)
    vg = xg.var(0).clamp_min(1e-6); m = fl.DMb[None].expand_as(vg)
    d = torch.log(vg) - torch.log(fl.var_dns)
    return (d[m]**2).mean()

# ============================================================ training (A1, tau_w-conditioned, SAME budget)
def train(family, fl, steps_train, seed=1234):
    net = CondUNet(fl.C, ch=args.ch).to(dev); ema = CondUNet(fl.C, ch=args.ch).to(dev); ema.load_state_dict(net.state_dict())
    n_par = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=2e-4, betas=(0.9, 0.99)); net.train()
    gen = torch.Generator(device=dev).manual_seed(seed); vmgen = torch.Generator(device=dev).manual_seed(seed + 7)
    DMloss = fl.DMcol[None, None]; vm_hist = []
    tau_inst_tr = fl.tau_inst[:fl.n_tr]
    for it in range(steps_train):
        idx = torch.randint(0, fl.n_tr, (args.batch,), generator=gen, device=dev); x = fl.Xtr[idx]
        sh = int(torch.randint(0, fl.W, (1,), generator=gen, device=dev)); x = torch.roll(x, sh, dims=3)
        r = torch.rand(1, generator=gen, device=dev).item()
        if r < args.p_empty:
            cond = fl.empty_cond(x.shape[0])
        else:
            keep = fl.band_keepmask(sh)
            tau_col = torch.roll(fl.std_tau(tau_inst_tr[idx]), sh, dims=1)   # (batch,W) standardized, rolled
            cond = fl.make_cond(tau_col, keep, jitter=True, seed=None)
        if family == "diffusion":
            lnsig = P_MEAN + P_STD*torch.randn(x.shape[0], device=dev, generator=gen); sig = lnsig.exp()
            xn = x + torch.randn_like(x)*sig.view(-1,1,1,1)
            w = (sig**2 + SIGMA_DATA**2)/((sig*SIGMA_DATA)**2)
            D = denoise(net, xn, sig, cond)
            loss = (w.view(-1,1,1,1)*DMloss*(D - x)**2).sum() / (DMloss.sum()*x.shape[0]*fl.C)
        elif family == "flow_matching":
            t = torch.rand(x.shape[0], device=dev, generator=gen).clamp(1e-3, 1.0); eps = torch.randn_like(x)
            xt = (1 - t).view(-1,1,1,1)*x + t.view(-1,1,1,1)*eps
            v = net(xt, t, cond)
            loss = (DMloss*(v - (eps - x))**2).sum() / (DMloss.sum()*x.shape[0]*fl.C)
        else:
            raise ValueError(family)
        vm_val = 0.0
        if args.vm_lambda > 0 and (it+1) % args.vm_every == 0 and (it+1) > 0.4*steps_train:
            vml = variance_match_loss(family, net, fl, vmgen); loss = loss + args.vm_lambda*vml
            vm_val = float(vml.detach()); vm_hist.append(vm_val)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
        with torch.no_grad():
            for pe, pn in zip(ema.parameters(), net.parameters()): pe.mul_(0.9995).add_(pn, alpha=0.0005)
            for be, bn in zip(ema.buffers(), net.buffers()): be.copy_(bn)
        if (it+1) % max(1, steps_train//10) == 0:
            log(f"[train:{family}] it {it+1}/{steps_train} loss={loss.item():.4f} vm={vm_val:.3f} wall={time.time()-t0:.0f}s")
    ema.eval(); return ema, n_par, (float(np.mean(vm_hist[-5:])) if vm_hist else None)

# ============================================================ metrics / bootstrap (identical machinery)
def r2_pooled(pred, truth, region):
    m = region[None, None].expand_as(truth); p = pred[m]; t = truth[m]
    return float(1 - ((p-t)**2).sum()/(((t-t.mean())**2).sum()+1e-12))
def r2_components(pred, truth, region, tmean):
    p = pred[:, :, region]; t = truth[:, :, region]
    return (((p-t)**2).sum(dim=(1,2))).detach().cpu().numpy(), (((t-tmean)**2).sum(dim=(1,2))).detach().cpu().numpy()
def block_boot_idx(n, block, B, rng):
    nbl = int(np.ceil(n/block)); starts = rng.integers(0, n, size=(B, nbl)); off = np.arange(block)[None, None, :]
    return ((starts[:, :, None] + off).reshape(B, -1) % n)[:, :n]
def pooled_r2_from_comp(sse, sst, idx): return 1.0 - sse[idx].sum(1)/(sst[idx].sum(1)+1e-12)

def crps_coverage(ens, truth, region):
    n, L = ens.shape[0], ens.shape[1]
    e = ens[:, :, :, region]; y = truth[:, :, region]
    term1 = (e - y[:, None]).abs().mean(dim=1)
    pair = torch.zeros_like(term1)
    for i in range(L): pair = pair + (e[:, i:i+1] - e).abs().sum(dim=1)
    pair = pair / (L * L)
    crps = (term1 - 0.5*pair).mean().item()
    mean = e.mean(dim=1); crps_point = (mean - y).abs().mean().item()
    q05 = torch.quantile(e, 0.05, dim=1); q95 = torch.quantile(e, 0.95, dim=1)
    cov90 = (((y >= q05) & (y <= q95)).float().mean()).item(); sharp = ((q95 - q05).mean()).item()
    return dict(CRPS_ensemble=crps, CRPS_point=crps_point, crps_skill_vs_point=float(1 - crps/(crps_point+1e-12)),
                coverage90=cov90, sharpness90=sharp, L=int(L))

def run_arms(family, model, fl):
    keep = fl.band_keepmask(0); obs = keep > 0.5; unobs = fl.DMb & (~obs)
    Xte = fl.Xte; Xmean = fl.Xmean; Xte_f = Xte - Xmean; n_test = fl.n_test
    tau_inst_te = fl.tau_inst[fl.te_idx]                              # (n_test,W) physical
    # arms as standardized per-column tau vectors (n_test,W)
    correct = fl.std_tau(tau_inst_te)
    passth = fl.std_tau(fl.tau_mean[None].expand(n_test, fl.W))
    closure = fl.std_tau(fl.tau_closure[None].expand(n_test, fl.W))
    flat_val = fl.tau_mean[fl.tau_valid & fl.in_rng].mean()
    wrongm = fl.std_tau(flat_val.expand(n_test, fl.W))
    g = torch.Generator(device=dev).manual_seed(9); perm = torch.randperm(fl.W, generator=g, device=dev)
    random_ = correct[:, perm]                                       # shuffle columns (right magnitude, wrong x)
    arm_tau = dict(correct=correct, passthrough=passth, closure=closure, wrong_mean=wrongm, random=random_)
    tmean_t = Xte[:, :, unobs].mean(); tmean_f = Xte_f[:, :, unobs].mean()
    res = {}; comp_t = {}; comp_f = {}; fields_out = {}; meanfield = {}; uq = {}
    for name in ARMS:
        if name == "no_wall":
            cond = fl.empty_cond(n_test)
        else:
            cond = fl.make_cond(arm_tau[name], keep, jitter=True, seed=4242)
        ens = sample(model, cond, family, fl.H, fl.W, fl.C, fl.DMcol, fl.L_eval, args.steps)
        pm = ens.mean(1); pm_f = pm - Xmean
        comp_t[name] = r2_components(pm, Xte, unobs, tmean_t); comp_f[name] = r2_components(pm_f, Xte_f, unobs, tmean_f)
        gmean = pm.mean(0); dmean = Xte.mean(0)
        res[name] = dict(R2_total=r2_pooled(pm, Xte, unobs), R2_fluct=r2_pooled(pm_f, Xte_f, unobs),
                         R2_meanfield=r2_pooled(gmean[None], dmean[None], unobs),
                         rmse_unobs=float(((pm-Xte)[:, :, unobs]**2).mean().sqrt()))
        if name in ("closure", "correct", "no_wall", "passthrough", "wrong_mean"):
            uq[name] = crps_coverage(ens, Xte, unobs)
        fields_out[name] = pm[0].detach().cpu().numpy(); meanfield[name] = pm.mean(0).detach().cpu().numpy()
        del ens; torch.cuda.empty_cache() if dev == "cuda" else None
        log(f"[{family}:{name:11s}] R2_total={res[name]['R2_total']:+.4f} R2_fluct={res[name]['R2_fluct']:+.4f} "
            f"R2_mean={res[name]['R2_meanfield']:+.4f}" + (f" | CRPS={uq[name]['CRPS_ensemble']:.4f}"
            f"(pt {uq[name]['CRPS_point']:.4f}) cov90={uq[name]['coverage90']:.2f}" if name in uq else ""))
    return dict(res=res, comp_t=comp_t, comp_f=comp_f, fields=fields_out, meanfield=meanfield, uq=uq, obs=obs, unobs=unobs)

def error_hiding(family, model, fl):
    keep = fl.band_keepmask(0); obs = keep > 0.5; unobs = fl.DMb & (~obs)
    tau_inst_te = fl.tau_inst[fl.te_idx]
    condC = fl.make_cond(fl.std_tau(fl.tau_closure[None].expand(fl.n_test, fl.W)), keep, jitter=True, seed=4242)
    condN = fl.empty_cond(fl.n_test)
    pmC = sample(model, condC, family, fl.H, fl.W, fl.C, fl.DMcol, fl.L_eval, args.steps).mean(1)
    pmN = sample(model, condN, family, fl.H, fl.W, fl.C, fl.DMcol, fl.L_eval, args.steps).mean(1)
    dist_h = (torch.arange(fl.H, device=dev)[:, None] - fl.wall_row[None, :]).float() * fl.dy / fl.h
    near = unobs & (dist_h <= 0.25); outer = unobs & (dist_h >= 0.75)
    r2 = lambda pred, reg: r2_pooled(pred, fl.Xte, reg)
    rmse = lambda pred, reg: float(((pred-fl.Xte)[:, :, reg]**2).mean().sqrt())
    if int(near.sum()) < 20 or int(outer.sum()) < 20:
        return dict(note="insufficient near/outer cells", n_near=int(near.sum()), n_outer=int(outer.sum()))
    adv = lambda reg: r2(pmC, reg) - r2(pmN, reg)
    return dict(R2_adv_nearwall=adv(near), R2_adv_outer=adv(outer), R2_adv_whole=adv(unobs),
                RMSE_red_nearwall=rmse(pmN, near) - rmse(pmC, near), RMSE_red_outer=rmse(pmN, outer) - rmse(pmC, outer),
                n_near=int(near.sum()), n_outer=int(outer.sum()))

def deltas_and_phi(comp, fl, B=4000):
    rng = np.random.default_rng(0); idx = block_boot_idx(fl.n_test, fl.block, B, rng)
    r2b = {k: pooled_r2_from_comp(comp[k][0], comp[k][1], idx) for k in ARMS}
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    out = {"_block": fl.block, "_B": B, "_n_eff": float(fl.n_eff)}
    for k in ARMS: out[k] = dict(mean=float(r2b[k].mean()), ci95=ci(r2b[k]))
    for base in ["no_wall", "random", "wrong_mean", "passthrough"]:
        d = r2b["closure"] - r2b[base]; out[f"d_closure_minus_{base}"] = dict(mean=float(d.mean()), ci95=ci(d), ci_pos=bool(ci(d)[0] > 0))
    for base in ["no_wall", "random"]:
        d = r2b["passthrough"] - r2b[base]; out[f"d_passthrough_minus_{base}"] = dict(mean=float(d.mean()), ci95=ci(d), ci_pos=bool(ci(d)[0] > 0))
    d = r2b["correct"] - r2b["no_wall"]; out["d_correct_minus_no_wall"] = dict(mean=float(d.mean()), ci95=ci(d), ci_pos=bool(ci(d)[0] > 0))
    num = r2b["closure"] - r2b["no_wall"]
    finite = lambda a: a[np.isfinite(a)]
    for nm, den in [("phi_oracle", r2b["correct"] - r2b["no_wall"]),
                    ("phi_passthrough", r2b["passthrough"] - r2b["no_wall"]),
                    ("phi_wrongmean", r2b["wrong_mean"] - r2b["no_wall"])]:
        ph = num / np.where(np.abs(den) < 1e-6, np.nan, den)
        out[nm] = dict(mean=float(np.nanmean(ph)), ci95=ci(finite(ph)))
    return out

@torch.no_grad()
def g0_adequacy(family, model, fl):
    cond = fl.empty_cond(args.g0_samp)
    gen = sample(model, cond, family, fl.H, fl.W, fl.C, fl.DMcol, 1, args.g0_steps)[:, 0]
    m = fl.DMb
    def stats(fld):
        U = fld[:, 0].mean(0); uu = fld[:, 0].var(0); vv = fld[:, 1].var(0)
        row = fl.H//2; strip = fld[:, 0, row, :]
        Fk = torch.fft.rfft(strip - strip.mean(1, keepdim=True), dim=1); Ek = (Fk.real**2 + Fk.imag**2).mean(0)
        return U, uu, vv, Ek
    Ug, uug, vvg, Ekg = stats(gen); Ud, uud, vvd, Ekd = stats(fl.fields)
    prof_r2 = lambda a, b: float(1 - ((a-b)[m]**2).sum()/(((b-b[m].mean())[m])**2).sum())
    ks = torch.arange(2, min(40, Ekg.shape[0]))
    slope = lambda E: float(np.polyfit(np.log(ks.cpu().numpy()), np.log(E[ks].cpu().numpy()+1e-20), 1)[0])
    out = dict(meanU_R2=prof_r2(Ug, Ud), uu_R2=prof_r2(uug, uud), vv_R2=prof_r2(vvg, vvd),
               uu_ratio=float(uug[m].mean()/uud[m].mean()), spectrum_slope_gen=slope(Ekg), spectrum_slope_dns=slope(Ekd))
    log(f"[G0:{family}] meanU_R2={out['meanU_R2']:.3f} uu_R2={out['uu_R2']:.3f} uu_ratio={out['uu_ratio']:.2f} "
        f"slope g/d={out['spectrum_slope_gen']:.2f}/{out['spectrum_slope_dns']:.2f}")
    return out

# ============================================================ per-config driver
def run_config(npz, band_h, closure_inputs, re_h, label, fams):
    fl = Flow(npz, band_h, closure_inputs, re_h, label); fl.L_eval = args.L_eval; fl.p_empty = args.p_empty
    tag = npz.split("_")[0] + f"_bh{band_h}"
    log(f"\n[==== CONFIG {tag} | {label} | {fl.H}x{fl.W} n_test={fl.n_test} N_eff~{fl.n_eff:.1f} ====]")
    per = {}; fields_save = {}; n_par_ref = None
    for fam in fams:
        log(f"\n[---- {fam} ---- {tag}] training A1 (tau_w-conditioned), same budget ...")
        mA1, n_par, vm1 = train(fam, fl, args.steps_train, seed=1234); n_par_ref = n_par
        torch.save(dict(ema=mA1.state_dict(), n_par=n_par, family=fam, prior="A1_wallstress", data=npz, band_h=band_h),
                   os.path.join(OUT, f"wscond_{tag}_{fam}_A1.pt"))
        A = run_arms(fam, mA1, fl)
        boot_t = deltas_and_phi(A["comp_t"], fl); boot_f = deltas_and_phi(A["comp_f"], fl)
        g0 = g0_adequacy(fam, mA1, fl); eh = error_hiding(fam, mA1, fl)
        gates = dict(
            G_passthrough_beats_no_wall_total=boot_t["d_passthrough_minus_no_wall"]["ci_pos"],     # THE repair gate
            G_closure_beats_no_wall_total=boot_t["d_closure_minus_no_wall"]["ci_pos"],
            G_closure_beats_no_wall_fluct=boot_f["d_closure_minus_no_wall"]["ci_pos"],
            G_closure_beats_wrong_mean_total=boot_t["d_closure_minus_wrong_mean"]["ci_pos"],
            G_closure_beats_random_total=boot_t["d_closure_minus_random"]["ci_pos"],
            G_closure_ge_passthrough_total=bool(boot_t["closure"]["mean"] >= boot_t["passthrough"]["mean"] - 0.002),
            G_correct_beats_no_wall_total=boot_t["d_correct_minus_no_wall"]["ci_pos"],
            G_order_correct_ge_passthrough_ge_closure_total=bool(
                boot_t["correct"]["mean"] >= boot_t["passthrough"]["mean"] >= boot_t["closure"]["mean"] - 0.003),
            G_phi_passthrough_pos_total=bool(boot_t["phi_passthrough"]["ci95"][0] > 0),
            G_crps_beats_point=bool(A["uq"]["closure"]["CRPS_ensemble"] < A["uq"]["closure"]["CRPS_point"]))
        per[fam] = dict(n_par=n_par, vm_final_A1=vm1, arms=A["res"], uq=A["uq"], G0=g0, error_hiding=eh,
                        boot_total=boot_t, boot_fluct=boot_f, gates=gates)
        fields_save[fam] = dict(fields=A["fields"], meanfield=A["meanfield"])
        log(f"[phi:{fam}] TOTAL phi_passthrough={boot_t['phi_passthrough']['mean']:+.3f} CI{boot_t['phi_passthrough']['ci95']} "
            f"| d_pass-nowall={boot_t['d_passthrough_minus_no_wall']['mean']:+.4f} pos={boot_t['d_passthrough_minus_no_wall']['ci_pos']} "
            f"| d_clos-nowall={boot_t['d_closure_minus_no_wall']['mean']:+.4f} pos={boot_t['d_closure_minus_no_wall']['ci_pos']}")
        log(f"[gates:{fam}] {json.dumps(gates)}")
        _write(tag, npz, band_h, re_h, label, fl, per, fields_save, n_par_ref)
    return tag

def _write(tag, npz, band_h, re_h, label, fl, per, fields_save, n_par_ref):
    out = dict(_meta=dict(script="eval_wallstress_cond.py", node="development/node_003 L3",
                          conditioning="DIRECT wall-stress tau_w feature (no equilibrium bridge; not solver-coupled)", dev=str(dev),
                          evidence_level="closure-conditioned offline reconstruction / DNS-mean-input diagnostic",
                          data=npz, tag=tag, regime=label, T=fl.T, H=fl.H, W=fl.W, C=fl.C, n_test=fl.n_test,
                          tau=fl.tau, block=fl.block, n_eff=float(fl.n_eff), band_h=band_h, L_eval=args.L_eval,
                          steps=args.steps, steps_train=args.steps_train, arms=ARMS, families=list(per.keys()),
                          n_par=n_par_ref, dy_over_h=fl.dy/fl.h, re_h=re_h, vm_lambda=args.vm_lambda,
                          sig_tau=args.sig_tau, closure_inputs=fl.closure_inputs),
               closure=fl.transfer, families=per)
    npzp = os.path.join(OUT, f"wallstress_cond_{tag}_fields.npz")
    save_kw = dict(mu=fl.mu.numpy(), sd=fl.sd.numpy(), dy_over_h=fl.dy/fl.h, domain=fl.DMcol.cpu().numpy(),
                   truth0=fl.Xte[0].detach().cpu().numpy(), truthmean=fl.Xte.mean(0).detach().cpu().numpy(),
                   wall_row=fl.wall_row.cpu().numpy(), tau_mean=fl.tau_mean.cpu().numpy(),
                   tau_closure=fl.tau_closure.cpu().numpy())
    for fam in fields_save:
        for k, v in fields_save[fam]["fields"].items(): save_kw[f"{fam}_field_{k}"] = v
        for k, v in fields_save[fam]["meanfield"].items(): save_kw[f"{fam}_mean_{k}"] = v
    np.savez_compressed(npzp, **save_kw)
    path = os.path.join(OUT, f"wallstress_cond_{tag}_results.json")
    txt = json.dumps(out, indent=2, sort_keys=True, default=float); open(path, "w").write(txt)
    log(f"[out] wrote {path} md5={hashlib.md5(txt.encode()).hexdigest()}")

def main():
    torch.manual_seed(0); np.random.seed(0)
    fams = [f.strip() for f in args.families.split(",") if f.strip()]
    cfgs = []
    for c in args.configs.split(","):
        parts = c.split(":"); cfgs.append((parts[0], int(parts[1]), parts[2], float(parts[3]), parts[4]))
    log(f"[run] === L3 DIRECT WALL-STRESS CONDITIONING | families={fams} | {len(cfgs)} configs ===")
    for npz, band_h, ci_path, re_h, label in cfgs:
        try:
            run_config(npz, band_h, ci_path, re_h, label, fams)
        except Exception as e:
            import traceback; log(f"[ERROR] config {npz} failed: {e}\n{traceback.format_exc()}")
    log("=== done ===")

if __name__ == "__main__":
    main()
