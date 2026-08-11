#!/usr/bin/env python3
"""
gen_cube_mesh.py -- genuinely-3D wall-mounted CUBE-ARRAY hex mesh (.re2) for
nekRS wall-resolved LES.  Coceal-2006 aligned cubes: h=1, pitch 2h,
lambda_p=0.25, one cube per periodic cell.

Domain (Coceal-exact; matches OpenFOAM RANS blockMeshDict of
repeating_structure_wall_model/cube_array_prod):
    Lx=2, Ly=4, Lz=2 ;  cube = [0.5,1.5] x [0,1] x [0.5,1.5]  SOLID (removed).
    x,z periodic ; floor y=0, top y=4, cube faces = no-slip walls.

WALL-RESOLVED grading: element edges coincide with the cube faces (so spectral
GLL points cluster ON the walls), AND elements are geometrically CLUSTERED
toward every wall -- floor, cube top, and the cube SIDE/FRONT/BACK faces (the
v1 uniform side-slabs left the cube faces at y+~7; here they are clustered to
first-element ~h_wall so the first GLL point sits at y+~1-2).  y is
double-graded in the canopy (floor + cube-top) and graded above the canopy.

BCs by the NEIGHBOUR rule; periodic partners are exact structured translates.
Writes cube.re2 via pymech and self-validates + prints a y+ estimate.
"""
import sys, numpy as np
from pymech.core import HexaData
from pymech.neksuite import writere2, readre2

# ---------------- resolution ----------------
NX_SIDE = 6     # elements over [0,0.5] and [1.5,2]  (clustered to cube face)
NX_MID  = 16    # elements over [0.5,1.5] (cube span, double-sided)
NZ_SIDE = 6
NZ_MID  = 16
NY_CAN  = 12    # canopy [0,1]  (double-sided: floor + cube top)
NY_ABV  = 10    # above  [1,4]  (graded fine at y=1)
# Wall first-element thickness. Y-normal walls (floor + cube top) carry the
# HIGH skin friction (attached/accelerating BL near the leading edge) -> fine.
# X/Z-normal cube faces (front stagnation, sides/back recirculation) carry LOW
# skin friction and their separation is EDGE-fixed (sharp cube) -> gentler.
H_WALL_Y  = 0.008
H_WALL_XZ = 0.012

H = 1.0
X_CUTS = [0.0, 0.5, 1.5, 2.0]
Z_CUTS = [0.0, 0.5, 1.5, 2.0]
Y_CANOPY, Y_TOP = 1.0, 4.0

# ---------------- geometric node helpers ----------------
def _ratio_for_first(L, n, s):
    if n * s >= L:
        return 1.0
    lo, hi = 1.0000001, 4.0
    for _ in range(100):
        g = 0.5 * (lo + hi)
        span = s * (g**n - 1.0) / (g - 1.0)
        if span > L: hi = g
        else: lo = g
    return 0.5 * (lo + hi)

def geom_fine_lo(x0, x1, n, s):
    g = _ratio_for_first(x1 - x0, n, s)
    if g == 1.0:
        return np.linspace(x0, x1, n + 1)
    sizes = s * g**np.arange(n)
    sizes *= (x1 - x0) / sizes.sum()
    return x0 + np.concatenate([[0.0], np.cumsum(sizes)])

def geom_fine_hi(x0, x1, n, s):
    return x1 - geom_fine_lo(0.0, x1 - x0, n, s)[::-1]

def geom_double(x0, x1, n, s):
    assert n % 2 == 0
    xm = 0.5 * (x0 + x1)
    lo = geom_fine_lo(x0, xm, n // 2, s)
    hi = geom_fine_hi(xm, x1, n // 2, s)
    return np.concatenate([lo[:-1], hi])

xn = np.concatenate([
    geom_fine_hi(X_CUTS[0], X_CUTS[1], NX_SIDE, H_WALL_XZ)[:-1],
    geom_double(X_CUTS[1], X_CUTS[2], NX_MID, H_WALL_XZ)[:-1],
    geom_fine_lo(X_CUTS[2], X_CUTS[3], NX_SIDE, H_WALL_XZ),
])
zn = np.concatenate([
    geom_fine_hi(Z_CUTS[0], Z_CUTS[1], NZ_SIDE, H_WALL_XZ)[:-1],
    geom_double(Z_CUTS[1], Z_CUTS[2], NZ_MID, H_WALL_XZ)[:-1],
    geom_fine_lo(Z_CUTS[2], Z_CUTS[3], NZ_SIDE, H_WALL_XZ),
])
y_can = geom_double(0.0, Y_CANOPY, NY_CAN, H_WALL_Y)
y_abv = geom_fine_lo(Y_CANOPY, Y_TOP, NY_ABV, H_WALL_Y)
yn = np.concatenate([y_can[:-1], y_abv])

NEX, NEZ, NEY = len(xn) - 1, len(zn) - 1, len(yn) - 1
IX0, IX1 = NX_SIDE, NX_SIDE + NX_MID
IZ0, IZ1 = NZ_SIDE, NZ_SIDE + NZ_MID
IY0, IY1 = 0, NY_CAN

def in_cube(ix, iy, iz):
    return (IX0 <= ix < IX1) and (IY0 <= iy < IY1) and (IZ0 <= iz < IZ1)

idx_of = {}; elems = []
for iy in range(NEY):
    for iz in range(NEZ):
        for ix in range(NEX):
            if in_cube(ix, iy, iz):
                continue
            idx_of[(ix, iy, iz)] = len(elems); elems.append((ix, iy, iz))
E = len(elems)
print(f"[mesh] grid {NEX}x{NEY}x{NEZ}={NEX*NEY*NEZ} minus cube "
      f"{NX_MID*NY_CAN*NZ_MID} => E={E} elements  (~{E*512/1e6:.2f}M pts at p=7)")

data = HexaData(ndim=3, nel=E, lr1=(2, 2, 2), var=(3, 0, 0, 0, 0), nbc=1)
NEKFACE = [1, 2, 3, 4, 5, 6]

def set_pos(el, x0, x1, y0, y1, z0, z1):
    for iz in (0, 1):
        for iy in (0, 1):
            for ix in (0, 1):
                el.pos[0, iz, iy, ix] = x1 if ix else x0
                el.pos[1, iz, iy, ix] = y1 if iy else y0
                el.pos[2, iz, iy, ix] = z1 if iz else z0

nW = nP = 0
for gid, (ix, iy, iz) in enumerate(elems):
    el = data.elem[gid]
    set_pos(el, xn[ix], xn[ix+1], yn[iy], yn[iy+1], zn[iz], zn[iz+1])
    iel = gid + 1
    def bc(fidx, typ, c1=0.0, c2=0.0):
        el.bcs[0][fidx] = (typ, iel, NEKFACE[fidx], float(c1), float(c2), 0.0, 0.0, 0.0)
    if ix == 0:
        bc(3, 'P', idx_of[(NEX-1, iy, iz)] + 1, NEKFACE[1]); nP += 1
    elif in_cube(ix-1, iy, iz):
        bc(3, 'W'); nW += 1
    if ix == NEX-1:
        bc(1, 'P', idx_of[(0, iy, iz)] + 1, NEKFACE[3]); nP += 1
    elif in_cube(ix+1, iy, iz):
        bc(1, 'W'); nW += 1
    if iz == 0:
        bc(4, 'P', idx_of[(ix, iy, NEZ-1)] + 1, NEKFACE[5]); nP += 1
    elif in_cube(ix, iy, iz-1):
        bc(4, 'W'); nW += 1
    if iz == NEZ-1:
        bc(5, 'P', idx_of[(ix, iy, 0)] + 1, NEKFACE[4]); nP += 1
    elif in_cube(ix, iy, iz+1):
        bc(5, 'W'); nW += 1
    if iy == 0 or in_cube(ix, iy-1, iz):
        bc(0, 'W'); nW += 1
    if iy == NEY-1 or in_cube(ix, iy+1, iz):
        bc(2, 'W'); nW += 1

print(f"[mesh] BC faces: W={nW}  P={nP}")
data.lr1 = [2, 2, data.ndim - 1]

OUT = "cube.re2"; writere2(OUT, data); print(f"[mesh] wrote {OUT}")

# ---------------- self-validation + y+ estimate ----------------
m = readre2(OUT)
xs = np.array([e.pos[0] for e in m.elem]); ys = np.array([e.pos[1] for e in m.elem]); zs = np.array([e.pos[2] for e in m.elem])
print(f"[check] nel={m.nel} bbox x[{xs.min():.3f},{xs.max():.3f}] y[{ys.min():.3f},{ys.max():.3f}] z[{zs.min():.3f},{zs.max():.3f}]")
badpair = 0; nWc = nPc = 0
for gid, e in enumerate(m.elem):
    for f in range(6):
        b = e.bcs[0][f]
        if b[0] == 'P':
            nPc += 1; j = int(b[3]) - 1; jf = int(b[4]) - 1
            bk = m.elem[j].bcs[0][jf]
            if not (bk[0] == 'P' and int(bk[3]) - 1 == gid): badpair += 1
        elif b[0] == 'W': nWc += 1
print(f"[check] readback W={nWc} P={nPc} unreciprocated_P={badpair}")
try:
    m.check_connectivity(); print("[check] check_connectivity: PASS")
except Exception as ex:
    print(f"[check] check_connectivity: {ex}")

GLL = 0.0641   # p=7 GLL: first interior node 0.8717 -> half-gap 0.0641 of element
dy_floor = yn[1] - yn[0]
dy_cubetop = y_abv[1] - y_abv[0]
dx_face = X_CUTS[1] - xn[NX_SIDE-1]
g_floor, g_top, g_face = GLL*dy_floor, GLL*dy_cubetop, GLL*dx_face
print(f"[res] first-element dy_floor={dy_floor:.4f} dy_cubetop={dy_cubetop:.4f} dx_cubeface={dx_face:.4f}")
NU = 1.0/5000.0
for utau in (0.15, 0.25, 0.35):
    print(f"[res] u_tau={utau:.2f} (Re_tau={utau/NU:.0f}): "
          f"y+floor={g_floor*utau/NU:.2f}  y+cubetop={g_top*utau/NU:.2f}  y+cubeface={g_face*utau/NU:.2f}")
dx_mid = np.diff(xn)[NX_SIDE:NX_SIDE+NX_MID].max()
print(f"[res] cube-span max element dx={dx_mid:.4f} -> max in-plane GLL gap ~{0.209*dx_mid:.4f} "
      f"(Dx+ at u_tau=0.25 ~{0.209*dx_mid*0.25/NU:.1f})")
print("[check] OK" if badpair == 0 else "[check] PERIODIC BROKEN")
